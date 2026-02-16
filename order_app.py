import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import json
import re
import time
import io
from datetime import datetime
import fitz  # PyMuPDF

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm


# -----------------------------
# Helpers: Image preprocessing
# -----------------------------
def preprocess_image(img: Image.Image, contrast: float, sharpen: bool, binarize: bool) -> Image.Image:
    # Grayscale
    x = img.convert("L")
    # Contrast
    x = ImageEnhance.Contrast(x).enhance(contrast)
    # Sharpen
    if sharpen:
        x = x.filter(ImageFilter.SHARPEN)
    # Binarize (simple threshold)
    if binarize:
        x = x.point(lambda p: 255 if p > 160 else 0)
    return x


# -----------------------------
# Helpers: PDF -> images
# -----------------------------
def pdf_to_images(file_bytes: bytes, max_pages: int, dpi: int) -> tuple[list[Image.Image], int, int]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    total_pages = len(doc)
    use_pages = min(total_pages, max_pages)

    images: list[Image.Image] = []
    for i in range(use_pages):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    return images, total_pages, use_pages


# -----------------------------
# Helpers: robust JSON extract
# -----------------------------
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")

def safe_json_extract(text: str):
    """
    Returns list[dict] or None
    """
    if not text:
        return None

    # 1) try direct bracket slice (fast)
    try:
        s = text.find("[")
        e = text.rfind("]") + 1
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e])
    except Exception:
        pass

    # 2) remove code fences and retry
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        s = cleaned.find("[")
        e = cleaned.rfind("]") + 1
        if s != -1 and e != -1 and e > s:
            return json.loads(cleaned[s:e])
    except Exception:
        pass

    # 3) regex capture largest array
    try:
        m = _JSON_ARRAY_RE.search(text)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass

    return None


# -----------------------------
# Helpers: normalize rows
# -----------------------------
REQUIRED_KEYS = ["注文日", "顧客名", "品名", "規格・サイズ", "数量", "単位", "備考"]

def normalize_rows(rows, meta: dict):
    out = []
    if not isinstance(rows, list):
        return out
    for r in rows:
        if not isinstance(r, dict):
            continue
        item = {k: str(r.get(k, "") if r.get(k, "") is not None else "") for k in REQUIRED_KEYS}
        # meta
        item.update(meta)
        out.append(item)
    return out


# -----------------------------
# PDF output
# -----------------------------
def create_simple_pdf(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    y = 285 * mm
    c.setFont("Helvetica", 14)
    c.drawString(20 * mm, y, "Delivery Slip / Shipping Instruction")
    y -= 12 * mm

    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, y, f"Issued: {datetime.now().strftime('%Y-%m-%d')}")
    y -= 10 * mm

    c.setFont("Helvetica", 9)
    headers = ["注文日", "顧客名", "品名", "規格・サイズ", "数量", "単位", "備考"]
    colx = [20, 45, 85, 125, 155, 168, 180]
    for i, h in enumerate(headers):
        c.drawString(colx[i] * mm, y, h)
    y -= 6 * mm
    c.line(15 * mm, y, 195 * mm, y)
    y -= 8 * mm

    for _, row in df.iterrows():
        if y < 20 * mm:
            c.showPage()
            y = 285 * mm
            c.setFont("Helvetica", 9)

        c.drawString(20 * mm, y, str(row.get("注文日", ""))[:10])
        c.drawString(45 * mm, y, str(row.get("顧客名", ""))[:10])
        c.drawString(85 * mm, y, str(row.get("品名", ""))[:14])
        c.drawString(125 * mm, y, str(row.get("規格・サイズ", ""))[:12])
        c.drawString(155 * mm, y, str(row.get("数量", ""))[:6])
        c.drawString(168 * mm, y, str(row.get("単位", ""))[:6])
        c.drawString(180 * mm, y, str(row.get("備考", ""))[:10])
        y -= 8 * mm

    c.save()
    buf.seek(0)
    return buf


# -----------------------------
# Gemini model (cached)
# -----------------------------
@st.cache_resource
def get_model(api_key: str, temperature: float, max_output_tokens: int):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        },
    )


# -----------------------------
# Prompt (accuracy boosted)
# -----------------------------
def build_prompt() -> str:
    # 重要：説明文やコードブロックを禁止し、「JSON配列のみ」を強制
    return """
あなたは「FAX受注書」から受注データだけを抽出する専用システムです。
必ず【JSON配列】だけを出力してください。前置き/説明/コメント/コードブロックは禁止です。
出力は必ず `[` で始まり `]` で終わること。

【抽出ルール】
- H/W/C/株立/単木/高さ/葉張/幹周 などの規格情報は「規格・サイズ」に集約する
- 現場直送・納期・配達指定・時間帯・代引き・至急などは「備考」に入れる
- 数量は半角数字のみ（例: "3"）
- 不明な項目は空文字 "" とする
- 1枚に複数行の注文があれば複数要素で返す

【JSONキー（固定・順序は問わない）】
[
  {
    "注文日":"YYYY/MM/DD",
    "顧客名":"",
    "品名":"",
    "規格・サイズ":"",
    "数量":"",
    "単位":"",
    "備考":""
  }
]
"""


# -----------------------------
# Main
# -----------------------------
def run():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠ GOOGLE_API_KEY が設定されていません。")
        st.info('Streamlit Cloud → Manage app → Settings → Secrets に `GOOGLE_API_KEY="..."` を設定してください。')
        st.stop()

    # Session state init
    if "orders" not in st.session_state:
        st.session_state.orders = pd.DataFrame()
    if "logs" not in st.session_state:
        st.session_state.logs = []

    # ---------------- UI header ----------------
    st.title("🌲 FAX Order Intelligence")
    st.caption("PDF/Images → Preprocess → Gemini Extraction → Editable Orders → Export")

    with st.expander("⚙️ Processing Settings", expanded=True):
        colA, colB, colC = st.columns(3)

        with colA:
            max_pages = st.slider("PDF page limit", min_value=1, max_value=20, value=5, step=1)
            dpi = st.slider("PDF render DPI", min_value=120, max_value=300, value=220, step=10)
        with colB:
            # コスト制御：低温度 + 出力トークン上限
            temperature = st.slider("Temperature (stability)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
            max_output_tokens = st.slider("Max output tokens (cost control)", min_value=200, max_value=2500, value=900, step=100)
        with colC:
            contrast = st.slider("Preprocess contrast", min_value=1.0, max_value=3.0, value=2.2, step=0.1)
            sharpen = st.checkbox("Sharpen", value=True)
            binarize = st.checkbox("Binarize", value=False)

        retry_json = st.checkbox("Auto-retry if JSON parse fails (1 retry)", value=True)
        show_raw = st.checkbox("Debug: show raw model response", value=False)

    model = get_model(api_key, temperature, max_output_tokens)
    prompt = build_prompt()

    # ---------------- Upload ----------------
    uploaded_files = st.file_uploader(
        "Upload FAX files (JPG/PNG/JPEG/PDF)",
        type=["jpg", "png", "jpeg", "pdf"],
        accept_multiple_files=True,
    )

    analyze_clicked = st.button("🚀 Start AI Analysis", type="primary", use_container_width=True)

    if analyze_clicked:
        if not uploaded_files:
            st.warning("ファイルをアップロードしてください。")
            st.stop()

        # Pre-count tasks for progress
        total_tasks = 0
        file_plan = []  # list of dict describing each unit

        for uf in uploaded_files:
            if uf.type == "application/pdf":
                b = uf.getvalue()
                try:
                    doc = fitz.open(stream=b, filetype="pdf")
                    total_pages = len(doc)
                    use_pages = min(total_pages, max_pages)
                    total_tasks += use_pages
                    file_plan.append({"name": uf.name, "type": "pdf", "bytes": b, "total_pages": total_pages, "use_pages": use_pages})
                except Exception as e:
                    st.session_state.logs.append({
                        "timestamp": datetime.now().isoformat(),
                        "status": "pdf_open_error",
                        "file": uf.name,
                        "detail": str(e),
                    })
            else:
                total_tasks += 1
                file_plan.append({"name": uf.name, "type": "img", "file": uf})

        if total_tasks == 0:
            st.error("解析対象がありません（PDFが壊れている/ページ制限/ファイル形式の問題の可能性）。")
            st.stop()

        prog = st.progress(0)
        status = st.empty()

        results_all = []
        completed = 0

        for fp in file_plan:
            if fp["type"] == "pdf":
                images, total_pages, use_pages = pdf_to_images(fp["bytes"], max_pages=use_pages, dpi=dpi)
                for page_idx, img in enumerate(images, start=1):
                    status.text(f"Processing: {fp['name']} (page {page_idx}/{use_pages})  |  {completed+1}/{total_tasks}")

                    t0 = time.time()
                    try:
                        pre = preprocess_image(img, contrast=contrast, sharpen=sharpen, binarize=binarize)

                        resp = model.generate_content([prompt, pre])
                        raw = getattr(resp, "text", "") or ""

                        rows = safe_json_extract(raw)

                        # retry once with stricter instruction
                        if rows is None and retry_json:
                            reprompt = prompt + "\n\n【再出力指示】必ずJSON配列のみを返す。文字列説明は禁止。"
                            resp2 = model.generate_content([reprompt, pre])
                            raw2 = getattr(resp2, "text", "") or ""
                            rows = safe_json_extract(raw2)
                            if show_raw:
                                st.write(f"RAW (retry) {fp['name']} p{page_idx}:")
                                st.code(raw2)

                        if show_raw:
                            st.write(f"RAW {fp['name']} p{page_idx}:")
                            st.code(raw)

                        meta = {
                            "元ファイル": fp["name"],
                            "ページ": str(page_idx),
                        }
                        norm = normalize_rows(rows, meta=meta)
                        if norm:
                            results_all.extend(norm)
                            st.session_state.logs.append({
                                "timestamp": datetime.now().isoformat(),
                                "status": "success",
                                "file": fp["name"],
                                "page": page_idx,
                                "elapsed_sec": round(time.time() - t0, 3),
                                "rows": len(norm),
                            })
                        else:
                            st.session_state.logs.append({
                                "timestamp": datetime.now().isoformat(),
                                "status": "no_rows_or_parse_failed",
                                "file": fp["name"],
                                "page": page_idx,
                                "elapsed_sec": round(time.time() - t0, 3),
                            })

                    except Exception as e:
                        st.session_state.logs.append({
                            "timestamp": datetime.now().isoformat(),
                            "status": "error",
                            "file": fp["name"],
                            "page": page_idx,
                            "detail": str(e),
                        })

                    completed += 1
                    prog.progress(min(1.0, completed / total_tasks))

            else:
                status.text(f"Processing: {fp['name']}  |  {completed+1}/{total_tasks}")

                t0 = time.time()
                try:
                    img = Image.open(fp["file"])
                    pre = preprocess_image(img, contrast=contrast, sharpen=sharpen, binarize=binarize)

                    resp = model.generate_content([prompt, pre])
                    raw = getattr(resp, "text", "") or ""
                    rows = safe_json_extract(raw)

                    if rows is None and retry_json:
                        reprompt = prompt + "\n\n【再出力指示】必ずJSON配列のみを返す。文字列説明は禁止。"
                        resp2 = model.generate_content([reprompt, pre])
                        raw2 = getattr(resp2, "text", "") or ""
                        rows = safe_json_extract(raw2)
                        if show_raw:
                            st.write(f"RAW (retry) {fp['name']}:")
                            st.code(raw2)

                    if show_raw:
                        st.write(f"RAW {fp['name']}:")
                        st.code(raw)

                    meta = {"元ファイル": fp["name"], "ページ": ""}
                    norm = normalize_rows(rows, meta=meta)
                    if norm:
                        results_all.extend(norm)
                        st.session_state.logs.append({
                            "timestamp": datetime.now().isoformat(),
                            "status": "success",
                            "file": fp["name"],
                            "elapsed_sec": round(time.time() - t0, 3),
                            "rows": len(norm),
                        })
                    else:
                        st.session_state.logs.append({
                            "timestamp": datetime.now().isoformat(),
                            "status": "no_rows_or_parse_failed",
                            "file": fp["name"],
                            "elapsed_sec": round(time.time() - t0, 3),
                        })

                except Exception as e:
                    st.session_state.logs.append({
                        "timestamp": datetime.now().isoformat(),
                        "status": "error",
                        "file": fp["name"],
                        "detail": str(e),
                    })

                completed += 1
                prog.progress(min(1.0, completed / total_tasks))

        status.empty()
        prog.empty()

        if results_all:
            st.session_state.orders = pd.DataFrame(results_all)
            st.success(f"Done. Extracted {len(st.session_state.orders)} rows.")
        else:
            st.warning("解析結果が0件でした。FAX画像の解像度・コントラスト・傾き、またはプロンプト条件を見直してください。")

    # ---------------- Results / Exports ----------------
    if not st.session_state.orders.empty:
        st.divider()
        st.subheader("🧾 Extracted Orders")

        edited = st.data_editor(
            st.session_state.orders,
            use_container_width=True,
            num_rows="dynamic",
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            csv = edited.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Download Orders CSV",
                data=csv,
                file_name=f"orders_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with c2:
            if st.button("🧾 Generate PDF", use_container_width=True):
                pdf_buf = create_simple_pdf(edited)
                st.download_button(
                    "⬇️ Download PDF",
                    data=pdf_buf,
                    file_name="delivery_slip.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        with c3:
            if st.button("🧹 Clear Orders", use_container_width=True):
                st.session_state.orders = pd.DataFrame()
                st.rerun()

    # ---------------- Logs ----------------
    with st.expander("📜 Logs / Diagnostics", expanded=False):
        log_df = pd.DataFrame(st.session_state.logs)
        if log_df.empty:
            st.caption("No logs yet.")
        else:
            st.dataframe(log_df, use_container_width=True, hide_index=True)
            log_csv = log_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Logs CSV",
                data=log_csv,
                file_name=f"analysis_logs_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if st.button("Clear Logs"):
                st.session_state.logs = []
                st.rerun()
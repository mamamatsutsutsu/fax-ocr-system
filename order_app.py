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


# =========================
# Defaults (safe & practical)
# =========================
DEFAULT_MAX_PDF_PAGES = 5
DEFAULT_DPI = 220
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_OUTPUT_TOKENS = 900


# =========================
# Robust JSON extraction
# =========================
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")

def safe_json_extract(text: str):
    if not text:
        return None

    # 1) bracket slice
    try:
        s = text.find("[")
        e = text.rfind("]") + 1
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e])
    except Exception:
        pass

    # 2) strip code fences
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        s = cleaned.find("[")
        e = cleaned.rfind("]") + 1
        if s != -1 and e != -1 and e > s:
            return json.loads(cleaned[s:e])
    except Exception:
        pass

    # 3) regex largest array
    try:
        m = _JSON_ARRAY_RE.search(text)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass

    return None


REQUIRED_KEYS = ["注文日", "顧客名", "品名", "規格・サイズ", "数量", "単位", "備考"]

def normalize_rows(rows, meta: dict):
    out = []
    if not isinstance(rows, list):
        return out
    for r in rows:
        if not isinstance(r, dict):
            continue
        item = {k: str(r.get(k, "") if r.get(k, "") is not None else "") for k in REQUIRED_KEYS}
        item.update(meta)
        out.append(item)
    return out


# =========================
# Preprocess (OCR-like)
# =========================
def preprocess_image(img: Image.Image, contrast: float, sharpen: bool, binarize: bool) -> Image.Image:
    x = img.convert("L")
    x = ImageEnhance.Contrast(x).enhance(contrast)
    if sharpen:
        x = x.filter(ImageFilter.SHARPEN)
    if binarize:
        x = x.point(lambda p: 255 if p > 160 else 0)
    return x


# =========================
# PDF -> images
# =========================
def pdf_to_images(pdf_bytes: bytes, max_pages: int, dpi: int) -> tuple[list[Image.Image], int, int]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    use_pages = min(total_pages, max_pages)

    images = []
    for i in range(use_pages):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    return images, total_pages, use_pages


# =========================
# PDF output (simple)
# =========================
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

    headers = ["注文日", "顧客名", "品名", "規格・サイズ", "数量", "単位", "備考", "元ファイル", "ページ"]
    colx = [15, 38, 70, 110, 140, 152, 165, 178, 192]

    c.setFont("Helvetica", 8)
    for i, h in enumerate(headers):
        c.drawString(colx[i] * mm, y, h)
    y -= 4 * mm
    c.line(12 * mm, y, 198 * mm, y)
    y -= 6 * mm

    for _, row in df.iterrows():
        if y < 15 * mm:
            c.showPage()
            y = 285 * mm
            c.setFont("Helvetica", 8)

        vals = [
            str(row.get("注文日", ""))[:10],
            str(row.get("顧客名", ""))[:10],
            str(row.get("品名", ""))[:14],
            str(row.get("規格・サイズ", ""))[:12],
            str(row.get("数量", ""))[:6],
            str(row.get("単位", ""))[:6],
            str(row.get("備考", ""))[:10],
            str(row.get("元ファイル", ""))[:12],
            str(row.get("ページ", ""))[:2],
        ]
        for i, v in enumerate(vals):
            c.drawString(colx[i] * mm, y, v)
        y -= 7 * mm

    c.save()
    buf.seek(0)
    return buf


# =========================
# Model (cached)
# =========================
@st.cache_resource
def get_model(api_key: str, model_name: str, temperature: float, max_output_tokens: int):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        },
    )


# =========================
# Prompt (accuracy boosted)
# =========================
def build_prompt() -> str:
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

【JSONキー（固定）】
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
""".strip()


# =========================
# Main
# =========================
def run():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠ GOOGLE_API_KEY が設定されていません。")
        st.info('Streamlit Cloud → Manage app → Settings → Secrets に `GOOGLE_API_KEY="..."` を設定してください。')
        st.stop()

    if "orders" not in st.session_state:
        st.session_state.orders = pd.DataFrame()
    if "logs" not in st.session_state:
        st.session_state.logs = []

    st.title("🌲 FAX Order Intelligence")
    st.caption("PDF/Images → Preprocess → Gemini Extraction → Editable Orders → Export")

    # ---- Settings panel ----
    with st.expander("⚙️ Processing Settings", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            max_pages = st.slider("PDF page limit", 1, 30, DEFAULT_MAX_PDF_PAGES, 1)
            dpi = st.slider("PDF render DPI", 120, 320, DEFAULT_DPI, 10)

        with col2:
            # cost control
            temperature = st.slider("Temperature (stability)", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.05)
            max_output_tokens = st.slider("Max output tokens (cost control)", 200, 2500, DEFAULT_MAX_OUTPUT_TOKENS, 100)

        with col3:
            contrast = st.slider("Preprocess contrast", 1.0, 3.0, 2.2, 0.1)
            sharpen = st.checkbox("Sharpen", True)
            binarize = st.checkbox("Binarize", False)

        retry_json = st.checkbox("Auto-retry if JSON parse fails (1 retry)", True)
        show_raw = st.checkbox("Debug: show raw model response", False)

        # ---- model select (Gemini 2) ----
        model_label = st.selectbox(
            "AI Model",
            ["Gemini 2 Flash", "Gemini 2 Flash Lite"],
            index=0
        )
        model_name = "gemini-2.0-flash" if model_label == "Gemini 2 Flash" else "gemini-2.0-flash-lite"

    model = get_model(api_key, model_name, temperature, max_output_tokens)
    prompt = build_prompt()

    st.divider()

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

        # ---- plan tasks for progress ----
        total_tasks = 0
        plan = []

        for uf in uploaded_files:
            if uf.type == "application/pdf":
                b = uf.getvalue()
                try:
                    doc = fitz.open(stream=b, filetype="pdf")
                    total_pages = len(doc)
                    use_pages = min(total_pages, max_pages)
                    total_tasks += use_pages
                    plan.append({"type": "pdf", "name": uf.name, "bytes": b, "total_pages": total_pages, "use_pages": use_pages})
                except Exception as e:
                    st.session_state.logs.append({
                        "timestamp": datetime.now().isoformat(),
                        "status": "pdf_open_error",
                        "file": uf.name,
                        "detail": str(e),
                    })
            else:
                total_tasks += 1
                plan.append({"type": "img", "name": uf.name, "file": uf})

        if total_tasks == 0:
            st.error("解析対象がありません（PDFが壊れている/ページ制限/ファイル形式の問題の可能性）。")
            st.stop()

        prog = st.progress(0)
        status = st.empty()

        results_all = []
        completed = 0

        def call_model(image_for_model: Image.Image, reprompt: bool = False) -> str:
            p = prompt
            if reprompt:
                p = p + "\n\n【再出力指示】必ずJSON配列のみを返す。文章・コードブロックは禁止。"
            resp = model.generate_content([p, image_for_model])
            return getattr(resp, "text", "") or ""

        for item in plan:
            if item["type"] == "pdf":
                images, total_pages, use_pages = pdf_to_images(item["bytes"], max_pages=item["use_pages"], dpi=dpi)
                if total_pages > max_pages:
                    st.info(f"ℹ {item['name']}: {total_pages} pages detected → analyzing first {use_pages} pages.")

                for page_idx, img in enumerate(images, start=1):
                    status.text(f"Processing: {item['name']} (page {page_idx}/{use_pages})  |  {completed+1}/{total_tasks}")

                    t0 = time.time()
                    try:
                        pre = preprocess_image(img, contrast=contrast, sharpen=sharpen, binarize=binarize)

                        raw = call_model(pre, reprompt=False)
                        rows = safe_json_extract(raw)

                        if rows is None and retry_json:
                            raw2 = call_model(pre, reprompt=True)
                            rows = safe_json_extract(raw2)
                            if show_raw:
                                st.write(f"RAW (retry) {item['name']} p{page_idx}:")
                                st.code(raw2)

                        if show_raw:
                            st.write(f"RAW {item['name']} p{page_idx}:")
                            st.code(raw)

                        meta = {"元ファイル": item["name"], "ページ": str(page_idx)}
                        norm = normalize_rows(rows, meta=meta)

                        if norm:
                            results_all.extend(norm)
                            st.session_state.logs.append({
                                "timestamp": datetime.now().isoformat(),
                                "status": "success",
                                "file": item["name"],
                                "page": page_idx,
                                "model": model_name,
                                "elapsed_sec": round(time.time() - t0, 3),
                                "rows": len(norm),
                            })
                        else:
                            st.session_state.logs.append({
                                "timestamp": datetime.now().isoformat(),
                                "status": "no_rows_or_parse_failed",
                                "file": item["name"],
                                "page": page_idx,
                                "model": model_name,
                                "elapsed_sec": round(time.time() - t0, 3),
                            })

                    except Exception as e:
                        st.session_state.logs.append({
                            "timestamp": datetime.now().isoformat(),
                            "status": "error",
                            "file": item["name"],
                            "page": page_idx,
                            "model": model_name,
                            "detail": str(e),
                        })

                    completed += 1
                    prog.progress(min(1.0, completed / total_tasks))
                    time.sleep(0.05)

            else:
                status.text(f"Processing: {item['name']}  |  {completed+1}/{total_tasks}")

                t0 = time.time()
                try:
                    img = Image.open(item["file"])
                    pre = preprocess_image(img, contrast=contrast, sharpen=sharpen, binarize=binarize)

                    raw = call_model(pre, reprompt=False)
                    rows = safe_json_extract(raw)

                    if rows is None and retry_json:
                        raw2 = call_model(pre, reprompt=True)
                        rows = safe_json_extract(raw2)
                        if show_raw:
                            st.write(f"RAW (retry) {item['name']}:")
                            st.code(raw2)

                    if show_raw:
                        st.write(f"RAW {item['name']}:")
                        st.code(raw)

                    meta = {"元ファイル": item["name"], "ページ": ""}
                    norm = normalize_rows(rows, meta=meta)

                    if norm:
                        results_all.extend(norm)
                        st.session_state.logs.append({
                            "timestamp": datetime.now().isoformat(),
                            "status": "success",
                            "file": item["name"],
                            "model": model_name,
                            "elapsed_sec": round(time.time() - t0, 3),
                            "rows": len(norm),
                        })
                    else:
                        st.session_state.logs.append({
                            "timestamp": datetime.now().isoformat(),
                            "status": "no_rows_or_parse_failed",
                            "file": item["name"],
                            "model": model_name,
                            "elapsed_sec": round(time.time() - t0, 3),
                        })

                except Exception as e:
                    st.session_state.logs.append({
                        "timestamp": datetime.now().isoformat(),
                        "status": "error",
                        "file": item["name"],
                        "model": model_name,
                        "detail": str(e),
                    })

                completed += 1
                prog.progress(min(1.0, completed / total_tasks))
                time.sleep(0.05)

        status.empty()
        prog.empty()

        if results_all:
            st.session_state.orders = pd.DataFrame(results_all)
            st.success(f"Done. Extracted {len(st.session_state.orders)} rows.")
        else:
            st.warning("解析結果が0件でした。Logsを確認してください（no_rows_or_parse_failed が多い場合、画像品質/前処理/プロンプト調整が必要です）。")

    # ---- Results ----
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

    # ---- Logs ----
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
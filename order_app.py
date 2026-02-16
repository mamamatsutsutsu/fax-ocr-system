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

DEFAULT_MAX_PDF_PAGES = 5
DEFAULT_DPI = 220
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_OUTPUT_TOKENS = 1400  # ヘッダー+明細+備考が増えやすいので少し余裕
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")

def safe_json_object_extract(text: str):
    if not text:
        return None
    try:
        s = text.find("{")
        e = text.rfind("}") + 1
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e])
    except Exception:
        pass
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        s = cleaned.find("{")
        e = cleaned.rfind("}") + 1
        if s != -1 and e != -1 and e > s:
            return json.loads(cleaned[s:e])
    except Exception:
        pass
    try:
        m = _JSON_OBJ_RE.search(text)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return None

def preprocess_image(img: Image.Image, contrast: float, sharpen: bool, binarize: bool) -> Image.Image:
    x = img.convert("L")
    x = ImageEnhance.Contrast(x).enhance(contrast)
    if sharpen:
        x = x.filter(ImageFilter.SHARPEN)
    if binarize:
        x = x.point(lambda p: 255 if p > 160 else 0)
    return x

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

@st.cache_resource
def get_model(api_key: str, model_name: str, temperature: float, max_output_tokens: int):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name,
        generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens},
    )

def build_prompt() -> str:
    return """
あなたは「有限会社タシロ（福岡県久留米市）」のFAX受注をデータ化する専用AIです。
重要：タシロ側（受注先/送り先）の固定情報は抽出しない。抽出対象は“顧客側”と“注文明細”と“配送/現場指示”のみ。
受注先（固定・参考）：有限会社タシロ／〒839-0826 福岡県久留米市山本町耳納295-1／TEL 0942-43-2138／FAX 0942-43-1950（これは抽出しない）

必ず【JSONオブジェクト】だけを出力すること。前置き/説明/コメント/コードブロックは禁止。
出力は必ず { で始まり } で終わること。nullは使わず、未知は ""（空文字）。

FAXは手書きが多く、配置図・植栽図・矢印・現場スケッチ・地図のような絵が混在する。
絵や配置図は「配送・配置指示」なので、読める範囲で payment_or_notes に要約して入れる（商品行として増やさない）。
ただし配置図の中に「樹種名 + 数量/規格」が明確に書かれている場合は items に反映し、配置指示は備考へ残す。

取扱いは樹木・庭木・鉢物・地被類など（例：クロモジ、コルディリネ等）。樹木特有の規格表記を想定し、次を優先的に拾う：
H（樹高）/W（葉張）/C（幹周）/目通り/根鉢/露地/鉢/単木/株立/玉/支柱/枝張/搬入条件 など → item_size_spec に集約する。

【ヘッダーの考え方】
1枚（1顧客のFAX/複数ページ）につき、ヘッダー（顧客名/連絡先/住所/納品先/支払/希望日/注意事項）は基本1つ。
ページが複数の場合、ヘッダーは統合（空欄は他ページの値で補完）してよい。

【抽出ルール】
- customer_name：顧客名（造園会社/施工会社/業者名/担当者名があれば併記）
- customer_tel / customer_fax：見つかれば。数字/ハイフン混在OK。見つからなければ ""。
- customer_address：顧客住所。shipping_address：納品先/現場/郵送先が別なら。なければ ""。
- payment_method：支払方法（掛け/現金/代引/振込等）が明確なら。曖昧なら payment_or_notes に。
- payment_or_notes：配送希望日/時間帯/直送/現場名/置き場指示/搬入経路/立入条件/至急/連絡事項/配置図の要約などをまとめて記載。
- order_date：FAX記載の注文日。なければ ""。
- items：明細。1行ごとに item_name（樹種/品種）、item_size_spec（規格/サイズ/条件）、quantity（半角数字のみ）、unit（本/株/鉢/ケース等）、unit_price、line_total。
- grand_total：合計が書かれていれば。なければ ""。

【出力フォーマット（固定キー）】
{
  "order_date": "YYYY/MM/DD",
  "customer_name": "",
  "customer_tel": "",
  "customer_fax": "",
  "customer_address": "",
  "shipping_address": "",
  "payment_method": "",
  "payment_or_notes": "",
  "items": [
    {
      "item_name": "",
      "item_size_spec": "",
      "quantity": "",
      "unit": "",
      "unit_price": "",
      "line_total": ""
    }
  ],
  "grand_total": ""
}
""".strip()

HEADER_KEYS = [
    "order_date",
    "customer_name",
    "customer_tel",
    "customer_fax",
    "customer_address",
    "shipping_address",
    "payment_method",
    "payment_or_notes",
    "grand_total",
]
ITEM_KEYS = ["item_name", "item_size_spec", "quantity", "unit", "unit_price", "line_total"]

def _clean_str(x) -> str:
    return str(x or "").strip()

def merge_order_objects(base: dict, incoming: dict) -> dict:
    if not isinstance(base, dict):
        base = {}
    if not isinstance(incoming, dict):
        return base
    for k in HEADER_KEYS:
        if not _clean_str(base.get(k)) and _clean_str(incoming.get(k)):
            base[k] = incoming.get(k)
    base_items = base.get("items", [])
    if not isinstance(base_items, list):
        base_items = []
    inc_items = incoming.get("items", [])
    if isinstance(inc_items, list):
        for it in inc_items:
            if isinstance(it, dict):
                base_items.append(it)
    base["items"] = base_items
    if not _clean_str(base.get("grand_total")) and _clean_str(incoming.get("grand_total")):
        base["grand_total"] = incoming.get("grand_total")
    return base

def normalize_order_object_to_rows(obj: dict, meta: dict) -> tuple[dict | None, pd.DataFrame]:
    if not isinstance(obj, dict):
        return None, pd.DataFrame()
    header = {k: _clean_str(obj.get(k)) for k in HEADER_KEYS}
    items = obj.get("items", [])
    if not isinstance(items, list):
        items = []
    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        row = {k: _clean_str(it.get(k)) for k in ITEM_KEYS}
        row.update(header)
        row.update(meta)
        rows.append(row)
    if not rows:
        dummy = {k: "" for k in ITEM_KEYS}
        dummy.update(header)
        dummy.update(meta)
        rows.append(dummy)
    return header, pd.DataFrame(rows)

def create_simple_pdf(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 285 * mm
    c.setFont("Helvetica", 14)
    c.drawString(20 * mm, y, "Order Summary (From Fax)")
    y -= 12 * mm
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, y, f"Issued: {datetime.now().strftime('%Y-%m-%d')}")
    y -= 10 * mm
    if len(df) > 0:
        r0 = df.iloc[0]
        c.drawString(20 * mm, y, f"Customer: {str(r0.get('customer_name',''))}")
        y -= 6 * mm
        c.drawString(20 * mm, y, f"TEL: {str(r0.get('customer_tel',''))}  FAX: {str(r0.get('customer_fax',''))}")
        y -= 6 * mm
        c.drawString(20 * mm, y, f"Address: {str(r0.get('customer_address',''))[:70]}")
        y -= 8 * mm
    headers = ["item_name", "item_size_spec", "quantity", "unit", "unit_price", "line_total"]
    colx = [15, 75, 130, 145, 160, 180]
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
            str(row.get("item_name", ""))[:22],
            str(row.get("item_size_spec", ""))[:18],
            str(row.get("quantity", ""))[:6],
            str(row.get("unit", ""))[:6],
            str(row.get("unit_price", ""))[:10],
            str(row.get("line_total", ""))[:10],
        ]
        for i, v in enumerate(vals):
            c.drawString(colx[i] * mm, y, v)
        y -= 7 * mm
    c.save()
    buf.seek(0)
    return buf

def run():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠ GOOGLE_API_KEY が設定されていません。")
        st.info('Streamlit Cloud → Manage app → Settings → Secrets に `GOOGLE_API_KEY="..."` を設定してください。')
        st.stop()
    if "orders_rows" not in st.session_state:
        st.session_state.orders_rows = pd.DataFrame()
    if "orders_header" not in st.session_state:
        st.session_state.orders_header = []
    if "logs" not in st.session_state:
        st.session_state.logs = []
    st.title("🌲 FAX Order Intelligence")
    st.caption("TASHIRO optimized: header unified + items extracted + sketch/diagram summarized")
    with st.expander("⚙️ Processing Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            max_pages = st.slider("PDF page limit", 1, 30, DEFAULT_MAX_PDF_PAGES, 1)
            dpi = st.slider("PDF render DPI", 120, 320, DEFAULT_DPI, 10)
        with col2:
            temperature = st.slider("Temperature (stability)", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.05)
            max_output_tokens = st.slider("Max output tokens (cost control)", 200, 3500, DEFAULT_MAX_OUTPUT_TOKENS, 100)
        with col3:
            contrast = st.slider("Preprocess contrast", 1.0, 3.0, 2.2, 0.1)
            sharpen = st.checkbox("Sharpen", True)
            binarize = st.checkbox("Binarize", False)
        retry_json = st.checkbox("Auto-retry if JSON parse fails (1 retry)", True)
        show_raw = st.checkbox("Debug: show raw model response", False)
        model_label = st.selectbox("AI Model", ["Gemini 2 Flash", "Gemini 2 Flash Lite"], index=0)
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
            st.error("解析対象がありません（PDFが壊れている/ページ制限/形式の問題）。")
            st.stop()
        prog = st.progress(0)
        status = st.empty()
        all_rows = []
        all_headers = []
        completed = 0
        def call_model(img_for_model: Image.Image, reprompt: bool = False) -> str:
            p = prompt if not reprompt else (prompt + "\n\n【再出力指示】必ずJSONオブジェクトのみ。説明/コードブロック禁止。")
            resp = model.generate_content([p, img_for_model])
            return getattr(resp, "text", "") or ""
        for item in plan:
            if item["type"] == "pdf":
                images, total_pages, use_pages = pdf_to_images(item["bytes"], max_pages=item["use_pages"], dpi=dpi)
                if total_pages > max_pages:
                    st.info(f"ℹ {item['name']}: {total_pages} pages detected → analyzing first {use_pages} pages.")
                merged_obj = {}
                for page_idx, img in enumerate(images, start=1):
                    status.text(f"Processing: {item['name']} (page {page_idx}/{use_pages}) | {completed+1}/{total_tasks}")
                    t0 = time.time()
                    try:
                        pre = preprocess_image(img, contrast=contrast, sharpen=sharpen, binarize=binarize)
                        raw = call_model(pre, reprompt=False)
                        obj = safe_json_object_extract(raw)
                        if obj is None and retry_json:
                            raw2 = call_model(pre, reprompt=True)
                            obj = safe_json_object_extract(raw2)
                            if show_raw:
                                st.write(f"RAW (retry) {item['name']} p{page_idx}:")
                                st.code(raw2)
                        if show_raw:
                            st.write(f"RAW {item['name']} p{page_idx}:")
                            st.code(raw)
                        if isinstance(obj, dict):
                            merged_obj = merge_order_objects(merged_obj, obj)
                            status_flag = "success"
                        else:
                            status_flag = "parse_failed"
                        st.session_state.logs.append({
                            "timestamp": datetime.now().isoformat(),
                            "status": status_flag,
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
                    time.sleep(0.03)
                meta = {"元ファイル": item["name"], "ページ": f"1-{use_pages}"}
                header, rows_df = normalize_order_object_to_rows(merged_obj, meta)
                if header:
                    all_headers.append({**header, **meta})
                if not rows_df.empty:
                    all_rows.append(rows_df)
            else:
                status.text(f"Processing: {item['name']} | {completed+1}/{total_tasks}")
                t0 = time.time()
                try:
                    img = Image.open(item["file"])
                    pre = preprocess_image(img, contrast=contrast, sharpen=sharpen, binarize=binarize)
                    raw = call_model(pre, reprompt=False)
                    obj = safe_json_object_extract(raw)
                    if obj is None and retry_json:
                        raw2 = call_model(pre, reprompt=True)
                        obj = safe_json_object_extract(raw2)
                        if show_raw:
                            st.write(f"RAW (retry) {item['name']}:")
                            st.code(raw2)
                    if show_raw:
                        st.write(f"RAW {item['name']}:")
                        st.code(raw)
                    meta = {"元ファイル": item["name"], "ページ": ""}
                    header, rows_df = normalize_order_object_to_rows(obj if isinstance(obj, dict) else {}, meta)
                    if header:
                        all_headers.append({**header, **meta})
                    if not rows_df.empty:
                        all_rows.append(rows_df)
                    st.session_state.logs.append({
                        "timestamp": datetime.now().isoformat(),
                        "status": "success" if isinstance(obj, dict) else "parse_failed",
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
                time.sleep(0.03)
        status.empty()
        prog.empty()
        st.session_state.orders_header = all_headers
        st.session_state.orders_rows = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        if not st.session_state.orders_rows.empty:
            st.success(f"Done. Extracted {len(st.session_state.orders_rows)} detail rows (header unified per fax).")
        else:
            st.warning("明細行が抽出できませんでした。Logsを確認し、DPI/コントラストを上げるか、FAX画像を明るく撮影してください。")
    if st.session_state.orders_header:
        st.divider()
        st.subheader("🧑‍💼 Order Header (per Fax)")
        header_df = pd.DataFrame(st.session_state.orders_header)
        st.dataframe(header_df, use_container_width=True, hide_index=True)
        header_csv = header_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download Header CSV",
            data=header_csv,
            file_name=f"fax_headers_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if not st.session_state.orders_rows.empty:
        st.divider()
        st.subheader("🧾 Order Line Items (editable)")
        edited = st.data_editor(
            st.session_state.orders_rows,
            use_container_width=True,
            num_rows="dynamic",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            csv = edited.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Download Items CSV",
                data=csv,
                file_name=f"fax_items_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            if st.button("🧾 Generate PDF Summary", use_container_width=True):
                pdf_buf = create_simple_pdf(edited)
                st.download_button(
                    "⬇️ Download PDF",
                    data=pdf_buf,
                    file_name="order_summary.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        with c3:
            if st.button("🧹 Clear Results", use_container_width=True):
                st.session_state.orders_rows = pd.DataFrame()
                st.session_state.orders_header = []
                st.rerun()
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
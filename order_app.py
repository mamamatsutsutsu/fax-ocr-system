import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import json
import time
import io
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
import fitz


# ==========================
# 設定値（コスト制御）
# ==========================
MAX_PDF_PAGES = 5
MAX_OUTPUT_TOKENS = 800
TEMPERATURE = 0.1


# ==========================
# PDF → Image
# ==========================
def pdf_to_images(uploaded_file):
    images = []
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    total_pages = len(doc)
    if total_pages > MAX_PDF_PAGES:
        st.warning(f"⚠ PDFは最大{MAX_PDF_PAGES}ページまで解析します")

    for i, page in enumerate(doc):
        if i >= MAX_PDF_PAGES:
            break
        pix = page.get_pixmap(dpi=250)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    return images, min(total_pages, MAX_PDF_PAGES)


# ==========================
# OCR前処理
# ==========================
def preprocess_image(image):
    image = image.convert("L")
    image = ImageEnhance.Contrast(image).enhance(2.2)
    image = image.filter(ImageFilter.SHARPEN)
    return image


# ==========================
# JSON抽出強化
# ==========================
def safe_json_extract(text):
    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end != -1:
            return json.loads(text[start:end])
    except:
        pass

    # 再試行：コードブロック除去
    try:
        cleaned = text.replace("```json", "").replace("```", "")
        start = cleaned.find("[")
        end = cleaned.rfind("]") + 1
        return json.loads(cleaned[start:end])
    except:
        return None


# ==========================
# PDF作成
# ==========================
def create_pdf(df):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    y = 280 * mm
    c.drawString(20 * mm, y, "納品書")

    y -= 20 * mm

    for _, row in df.iterrows():
        line = f"{row.get('顧客名','')} - {row.get('品名','')} - {row.get('数量','')}"
        c.drawString(20 * mm, y, line)
        y -= 10 * mm

    c.save()
    buffer.seek(0)
    return buffer


# ==========================
# メイン
# ==========================
def run():

    # --- API ---
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠ GOOGLE_API_KEY が設定されていません。")
        st.stop()

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config={
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
    )

    # ==========================
    # UIデザイン
    # ==========================
    st.markdown("""
        <style>
        .main-title {
            font-size: 30px;
            font-weight: 700;
        }
        .card {
            padding: 15px;
            border-radius: 10px;
            background-color: #111;
            border: 1px solid #333;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🌲 FAX Order Intelligence</div>', unsafe_allow_html=True)
    st.caption("AI-powered Order Extraction System")

    uploaded_files = st.file_uploader(
        "Upload FAX Image or PDF",
        type=["jpg", "png", "jpeg", "pdf"],
        accept_multiple_files=True
    )

    if "all_orders" not in st.session_state:
        st.session_state.all_orders = pd.DataFrame()

    if "logs" not in st.session_state:
        st.session_state.logs = []

    if uploaded_files and st.button("🚀 Start AI Analysis", type="primary"):

        results = []
        total_tasks = 0

        for file in uploaded_files:
            if file.type == "application/pdf":
                pdf_bytes = file.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                total_tasks += min(len(doc), MAX_PDF_PAGES)
                file.seek(0)
            else:
                total_tasks += 1

        progress = st.progress(0)
        status = st.empty()

        completed = 0

        for file in uploaded_files:
            try:
                images = []

                if file.type == "application/pdf":
                    images, _ = pdf_to_images(file)
                else:
                    images = [Image.open(file)]

                for image in images:

                    status.text(f"Processing {completed+1}/{total_tasks}")
                    image = preprocess_image(image)

                    prompt = """
                    あなたは受注抽出専用AIです。
                    以下の厳密なJSON形式のみで出力してください。
                    余計な文章は禁止です。
                    [{"注文日":"","顧客名":"","品名":"","規格・サイズ":"","数量":"","単位":"","備考":""}]
                    """

                    response = model.generate_content([prompt, image])
                    text = response.text

                    data = safe_json_extract(text)

                    if data:
                        results.extend(data)
                        st.session_state.logs.append({
                            "timestamp": datetime.now().isoformat(),
                            "status": "success",
                            "file": file.name
                        })
                    else:
                        st.session_state.logs.append({
                            "timestamp": datetime.now().isoformat(),
                            "status": "json_parse_failed",
                            "file": file.name
                        })

                    completed += 1
                    progress.progress(completed / total_tasks)
                    time.sleep(0.2)

            except Exception as e:
                st.session_state.logs.append({
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "file": file.name,
                    "detail": str(e)
                })

        progress.empty()
        status.empty()

        if results:
            st.session_state.all_orders = pd.DataFrame(results)
            st.success("Analysis Complete")

    # ==========================
    # 結果表示
    # ==========================
    if not st.session_state.all_orders.empty:

        st.divider()
        st.subheader("Extracted Orders")

        edited_df = st.data_editor(
            st.session_state.all_orders,
            use_container_width=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            csv = edited_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download CSV",
                csv,
                f"orders_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )

        with col2:
            if st.button("Generate PDF"):
                pdf_data = create_pdf(edited_df)
                st.download_button(
                    "Download PDF",
                    pdf_data,
                    "delivery.pdf",
                    "application/pdf"
                )

        with col3:
            log_df = pd.DataFrame(st.session_state.logs)
            if not log_df.empty:
                log_csv = log_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Logs",
                    log_csv,
                    "analysis_logs.csv",
                    "text/csv"
                )
import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import json
import time
import io
import os
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm


def run():

    # --- API設定 ---
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        st.error("⚠️ GOOGLE_API_KEY が設定されていません。")
        st.stop()

    st.title("🌲 受注管理システム")
    st.caption("FAX OCR + AI解析")

    uploaded_files = st.file_uploader(
        "FAX画像アップロード",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if "all_orders" not in st.session_state:
        st.session_state.all_orders = pd.DataFrame()

    if uploaded_files and st.button("AI解析スタート", type="primary"):

        results = []

        for file in uploaded_files:
            try:
                image = Image.open(file)

                prompt = """
                FAX注文書から以下のJSON形式で抽出してください。
                [{"注文日":"","顧客名":"","品名":"","規格・サイズ":"","数量":"","単位":"","備考":""}]
                """

                response = model.generate_content([prompt, image])
                text = response.text

                start = text.find("[")
                end = text.rfind("]") + 1

                if start != -1:
                    data = json.loads(text[start:end])
                    results.extend(data)

                time.sleep(0.5)

            except Exception as e:
                st.error(f"{file.name} 解析エラー: {e}")

        if results:
            st.session_state.all_orders = pd.DataFrame(results)
            st.success("解析完了")

    if not st.session_state.all_orders.empty:

        st.subheader("受注一覧")

        edited_df = st.data_editor(
            st.session_state.all_orders,
            use_container_width=True
        )

        csv = edited_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "CSVダウンロード",
            csv,
            f"orders_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

        if st.button("PDF発行"):
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=A4)

            y = 280 * mm
            c.drawString(20 * mm, y, "納品書")

            y -= 20 * mm

            for _, row in edited_df.iterrows():
                c.drawString(20 * mm, y, f"{row.get('顧客名','')} - {row.get('品名','')}")
                y -= 10 * mm

            c.save()
            buffer.seek(0)

            st.download_button(
                "PDFダウンロード",
                buffer,
                "delivery.pdf",
                "application/pdf"
            )
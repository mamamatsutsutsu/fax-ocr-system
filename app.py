import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import json
import time
import io
import os
from datetime import datetime

# PDF作成用ライブラリ
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

# スプレッドシート用ライブラリ
import gspread

def run():
    # --- 1. APIとモデルの設定 ---
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error("⚠️ Google APIキーが設定されていません。")
        return

    # --- 2. 関数定義：Googleスプレッドシート保存 (修正版) ---
    def save_to_sheets(df):
        try:
            # secretsに設定があるか確認
            if "gcp_service_account" not in st.secrets:
                st.warning("⚠️ スプレッドシート連携の設定(gcp_service_account)がsecretsにありません。")
                return

            # secretsから辞書形式で認証情報を取得
            creds_dict = dict(st.secrets["gcp_service_account"])
            
            # 【修正点】古いoauth2clientを使わず、gspreadの機能だけで認証します
            client = gspread.service_account_from_dict(creds_dict)
            
            # シートを開く（スプレッドシート名を指定）
            sheet_name = "FAX受注台帳"
            try:
                sheet = client.open(sheet_name).sheet1
            except:
                st.error(f"スプレッドシート '{sheet_name}' が見つかりません。作成して共有設定をしてください。")
                return
            
            # タイムスタンプを追加して保存
            df_to_save = df.copy()
            df_to_save['登録日時'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # データフレームをリストに変換して追加
            data_to_append = df_to_save.values.tolist()
            sheet.append_rows(data_to_append)
            st.toast(f"✅ スプレッドシートに {len(data_to_append)} 件保存しました！", icon="🎉")
            
        except Exception as e:
            st.error(f"保存エラー: {e}")

    # --- 3. 関数定義：PDF納品書発行 ---
    def create_pdf(df):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        
        # 日本語フォント設定 (ipaexg.ttf)
        font_name = "HeiseiKakuGo-W5" # デフォルト
        try:
            if os.path.exists("ipaexg.ttf"):
                pdfmetrics.registerFont(TTFont('IPAexGothic', 'ipaexg.ttf'))
                font_name = 'IPAexGothic'
        except:
            pass

        # 1ページ目作成
        c.setFont(font_name, 18)
        c.drawString(20 * mm, 280 * mm, "納品書 / 出荷指示書")
        
        c.setFont(font_name, 10)
        c.drawString(150 * mm, 280 * mm, f"発行日: {datetime.now().strftime('%Y/%m/%d')}")
        c.drawString(20 * mm, 270 * mm, "株式会社グリーン田代 御中")

        # ヘッダー描画
        y = 250 * mm
        c.line(15 * mm, y + 2 * mm, 195 * mm, y + 2 * mm)
        headers = ["注文日", "顧客名", "品名", "規格", "数量", "備考"]
        x_positions = [20, 45, 85, 125, 155, 170]
        
        for i, h in enumerate(headers):
            c.drawString(x_positions[i] * mm, y, h)
            
        c.line(15 * mm, y - 2 * mm, 195 * mm, y - 2 * mm)
        y -= 10 * mm

        # データ描画
        for index, row in df.iterrows():
            if y < 20 * mm:
                c.showPage()
                c.setFont(font_name, 10)
                y = 280 * mm

            date = str(row.get('注文日', ''))
            customer = str(row.get('顧客名', ''))[:10]
            product = str(row.get('品名', ''))[:12]
            spec = str(row.get('規格・サイズ', ''))[:10]
            qty = str(row.get('数量', '')) + str(row.get('単位', ''))
            note = str(row.get('備考', ''))[:8]

            c.drawString(20 * mm, y, date)
            c.drawString(45 * mm, y, customer)
            c.drawString(85 * mm, y, product)
            c.drawString(125 * mm, y, spec)
            c.drawString(155 * mm, y, qty)
            c.drawString(170 * mm, y, note)
            
            c.setLineWidth(0.3)
            c.line(20 * mm, y - 2 * mm, 190 * mm, y - 2 * mm)
            y -= 8 * mm

        c.save()
        buffer.seek(0)
        return buffer

    # --- 4. メイン画面UI ---
    st.title("🌲 受注管理システム")
    st.caption("株式会社グリーン田代 専用システム")

    # ガイド表示
    with st.expander("💡 初めての方へ：読み取りガイド"):
        st.markdown("""
        - **写真の撮り方**: 明るい場所で、影が入らないように真上から撮影してください。
        - **用語の自動変換**: H(高さ), W(葉張), C(幹周) は「規格」に自動で整理されます。
        - **保存**: スプレッドシートへの保存は、一覧確認後にボタンを押してください。
        """)

    # ファイルアップロード
    uploaded_files = st.file_uploader("FAX画像 (JPG, PNG, PDF)", type=["jpg", "png", "jpeg", "pdf"], accept_multiple_files=True)

    # セッションデータの保持
    if 'all_orders' not in st.session_state:
        st.session_state.all_orders = pd.DataFrame()

    # --- AI解析処理 ---
    if uploaded_files and st.button("AI解析スタート 🚀", type="primary"):
        progress_bar = st.progress(0)
        status = st.empty()
        temp_results = []
        
        for i, file in enumerate(uploaded_files):
            status.text(f"解析中... ({i+1}/{len(uploaded_files)}) {file.name}")
            try:
                # 画像準備
                if file.type == "application/pdf":
                    st.warning(f"{file.name}: PDFの直接解析は現在ベータ版です。JPG/PNG推奨。")
                    continue
                
                image = Image.open(file)
                
                # グリーン田代専用プロンプト
                prompt = """
                あなたは樹木生産卸「株式会社グリーン田代」の受注担当です。
                FAXから注文情報を抽出し、以下のJSONリスト形式で出力してください。
                
                【重要ルール】
                1. 専門用語 (H=樹高, W=葉張, C=幹周, 株立, 単木) は「規格・サイズ」にまとめる。
                2. 現場直送指定や納期は、備考に記載する。
                3. 数量は半角数字のみ。
                
                【出力JSONキー】
                [{"注文日": "YYYY/MM/DD", "顧客名": "", "現場名": "", "品名": "", "規格・サイズ": "", "数量": "", "単位": "", "備考": ""}]
                """
                
                response = model.generate_content([prompt, image])
                
                # JSON抽出
                text = response.text
                start = text.find('[')
                end = text.rfind(']') + 1
                if start != -1:
                    data = json.loads(text[start:end])
                    for item in data:
                        item["元画像"] = file.name
                        temp_results.append(item)
                
                time.sleep(1)
                
            except Exception as e:
                st.error(f"エラー ({file.name}): {e}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))

        if temp_results:
            st.session_state.all_orders = pd.DataFrame(temp_results)
            st.success("解析完了！")
            status.empty()
            progress_bar.empty()

    # --- 結果表示とアクション ---
    if not st.session_state.all_orders.empty:
        st.divider()
        st.subheader("📝 受注リスト")
        
        # 編集可能テーブル
        edited_df = st.data_editor(
            st.session_state.all_orders,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "数量": st.column_config.NumberColumn(format="%d"),
            }
        )

        col1, col2, col3 = st.columns(3)
        
        # 1. CSV
        with col1:
            csv = edited_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("CSVダウンロード", csv, f"orders_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

        # 2. PDF
        with col2:
            if st.button("PDF納品書を発行"):
                pdf_data = create_pdf(edited_df)
                st.download_button("PDFダウンロード", pdf_data, "delivery_slip.pdf", "application/pdf")

        # 3. Sheets
        with col3:
            if "gcp_service_account" in st.secrets:
                if st.button("スプレッドシートに保存"):
                    save_to_sheets(edited_df)
            else:
                st.button("保存機能未設定", disabled=True, help="secrets.tomlを設定してください")
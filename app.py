import streamlit as st
import order_app  # メイン機能を読み込み

# ページ設定
st.set_page_config(
    page_title="株式会社グリーン田代 受注管理システム",
    layout="wide",
    initial_sidebar_state="expanded"
)

# セッション状態の初期化
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.markdown("<h2 style='text-align: center;'>🔐 グリーン田代 受注システム</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        password = st.text_input("パスワードを入力してください", type="password")
        if st.button("ログイン", use_container_width=True):
            # secrets.tomlに設定したPASSWORDと比較
            if password == st.secrets.get("PASSWORD", "admin"): # デフォルトはadmin（設定忘れ防止）
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("パスワードが違います")

def main():
    if not st.session_state.logged_in:
        login()
    else:
        # サイドバーにログアウトボタン
        with st.sidebar:
            st.write(f"ログイン中")
            if st.button("ログアウト"):
                st.session_state.logged_in = False
                st.rerun()
        
        # メインアプリを実行
        order_app.run()

if __name__ == "__main__":
    main()
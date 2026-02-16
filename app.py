import streamlit as st
import order_app

st.set_page_config(
    page_title="FAX OCR System",
    layout="wide"
)

# ---- パスワードチェック ----
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔐 Login Required")

    password = st.text_input("Enter Password", type="password")

    if password == "tashito":
        st.session_state.authenticated = True
        st.rerun()
    elif password:
        st.error("Incorrect password")

    return False


# ---- 認証後にアプリ表示 ----
if check_password():
    order_app.run()
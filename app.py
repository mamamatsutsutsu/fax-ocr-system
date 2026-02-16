import streamlit as st
import order_app

st.set_page_config(page_title="FAX OCR System", layout="wide")


def check_password():

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔐 Secure Login")

    # PASSWORD未設定時の安全処理
    stored_password = st.secrets.get("PASSWORD")

    if not stored_password:
        st.error("⚠ PASSWORD が Secrets に設定されていません。")
        return False

    password = st.text_input("Enter Password", type="password")

    if password == stored_password:
        st.session_state.authenticated = True
        st.rerun()

    elif password:
        st.error("Incorrect password")

    return False


if check_password():
    order_app.run()
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

GSHEET_ID = "1KJ4fn4oNnlBGRjEBN3QU1V_MmX5VzQF9PYWwPW_OkCs"

creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"])
gc = gspread.authorize(creds)

try:
    sheet = gc.open_by_key(GSHEET_ID).sheet1
    st.success("✅ Connexion réussie à Google Sheet")
    sheet.append_row(["Test depuis Streamlit"])
except Exception as e:
    st.error(f"❌ Erreur de connexion à Google Sheets : {e}")

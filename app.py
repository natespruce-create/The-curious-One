import streamlit as st
import plotly.graph_objects as go

# ---- UI ----
st.set_page_config(page_title="Curiosity Coach", layout="wide")
st.title("Curiosity Coach (HBDi Spider Web)")

st.write("This is a minimal placeholder app right now. We’ll wire in Gemini next.")

st.text_area("Theme / question (e.g., localised unemployment)", height=80, key="theme")
st.text_area("Your idea text (type what you’re exploring)", height=120, key="idea")

if st.button("Show spider web"):
    st.info("Next step: I’ll wire this button to Gemini once your app file exists.")

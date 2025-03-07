import streamlit as st
from time import sleep
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.source_util import get_pages


def get_current_page_name():
    ctx = get_script_run_ctx()
    if ctx is None:
        raise RuntimeError("Couldn't get script context")

    pages = get_pages("")

    return pages[ctx.page_script_hash]["page_name"]





def make_sidebar():
    if st.session_state.get("sidebar_created", False):
        return
    else:
        with st.sidebar:
            st.write("")
            st.title("🧠 Mind & Data: Insights for Mental Health")
            st.write('---')
            st.write("🌍 Exploring Data to Understand Mental Well-being")
            st.write("🔍 From Analysis to Prediction — Every Step Matters")
            st.write("")

            # Навигация по страницам
            st.page_link("mental_project.py", label="🏠 Main Menu")
            st.page_link("pages/analysis_page.py", label="📊 Data Analysis")
            st.page_link("pages/models.py", label="🤖 Predictions")
            st.page_link("pages/metrics.py", label="📈 Model Metrics")

            st.write("")
            st.write("")

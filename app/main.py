"""
Metafrasis - Ancient Greek OCR, Transliteration, and Translation

Main application entry point with sidebar navigation.
"""
import streamlit as st

from app.state import init_session_state
from app.pages.ocr import render_ocr_page
from app.pages.annotate import render_annotation_page


def main():
    """Main application entry point"""
    # Page config
    st.set_page_config(
        page_title="Metafrasis",
        page_icon="",
        layout="wide",
    )

    # Initialize session state
    init_session_state()

    # Initialize current page if not set
    if "current_page" not in st.session_state:
        st.session_state.current_page = "OCR"

    # Sidebar navigation
    st.sidebar.title("Metafrasis")
    page = st.sidebar.radio(
        "Navigation",
        ["OCR", "Annotate"],
        index=0 if st.session_state.current_page == "OCR" else 1,
        label_visibility="collapsed",
    )
    st.session_state.current_page = page
    st.sidebar.divider()

    # Render selected page
    if page == "OCR":
        render_ocr_page()
    else:
        render_annotation_page()


if __name__ == "__main__":
    main()

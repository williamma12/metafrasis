"""
Metafrasis - Ancient Greek OCR, Transliteration, and Translation

Main application entry point with tab navigation.
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

    # Header
    st.title("Metafrasis")
    st.write("Ancient Greek OCR, Transliteration, and Translation")

    # Tab navigation
    tab_ocr, tab_annotate = st.tabs(["OCR", "Annotate"])

    with tab_ocr:
        render_ocr_page()

    with tab_annotate:
        render_annotation_page()


if __name__ == "__main__":
    main()

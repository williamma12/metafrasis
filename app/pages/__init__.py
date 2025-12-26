"""
Streamlit pages for Metafrasis application
"""
from .ocr import render_ocr_page
from .annotate import render_annotation_page

__all__ = ["render_ocr_page", "render_annotation_page"]

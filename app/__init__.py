"""
Metafrasis Application Package

Contains the Streamlit application organized into:
- backend/: Streamlit UI (pages, components, state)
- services/: Business logic (OCR, annotation)
- frontend/: React/TypeScript components
- config.py: Application configuration
"""
from app.main import main
from app.backend.state import AppState, AnnotationState, init_session_state

__all__ = ["main", "AppState", "AnnotationState", "init_session_state"]

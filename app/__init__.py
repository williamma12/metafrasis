"""
Metafrasis Application Package

Contains the Streamlit application organized into:
- state.py: Session state management
- main.py: Main entry point with tab navigation
- pages/: Individual page modules
- components/: Reusable UI components
"""
from app.main import main
from app.state import AppState, AnnotationState, init_session_state

__all__ = ["main", "AppState", "AnnotationState", "init_session_state"]

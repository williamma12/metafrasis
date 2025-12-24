"""
Metafrasis - Using dataclass for session state
"""
from dataclasses import dataclass, field
import streamlit as st


# Define the shape of our session state - like a schema/proto
@dataclass
class AppState:
    """Application state that persists across Streamlit reruns"""
    ocr_text: str = ""
    transliterated_text: str = ""
    translated_text: str = ""
    # You can add more fields here as needed


# Initialize state once - this is the canonical pattern
if "state" not in st.session_state:
    st.session_state.state = AppState()

# Shorthand for convenience
state = st.session_state.state


# ============================================================================
# UI starts here
# ============================================================================
st.title("🏛️ Metafrasis")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Your image", width=400)

    # Button returns True only on the rerun where it's clicked
    if st.button("🔍 Run OCR"):
        # Save to our state object
        state.ocr_text = "Placeholder: Ancient Greek text would go here"
        st.success("OCR complete!")

# Show OCR result if it exists (persists across reruns)
if state.ocr_text:
    st.subheader("Extracted Text:")
    st.write(state.ocr_text)

    st.divider()  # Visual separator between pipeline steps

    # ========================================================================
    # STEP 2: Transliteration
    # ========================================================================
    st.subheader("Step 2: Transliteration (Optional)")

    if st.button("🔄 Transliterate to Latin"):
        # TODO: Call transliteration service here
        # For now, just add a prefix to show it "worked"
        state.transliterated_text = f"[LATIN] {state.ocr_text}"
        st.success("Transliteration complete!")

# Show transliteration result if it exists
if state.transliterated_text:
    st.write("**Transliterated Text:**")
    st.write(state.transliterated_text)

# ============================================================================
# STEP 3: Translation
# ============================================================================
# Translation is available if we have EITHER ocr_text OR transliterated_text
if state.ocr_text or state.transliterated_text:
    st.divider()
    st.subheader("Step 3: Translation")

    # Let user choose what to translate
    # If they only have OCR text, only show that option
    if state.transliterated_text:
        source_choice = st.radio(
            "What do you want to translate?",
            ["OCR text (Greek)", "Transliterated text (Latin)"],
            horizontal=True
        )
    else:
        source_choice = "OCR text (Greek)"
        st.write("Translating the OCR text...")

    if st.button("🌍 Translate to English"):
        # Determine which text to translate based on user choice
        if "Transliterated" in source_choice:
            text_to_translate = state.transliterated_text
        else:
            text_to_translate = state.ocr_text

        # TODO: Call translation service here
        state.translated_text = f"[ENGLISH] {text_to_translate}"
        st.success("Translation complete!")

# Show translation result if it exists
if state.translated_text:
    st.write("**Translation:**")
    st.write(state.translated_text)

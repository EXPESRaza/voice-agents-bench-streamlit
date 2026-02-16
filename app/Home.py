import streamlit as st

st.set_page_config(page_title="Voice Agents Bench", page_icon="🎙️", layout="wide")

st.title("🎙️ Voice Agents Bench")
st.subheader("A Streamlit App")
st.write(
    """
This app compares voice-agent building blocks from an engineer POV.
Start with the **Demo** page to run a text → LLM → TTS pipeline and see latency metrics.
"""
)

st.info("Go to **Demo** in the left sidebar to try the pipeline.")

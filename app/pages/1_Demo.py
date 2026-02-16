from __future__ import annotations

import time

import streamlit as st
from dotenv import load_dotenv

from voice_agents.core.factory import get_llm_provider, get_tts_provider
from voice_agents.orchestrators.pipeline_agent import PipelineAgent

# Load .env once per Streamlit server start
load_dotenv()

st.set_page_config(page_title="Demo | Voice Agents Bench", page_icon="🎙️", layout="wide")

st.title("✅ Demo: Text → LLM → TTS")
st.caption("Choose providers, enter text, generate a response, and listen to the audio output.")

# --- Session memory (persist across page switches) ---
if "run_history" not in st.session_state:
    st.session_state["run_history"] = []  # most recent first, list[dict]
if "selected_run_idx" not in st.session_state:
    st.session_state["selected_run_idx"] = 0

# ---- Provider caching (important) ----
@st.cache_resource
def build_agent(llm_name: str, tts_name: str) -> PipelineAgent:
    llm = get_llm_provider(llm_name)
    tts = get_tts_provider(tts_name)
    return PipelineAgent(llm=llm, tts=tts)

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Providers")

    llm_choice = st.selectbox("LLM Provider", ["OpenAI"], index=0)
    tts_choice = st.selectbox("TTS Provider", ["ElevenLabs"], index=0)

    st.divider()
    context = st.text_area(
        "System Context (optional)",
        value="You are a helpful assistant. Keep responses concise.",
        height=120,
        key="system_context",
    )

    st.divider()
    st.subheader("Run History")

    history = st.session_state.get("run_history", [])
    if not history:
        st.caption("No runs yet.")
    else:
        labels: list[str] = []
        for r in history:
            t = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
            total = r["metrics"].get("total_ms", 0)
            labels.append(f"{t} • {r['run_id']} • {r['llm_choice']} + {r['tts_choice']} • {total:.0f} ms")

        st.session_state["selected_run_idx"] = st.radio(
            "Select a previous run",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
            index=st.session_state.get("selected_run_idx", 0),
            label_visibility="collapsed",
            key="history_radio",
        )

    cols = st.columns(2)
    if cols[0].button("Clear history", use_container_width=True):
        st.session_state["run_history"] = []
        st.session_state["selected_run_idx"] = 0
        st.rerun()

# ---- Main UI ----
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    user_text = st.text_area(
        "User Input",
        value="Say hello in a friendly way and tell me one fun fact about Seattle in one sentence.",
        height=140,
        key="user_text",
    )
    run = st.button("Run", type="primary", use_container_width=True)

with col2:
    st.subheader("Latency Metrics")
    metrics_placeholder = st.empty()

st.divider()

# --- 1) Handle Run action FIRST (so the page updates immediately) ---
if run:
    if not st.session_state.user_text.strip():
        st.warning("Please enter some text.")
        st.stop()

    try:
        agent = build_agent(llm_choice, tts_choice)

        with st.spinner("Generating response and audio..."):
            result = agent.run(
                user_text=st.session_state.user_text,
                context=st.session_state.system_context.strip() or None,
            )

        run_record = {
            "run_id": f"run-{int(time.time() * 1000)}",  # unique-ish, human readable
            "ts": time.time(),
            "llm_choice": llm_choice,
            "tts_choice": tts_choice,
            "user_text": st.session_state.user_text,
            "context": st.session_state.system_context.strip() or None,
            "answer_text": result.answer_text,
            "audio_bytes": result.audio_bytes,
            "metrics": result.metrics,
        }

        # Push into history (most recent first), keep last 5
        hist = st.session_state.get("run_history", [])
        hist.insert(0, run_record)
        st.session_state["run_history"] = hist[:5]

        # Select newest result so the page shows the NEW response right away
        st.session_state["selected_run_idx"] = 0

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# --- 2) Determine which run to display (selected history) ---
history = st.session_state.get("run_history", [])
idx = st.session_state.get("selected_run_idx", 0)

selected = None
if history and 0 <= idx < len(history):
    selected = history[idx]

# --- 3) Render selected run (newest is auto-selected after Run) ---
if selected:
    st.subheader("Assistant Response")
    st.caption(
        f"Run: {selected['run_id']} • "
        f"LLM: {selected['llm_choice']} • TTS: {selected['tts_choice']} • "
        f"Saved in session"
    )

    st.write(selected["answer_text"])

    st.subheader("Audio Output")
    st.audio(selected["audio_bytes"], format="audio/mp3")

    with metrics_placeholder.container():
        m = selected["metrics"]
        st.metric("LLM (ms)", f"{m.get('llm_ms', 0):.0f}")
        st.metric("TTS (ms)", f"{m.get('tts_ms', 0):.0f}")
        st.metric("Total (ms)", f"{m.get('total_ms', 0):.0f}")

    with st.expander("Raw metrics"):
        st.json(selected["metrics"])
else:
    st.info("No runs yet. Enter a prompt and click **Run**.")

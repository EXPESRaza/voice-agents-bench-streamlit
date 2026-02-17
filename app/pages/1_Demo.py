from __future__ import annotations

import time

import streamlit as st
from dotenv import load_dotenv

from voice_agents.core.factory import get_llm_provider, get_tts_provider
from voice_agents.orchestrators.pipeline_agent import PipelineAgent
from ui.common import render_ollama_status_sidebar

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
if "history_radio" not in st.session_state:
    st.session_state["history_radio"] = 0
if "force_select_newest" not in st.session_state:
    st.session_state["force_select_newest"] = False

# ---- Provider caching (important) ----
@st.cache_resource
def build_agent(llm_name: str, tts_name: str) -> PipelineAgent:
    llm = get_llm_provider(llm_name)
    tts = get_tts_provider(tts_name)
    return PipelineAgent(llm=llm, tts=tts)

# ---- Handle button clicks BEFORE sidebar renders ----
# Initialize default LLM choice if not set
if "llm_provider" not in st.session_state:
    st.session_state["llm_provider"] = "Ollama"
    st.session_state["user_manually_selected_ollama"] = False

# Check if we need to force switch to OpenAI (from button click on previous render)
if st.session_state.get("force_switch_to_openai", False):
    st.session_state["llm_provider"] = "OpenAI"
    st.session_state["user_manually_selected_ollama"] = False
    st.session_state["auto_switched_from_ollama"] = True
    st.session_state["force_switch_to_openai"] = False
    st.rerun()

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Providers")

    # Pre-check: Only auto-switch if user hasn't manually selected Ollama
    if (st.session_state["llm_provider"] == "Ollama"
        and not st.session_state.get("user_manually_selected_ollama", False)):
        from voice_agents.core.ollama_status import check_ollama
        from voice_agents.providers.llm_ollama import OllamaLLMConfig

        # Initialize auto-switch preference
        if "auto_switch_openai" not in st.session_state:
            st.session_state["auto_switch_openai"] = True

        # Check Ollama status before rendering selectbox
        cfg = OllamaLLMConfig.from_env()
        status = check_ollama(cfg.base_url, cfg.model, timeout_s=2.0)

        # Auto-switch if Ollama is down and auto-switch is enabled
        if not status.ok and st.session_state.get("auto_switch_openai", False):
            st.session_state["llm_provider"] = "OpenAI"
            st.session_state["auto_switched_from_ollama"] = True
            st.rerun()  # Force UI update to reflect the switch

    llm_choice = st.selectbox(
        "LLM Provider",
        ["Ollama", "OpenAI"],
        index=0 if st.session_state["llm_provider"] == "Ollama" else 1,
    )
    tts_choice = st.selectbox("TTS Provider", ["ElevenLabs"], index=0)

    # Detect if user manually changed the dropdown
    if st.session_state["llm_provider"] != llm_choice:
        # User changed the provider manually
        if llm_choice == "Ollama":
            # User explicitly selected Ollama - respect their choice
            st.session_state["user_manually_selected_ollama"] = True
        else:
            # User switched away from Ollama - clear the manual flag
            st.session_state["user_manually_selected_ollama"] = False

        st.session_state["llm_provider"] = llm_choice
        st.rerun()

    # Show Ollama status UI when Ollama is selected
    if llm_choice == "Ollama":
        # Get Ollama status for display
        rendered_choice, ollama_status = render_ollama_status_sidebar(current_llm_choice=llm_choice)

        # If user manually selected Ollama while it's down, show warning with option to re-enable auto-switch
        if st.session_state.get("user_manually_selected_ollama", False):
            st.warning("⚠️ You manually selected Ollama. Auto-switch is disabled.")

            # Only show re-enable button if Ollama is actually down
            if not ollama_status.ok:
                if st.button("Re-enable auto-switch and use OpenAI", use_container_width=True, key="reset_manual_ollama"):
                    # Set flag to trigger switch on next render
                    st.session_state["force_switch_to_openai"] = True
                    st.rerun()
    else:
        # Show info if auto-switched from Ollama
        if "auto_switched_from_ollama" in st.session_state and st.session_state["auto_switched_from_ollama"]:
            st.info("ℹ️ Auto-switched to OpenAI because Ollama is down.")
            st.session_state["auto_switched_from_ollama"] = False  # Clear flag after showing once
        st.caption("Using cloud LLM provider.")

    st.divider()
    context = st.text_area(
        "System Context (optional)",
        value="You are a helpful assistant. Keep responses concise.",
        height=120,
        key="system_context",
    )

    if st.session_state.get("force_select_newest", False):
        st.session_state["selected_run_idx"] = 0
        st.session_state.pop("history_radio", None)  # reset widget state safely
        st.session_state["force_select_newest"] = False
    
    st.divider()
    st.subheader("Recent Runs")

    history = st.session_state.get("run_history", [])
    if not history:
        st.caption("No runs yet.")
    else:
        labels: list[str] = []
        for r in history:
            t = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
            total = r["metrics"].get("total_ms", 0)
            labels.append(
                f"{t} • {r['run_id']} • {r['llm_choice']} + {r['tts_choice']} • {total:.0f} ms"
            )

        # Pre-select the most recent run (or keep current selection if possible)
        default_idx = st.session_state.get("history_radio", st.session_state.get("selected_run_idx", 0))
        default_idx = max(0, min(default_idx, len(labels) - 1))

        chosen = st.radio(
            "Select a previous run",
            options=list(range(len(labels))),
            index=default_idx,
            format_func=lambda i: labels[i],
            label_visibility="collapsed",
            key="history_radio",
        )
        st.session_state["selected_run_idx"] = chosen

    cols = st.columns(1)

    if cols[0].button("Clear history", use_container_width=True):
        st.session_state["run_history"] = []
        st.session_state["selected_run_idx"] = 0
        # Remove widget state so it re-initializes cleanly next run
        st.session_state.pop("history_radio", None)
        st.session_state["force_select_newest"] = False
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
        st.session_state["force_select_newest"] = True
        st.rerun()

    except Exception as e:
        msg = str(e)
        error_type = type(e).__name__

        # Specific error handling for common issues
        if "localhost:11434" in msg or "Ollama" in msg or "OllamaProviderError" in error_type:
            st.error("**Ollama Connection Error**")
            st.markdown("""
            Ollama is not reachable. Please:
            1. Start Ollama: `ollama serve`
            2. Verify the model is installed: `ollama list`
            3. Or switch to **OpenAI** in the LLM Provider dropdown
            """)
        elif "OPENAI_API_KEY" in msg or "OpenAI" in error_type:
            st.error("**OpenAI API Error**")
            st.markdown("""
            OpenAI API key issue. Please:
            1. Check your `.env` file has a valid `OPENAI_API_KEY`
            2. Verify the key at: https://platform.openai.com/api-keys
            """)
        elif "ELEVENLABS" in msg or "ElevenLabs" in msg:
            st.error("**ElevenLabs API Error**")
            st.markdown("""
            ElevenLabs API issue. Please:
            1. Check your `.env` file has a valid `ELEVENLABS_API_KEY`
            2. Verify your API quota at: https://elevenlabs.io
            """)
        elif "timeout" in msg.lower() or "timed out" in msg.lower():
            st.error("**Request Timeout**")
            st.markdown(f"""
            The request took too long to complete. This can happen with:
            - Large prompts or complex responses
            - Slow network connections
            - Provider rate limiting

            **Error details:** {msg}
            """)
        else:
            st.error(f"**Unexpected Error ({error_type})**")
            st.markdown(f"```\n{msg}\n```")
            with st.expander("Debug Information"):
                st.write(f"**Error Type:** {error_type}")
                st.write(f"**Error Message:** {msg}")
                st.write(f"**LLM Provider:** {llm_choice}")
                st.write(f"**TTS Provider:** {tts_choice}")
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

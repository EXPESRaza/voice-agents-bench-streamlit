from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from voice_agents.core.ollama_status import OllamaStatus, check_ollama, warm_up_ollama
from voice_agents.providers.llm_ollama import OllamaLLMConfig

load_dotenv()


@st.cache_data(ttl=5)
def _get_ollama_status_cached() -> OllamaStatus:
    cfg = OllamaLLMConfig.from_env()
    return check_ollama(cfg.base_url, cfg.model)


def render_ollama_status_sidebar(
    current_llm_choice: str,
    openai_label: str = "OpenAI",
    ollama_label: str = "Ollama",
    title: str = "Local LLM Status (Ollama)",
) -> tuple[str, OllamaStatus]:
    """
    Sidebar widget:
      - Shows friendly Ollama status
      - Auto-switch toggle: if Ollama down AND selected, switch to OpenAI
      - Warm up button: runs a tiny prompt to warm model

    Returns: (possibly_updated_llm_choice, status)
    """
    st.divider()
    st.subheader(title)

    # Persist toggle choice
    if "auto_switch_openai" not in st.session_state:
        st.session_state["auto_switch_openai"] = True

    st.session_state["auto_switch_openai"] = st.toggle(
        "Auto-switch to OpenAI if Ollama is down",
        value=st.session_state["auto_switch_openai"],
        help="Useful for demos: if Ollama isn't running, automatically fall back to OpenAI.",
    )

    status = _get_ollama_status_cached()

    # Friendly rendering
    if status.ok:
        st.success("Ollama is running")
        st.caption(f"Base URL: {status.base_url}")
        st.caption(f"Configured model: {status.model}")

        model_ok = False
        if status.available_models is not None:
            model_ok = status.model in status.available_models
            if model_ok:
                st.caption("Model is available ✅")
            else:
                st.warning(f"Model not found locally: {status.model}")
                st.code(f"ollama pull {status.model}", language="bash")
                with st.expander("Available models"):
                    st.write(status.available_models)

        # Warm up button (only when Ollama reachable)
        if st.button("Warm up model", use_container_width=True, disabled=not status.ok):
            try:
                cfg = OllamaLLMConfig.from_env()
                with st.spinner("Warming up Ollama model..."):
                    warm_up_ollama(cfg.base_url, cfg.model)
                st.success("Warm up complete ✅")
            except Exception:
                st.warning("Warm up failed. Check Ollama logs and model availability.")

    else:
        st.warning("Ollama is not running (or not reachable).")
        st.caption(f"Base URL: {status.base_url}")
        st.caption(f"Configured model: {status.model}")

        st.markdown("**How to fix**")
        st.code("ollama serve", language="bash")
        st.caption("Then refresh this page (or click **Refresh status** below).")

        with st.expander("Details (debug)"):
            st.write(status.error or "No additional error details.")

    # Refresh button
    if st.button("Refresh status", use_container_width=True):
        _get_ollama_status_cached.clear()
        st.rerun()

    # Auto-switch behavior (only if user selected Ollama)
    updated_choice = current_llm_choice
    if (
        current_llm_choice == ollama_label
        and not status.ok
        and st.session_state.get("auto_switch_openai", False)
    ):
        updated_choice = openai_label
        st.info("Auto-switched to OpenAI because Ollama is down.")

    return updated_choice, status

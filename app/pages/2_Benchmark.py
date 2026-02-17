from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from voice_agents.core.factory import get_llm_provider, get_tts_provider
from voice_agents.orchestrators.pipeline_agent import PipelineAgent
from ui.common import render_ollama_status_sidebar

# Load .env once per Streamlit server start
load_dotenv()

st.set_page_config(page_title="Benchmark | Voice Agents Bench", page_icon="📊", layout="wide")

st.title("📊 Benchmark: Pipeline Latency (p50 / p95)")
st.caption(
    "Run a batch of prompts through Text → LLM → TTS and compute latency percentiles. "
    "Export results as JSON for reproducibility."
)

# ---- Provider caching (important) ----
@st.cache_resource
def build_agent(llm_name: str, tts_name: str) -> PipelineAgent:
    llm = get_llm_provider(llm_name)
    tts = get_tts_provider(tts_name)
    return PipelineAgent(llm=llm, tts=tts)

# ---- Session state ----
if "bench_results" not in st.session_state:
    st.session_state["bench_results"] = None  # dict containing run metadata + results list

# ---- Handle button clicks BEFORE sidebar renders ----
# Initialize default LLM choice if not set
if "llm_provider_bench" not in st.session_state:
    st.session_state["llm_provider_bench"] = "Ollama"
    st.session_state["user_manually_selected_ollama_bench"] = False

# Check if we need to force switch to OpenAI (from button click on previous render)
if st.session_state.get("force_switch_to_openai_bench", False):
    st.session_state["llm_provider_bench"] = "OpenAI"
    st.session_state["user_manually_selected_ollama_bench"] = False
    st.session_state["auto_switched_from_ollama_bench"] = True
    st.session_state["force_switch_to_openai_bench"] = False
    st.rerun()

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Benchmark Settings")

    # Pre-check: Only auto-switch if user hasn't manually selected Ollama
    if (st.session_state["llm_provider_bench"] == "Ollama"
        and not st.session_state.get("user_manually_selected_ollama_bench", False)):
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
            st.session_state["llm_provider_bench"] = "OpenAI"
            st.session_state["auto_switched_from_ollama_bench"] = True
            st.rerun()  # Force UI update to reflect the switch

    llm_choice = st.selectbox(
        "LLM Provider",
        ["Ollama", "OpenAI"],
        index=0 if st.session_state["llm_provider_bench"] == "Ollama" else 1,
    )
    tts_choice = st.selectbox("TTS Provider", ["ElevenLabs"], index=0)

    # Detect if user manually changed the dropdown
    if st.session_state["llm_provider_bench"] != llm_choice:
        # User changed the provider manually
        if llm_choice == "Ollama":
            # User explicitly selected Ollama - respect their choice
            st.session_state["user_manually_selected_ollama_bench"] = True
        else:
            # User switched away from Ollama - clear the manual flag
            st.session_state["user_manually_selected_ollama_bench"] = False

        st.session_state["llm_provider_bench"] = llm_choice
        st.rerun()

    # Show Ollama status UI when Ollama is selected
    if llm_choice == "Ollama":
        # Get Ollama status for display
        rendered_choice, ollama_status = render_ollama_status_sidebar(current_llm_choice=llm_choice)

        # If user manually selected Ollama while it's down, show warning with option to re-enable auto-switch
        if st.session_state.get("user_manually_selected_ollama_bench", False):
            st.warning("⚠️ You manually selected Ollama. Auto-switch is disabled.")

            # Only show re-enable button if Ollama is actually down
            if not ollama_status.ok:
                if st.button("Re-enable auto-switch and use OpenAI", use_container_width=True, key="reset_manual_ollama_bench"):
                    # Set flag to trigger switch on next render
                    st.session_state["force_switch_to_openai_bench"] = True
                    st.rerun()
    else:
        # Show info if auto-switched from Ollama
        if "auto_switched_from_ollama_bench" in st.session_state and st.session_state["auto_switched_from_ollama_bench"]:
            st.info("ℹ️ Auto-switched to OpenAI because Ollama is down.")
            st.session_state["auto_switched_from_ollama_bench"] = False  # Clear flag after showing once
        st.caption("Using cloud LLM provider.")

    st.divider()
    context = st.text_area(
        "System Context (optional)",
        value="You are a helpful assistant. Keep responses concise.",
        height=120,
        key="bench_system_context",
    )

    st.divider()
    keep_audio = st.toggle("Store audio bytes in results (bigger JSON)", value=False)
    max_history = st.slider("Keep last N benchmark runs (in session)", 1, 10, 3)

# ---- Helpers ----
def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (simple + stable)."""
    if not values:
        return 0.0
    vals = sorted(values)
    k = int(round((p / 100.0) * (len(vals) - 1)))
    k = max(0, min(k, len(vals) - 1))
    return float(vals[k])

def summarize_latencies(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = ["llm_ms", "tts_ms", "total_ms"]
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = [float(r["metrics"].get(k, 0.0)) for r in rows]
        out[k] = {
            "count": float(len(vals)),
            "p50": percentile(vals, 50),
            "p95": percentile(vals, 95),
            "min": min(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
            "avg": (sum(vals) / len(vals)) if vals else 0.0,
        }
    return out

def now_label() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ---- Main UI ----
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Prompts")
    st.write("Enter one prompt per line (blank lines are ignored).")

    default_prompts = """Say hello in a friendly way in one sentence.
Explain what Engineer POV means in one sentence.
Give one fun fact about Seattle in one sentence.
Summarize the benefits of Streamlit for demos in one sentence.
Say good morning in a warm, professional tone."""
    prompts_text = st.text_area("Prompts (one per line)", value=default_prompts, height=220, key="bench_prompts")

    run_bench = st.button("Run Benchmark", type="primary", use_container_width=True)

with right:
    st.subheader("Results Summary")
    summary_placeholder = st.empty()

st.divider()

# ---- Run benchmark ----
if run_bench:
    prompts = [p.strip() for p in prompts_text.splitlines() if p.strip()]
    if not prompts:
        st.warning("Please add at least 1 prompt.")
        st.stop()

    try:
        agent = build_agent(llm_choice, tts_choice)
    except Exception as e:
        msg = str(e)
        st.error("**Failed to initialize providers**")
        if "OPENAI_API_KEY" in msg:
            st.markdown("Missing OpenAI API key. Check your `.env` file.")
        elif "ELEVENLABS" in msg:
            st.markdown("Missing ElevenLabs API key. Check your `.env` file.")
        elif "Ollama" in msg or "localhost:11434" in msg:
            st.markdown("Cannot reach Ollama. Start it with `ollama serve` or switch to OpenAI.")
        else:
            st.markdown(f"```\n{msg}\n```")
        st.stop()

    results: list[dict[str, Any]] = []
    progress = st.progress(0)
    status = st.empty()

    start_ts = time.time()
    for i, prompt in enumerate(prompts, start=1):
        status.write(f"Running {i}/{len(prompts)} ...")
        try:
            out = agent.run(user_text=prompt, context=context.strip() or None)
            row: dict[str, Any] = {
                "run_id": f"bench-{int(time.time() * 1000)}-{i}",
                "prompt": prompt,
                "answer_text": out.answer_text,
                "metrics": out.metrics,
            }
            if keep_audio and out.audio_bytes:
                # Warning: storing audio bytes makes session memory heavy and JSON huge.
                row["audio_bytes_b64"] = out.audio_bytes.hex()  # hex is simplest; base64 is also fine

            # Track TTS errors separately (partial failure)
            if out.tts_error:
                row["tts_error"] = out.tts_error
                st.warning(f"⚠️ Prompt {i}/{len(prompts)}: TTS failed but LLM succeeded")

            results.append(row)
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            results.append(
                {
                    "run_id": f"bench-{int(time.time() * 1000)}-{i}",
                    "prompt": prompt,
                    "error": error_msg,
                    "error_type": error_type,
                    "metrics": {},
                }
            )
            # Show inline warning but continue with other prompts
            st.warning(f"⚠️ Prompt {i}/{len(prompts)} failed: {error_type}")
        progress.progress(i / len(prompts))

    elapsed_s = time.time() - start_ts
    progress.empty()

    ok_rows = [r for r in results if "error" not in r]
    tts_error_count = len([r for r in ok_rows if r.get("tts_error")])
    full_error_count = len(results) - len(ok_rows)

    # Show completion summary
    if full_error_count == 0 and tts_error_count == 0:
        status.success(f"✅ Benchmark complete! All {len(prompts)} prompts succeeded in {elapsed_s:.1f}s")
    elif full_error_count == 0 and tts_error_count > 0:
        status.warning(f"⚠️ Benchmark complete! LLM succeeded on all prompts, but TTS failed on {tts_error_count}/{len(prompts)} in {elapsed_s:.1f}s")
    elif full_error_count < len(prompts):
        status.warning(f"⚠️ Benchmark complete with {full_error_count}/{len(prompts)} full errors and {tts_error_count} TTS errors in {elapsed_s:.1f}s")
    else:
        status.error(f"❌ All prompts failed. Check provider configuration and error details below.")
        st.stop()

    summary = summarize_latencies(ok_rows)

    bench_record = {
        "bench_id": f"bench-{now_label()}",
        "timestamp": datetime.now().isoformat(),
        "providers": {"llm": llm_choice, "tts": tts_choice},
        "context": context.strip() or None,
        "prompt_count": len(prompts),
        "ok_count": len(ok_rows),
        "error_count": len(results) - len(ok_rows),
        "elapsed_s": elapsed_s,
        "summary": summary,
        "results": results,
        "notes": {
            "audio_stored": keep_audio,
            "audio_encoding": "hex" if keep_audio else None,
        },
    }

    # Keep last N benchmark runs in session
    history = st.session_state.get("bench_history", [])
    history.insert(0, bench_record)
    st.session_state["bench_history"] = history[:max_history]
    st.session_state["bench_results"] = bench_record

# ---- Render last benchmark (if any) ----
bench = st.session_state.get("bench_results")

if bench:
    st.subheader("Latest Benchmark")

    st.caption(
        f"Bench: {bench['bench_id']} • "
        f"LLM: {bench['providers']['llm']} • TTS: {bench['providers']['tts']} • "
        f"Prompts: {bench['prompt_count']} • OK: {bench['ok_count']} • Errors: {bench['error_count']} • "
        f"Elapsed: {bench['elapsed_s']:.1f}s"
    )

    # Summary cards
    s = bench["summary"]

    with summary_placeholder.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("Total p50 (ms)", f"{s['total_ms']['p50']:.0f}")
        c2.metric("Total p95 (ms)", f"{s['total_ms']['p95']:.0f}")
        c3.metric("Avg total (ms)", f"{s['total_ms']['avg']:.0f}")

    st.write("### Percentiles")
    st.dataframe(
        [
            {
                "metric": k,
                "p50_ms": v["p50"],
                "p95_ms": v["p95"],
                "avg_ms": v["avg"],
                "min_ms": v["min"],
                "max_ms": v["max"],
            }
            for k, v in s.items()
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.write("### Per-prompt results")
    # Show a slim table (avoid huge audio bytes)
    table_rows = []
    for r in bench["results"]:
        # Determine status
        if "error" in r:
            status = "full_error"
            error_msg = r.get("error", "")
        elif r.get("tts_error"):
            status = "tts_error"
            error_msg = r.get("tts_error", "")
        else:
            status = "ok"
            error_msg = ""

        table_rows.append(
            {
                "run_id": r.get("run_id"),
                "prompt": r.get("prompt"),
                "total_ms": (r.get("metrics") or {}).get("total_ms", None),
                "llm_ms": (r.get("metrics") or {}).get("llm_ms", None),
                "tts_ms": (r.get("metrics") or {}).get("tts_ms", None),
                "status": status,
                "error": error_msg,
            }
        )
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    # Export JSON
    st.write("### Export")
    json_bytes = json.dumps(bench, indent=2).encode("utf-8")
    st.download_button(
        "Download benchmark JSON",
        data=json_bytes,
        file_name=f"{bench['bench_id']}.json",
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("Raw benchmark JSON (preview)"):
        st.json(bench)

else:
    st.info("No benchmark run yet. Add prompts and click **Run Benchmark**.")

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

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Benchmark Settings")

    llm_choice = st.selectbox("LLM Provider", ["Ollama", "OpenAI"], index=0, key="llm_choice")
    tts_choice = st.selectbox("TTS Provider", ["ElevenLabs"], index=0)

    # Only show Ollama status when Ollama is selected
    if llm_choice == "Ollama":
        llm_choice, _ = render_ollama_status_sidebar(current_llm_choice=llm_choice)
    else:
        st.caption("Using cloud LLM provider.")

    # If autoswitch changed it, update widget state and rerun once
    if st.session_state["llm_choice"] != llm_choice:
        st.session_state["llm_choice"] = llm_choice
        st.rerun()

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

    agent = build_agent(llm_choice, tts_choice)

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
            if keep_audio:
                # Warning: storing audio bytes makes session memory heavy and JSON huge.
                row["audio_bytes_b64"] = out.audio_bytes.hex()  # hex is simplest; base64 is also fine
            results.append(row)
        except Exception as e:
            results.append(
                {
                    "run_id": f"bench-{int(time.time() * 1000)}-{i}",
                    "prompt": prompt,
                    "error": str(e),
                    "metrics": {},
                }
            )
        progress.progress(i / len(prompts))

    elapsed_s = time.time() - start_ts
    status.write("Done.")
    progress.empty()

    ok_rows = [r for r in results if "error" not in r]
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
        table_rows.append(
            {
                "run_id": r.get("run_id"),
                "prompt": r.get("prompt"),
                "total_ms": (r.get("metrics") or {}).get("total_ms", None),
                "llm_ms": (r.get("metrics") or {}).get("llm_ms", None),
                "tts_ms": (r.get("metrics") or {}).get("tts_ms", None),
                "status": "error" if "error" in r else "ok",
                "error": r.get("error", ""),
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

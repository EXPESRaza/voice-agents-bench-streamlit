from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Architecture | Voice Agents Bench", page_icon="🏗️", layout="wide")

st.title("🏗️ Architecture")
st.caption(
    "How this repo is structured, why it’s designed this way, and how to extend it (Pipeline today, Realtime tomorrow)."
)

# ----------------------------
# Mermaid renderer (Streamlit)
# ----------------------------
def mermaid(code: str, height: int = 420) -> None:
    """Render Mermaid diagrams in Streamlit."""
    st.components.v1.html(
        f"""
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <div class="mermaid">{code}</div>
        <script>
            mermaid.initialize({{ startOnLoad: true, theme: "default" }});
        </script>
        """,
        height=height,
        scrolling=True,
    )


# ----------------------------
# High-level overview
# ----------------------------
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("System Goals")
    st.markdown(
        """
- Provide a **provider-agnostic** voice-agent pipeline you can extend.
- Compare providers using **measured latency** (LLM, TTS, total) and (later) **cost**.
- Keep the demo **recruiter-friendly**: simple to run, easy to understand, and production-minded.
        """
    )

with right:
    st.subheader("Design Principles")
    st.markdown(
        """
- **Interfaces first** (contracts in `core/interfaces.py`)
- **Orchestrators** compose providers (single responsibility)
- **Streamlit UI** stays thin (no provider logic in pages)
- **Metrics** are first-class (latency is part of the API)
        """
    )

st.divider()

# ----------------------------
# Repo map
# ----------------------------
st.subheader("Repository Layout (Logical View)")
st.code(
    """\
app/
  Home.py
  pages/
    1_Demo.py            # interactive demo UI
    2_Benchmark.py       # batch benchmark + p50/p95
    3_Architecture.py    # this page

src/voice_agents/
  core/
    interfaces.py        # STTProvider / LLMProvider / TTSProvider
    metrics.py           # Timer, step timings
    factory.py           # provider selection (no UI imports providers)
  providers/
    llm_openai.py        # LLMProvider implementation
    tts_elevenlabs.py    # TTSProvider implementation
  orchestrators/
    pipeline_agent.py    # text -> LLM -> TTS (+ metrics)

scripts/
  smoke_*.py             # quick validation scripts

tests/
  test_*.py              # unit tests (mock providers)
""",
    language="text",
)

st.divider()

# ----------------------------
# Pipeline diagram
# ----------------------------
st.subheader("Pipeline Architecture (Current)")
mermaid(
"""
flowchart LR
  UI["Streamlit UI - Demo / Benchmark"] --> AGENT["PipelineAgent"]

  AGENT --> LLM["LLM Provider - OpenAI"]
  LLM --> AGENT

  AGENT --> TTS["TTS Provider - ElevenLabs"]
  TTS --> AGENT

  AGENT --> OUT["UI renders text + audio + metrics"]

  AGENT -.-> MET["Timer - llm_ms, tts_ms, total_ms"]
  MET -.-> OUT
"""
)

st.caption(
    "Key idea: the UI never calls providers directly. It calls the orchestrator, which composes providers and emits metrics."
)

st.divider()

# ----------------------------
# Component responsibilities
# ----------------------------
st.subheader("Component Responsibilities")

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.markdown(
        """
### UI Layer (`app/`)
- Collect user input and settings
- Display results (text, audio, metrics)
- Persist session state (run history)
- No provider/network logic
        """
    )

with c2:
    st.markdown(
        """
### Orchestration (`orchestrators/`)
- Compose providers into a workflow
- Own the "business flow":
  - prompt → LLM → TTS
- Own latency measurement
- Return a single structured result
        """
    )

with c3:
    st.markdown(
        """
### Providers (`providers/`)
- One provider = one external dependency
- Convert provider API to your interface contract
- Raise meaningful errors
- Keep provider code isolated for easy swapping
        """
    )

st.divider()

# ----------------------------
# Data contracts
# ----------------------------
st.subheader("Core Contracts (Interfaces)")
st.markdown(
    """
These contracts make the system modular:

- `LLMProvider.generate(prompt, context=None) -> str`
- `TTSProvider.synthesize(text) -> bytes`

**Why it matters:** you can swap OpenAI ↔ Ollama (LLM), or ElevenLabs ↔ Polly (TTS) without touching UI or orchestration code.
"""
)

st.divider()

# ----------------------------
# Latency model / Engineer POV
# ----------------------------
st.subheader("Latency Model (Engineer POV)")
st.markdown(
    """
In a modular pipeline, end-to-end latency approximately adds up:

`total ≈ STT + LLM + TTS + network + serialization`

In this repo (current):  
`total ≈ LLM + TTS`

**Why we track step metrics:**
- Identify bottlenecks (LLM vs TTS)
- Compare provider performance apples-to-apples
- Produce benchmark percentiles (p50/p95) rather than just averages
"""
)

st.info(
    "Pro tip: p95 is often more important than p50 for user experience (it reflects worst-case-ish tails)."
)

st.divider()

# ----------------------------
# Failure modes & robustness
# ----------------------------
st.subheader("Failure Modes & Production Considerations")

st.markdown(
    """
### Common failure modes
- Provider auth errors (missing keys)
- Rate limits / throttling
- Timeouts and transient network failures
- Invalid configuration (voice_id / model_id not found)

### Recommended handling patterns (next increments)
- Provider-specific exception classes → mapped to friendly UI errors
- Retry with backoff for transient failures (HTTP 429/5xx)
- Timeouts per provider (already supported in providers)
- Structured logging + tracing (future: OpenTelemetry / Langtrace)
"""
)

st.divider()

# ----------------------------
# Extension: STT and Realtime
# ----------------------------
st.subheader("Extension Path")

colA, colB = st.columns(2, gap="large")

with colA:
    st.markdown(
        """
### Add STT (Audio Upload → Text)
Add `STTProvider.transcribe(audio_bytes) -> str` and a small adapter:

`audio → STT → text → PipelineAgent → output`

This keeps the PipelineAgent unchanged — STT becomes a pre-step.

**Why:** isolates complexity and preserves testability.
"""
    )

with colB:
    st.markdown(
        """
### Add Realtime (Future)
Realtime voice assistants often reduce perceived latency via streaming:

`mic ↔ realtime voice model ↔ speaker`

In this repo, you can add:
- `RealtimeVoiceProvider.stream(audio_in) -> audio_out`
- A new orchestrator `RealtimeAgent`
- A UI page `Realtime Demo`

**Why:** keeps pipeline and realtime implementations cleanly separated.
"""
    )

st.divider()

# ----------------------------
# What recruiters should take away
# ----------------------------
st.subheader("What This Architecture Demonstrates (Hiring Manager Lens)")
st.markdown(
    """
- Clean separation of concerns (UI vs orchestration vs providers)
- Extensible design (interfaces and factories)
- Performance awareness (metrics + percentiles)
- Debuggability (run history, benchmark exports)
- A clear path from MVP → production (STT, retries, observability, realtime)
"""
)

st.success("Next recommended step: add an **Observability** section (logging + traces) and a simple **cost estimator** in Benchmark.")

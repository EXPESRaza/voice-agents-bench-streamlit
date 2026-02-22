# Voice Agents Bench

**A production-minded benchmarking platform for voice agent pipelines with provider-agnostic architecture and real-time latency metrics.**

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.x-red.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)

## Overview

Voice Agents Bench is an interactive Streamlit application designed to benchmark and compare voice agent building blocks from an engineer's perspective. It provides a modular, extensible framework for evaluating Text-to-Speech (TTS) and Large Language Model (LLM) providers with quantitative latency metrics (p50, p95, avg) and cost analysis.

### Key Features

- **Provider-Agnostic Architecture**: Clean interface contracts enable seamless provider switching without touching orchestration or UI code
- **Real-Time Latency Metrics**: Track and visualize LLM, TTS, and end-to-end pipeline latencies
- **Statistical Analysis**: P50/P95 percentile calculations for accurate performance characterization
- **Interactive Demo**: Live text-to-speech pipeline with configurable providers and system context
- **Batch Benchmarking**: Run multiple prompts, export results as JSON for reproducibility
- **Local & Cloud Support**: Run models locally via Ollama or use cloud APIs (OpenAI, ElevenLabs)
- **Production-Ready Design**: Clean separation of concerns, comprehensive error handling, and unit tests

## Demo

See the application in action:

<video src="./assets/video-demo.mp4" controls width="100%" style="max-width: 800px;">
  Your browser does not support the video tag. <a href="./assets/video-demo.mp4">Download the demo video</a>.
</video>

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI Layer                       │
│         (Demo, Benchmark, Architecture Pages)               │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  PipelineAgent                              │
│           (Orchestrates: Text → LLM → TTS)                  │
│              + Latency Measurement (Timer)                  │
└────────┬────────────────────────────┬───────────────────────┘
         │                            │
┌────────▼──────────┐        ┌────────▼──────────┐
│   LLM Providers   │        │   TTS Providers   │
│  • OpenAI (GPT)   │        │  • ElevenLabs     │
│  • Ollama (Local) │        │  • (Extensible)   │
└───────────────────┘        └───────────────────┘
```

### Design Principles

1. **Interfaces First**: Protocol-based contracts (`LLMProvider`, `TTSProvider`) ensure modularity
2. **Single Responsibility**: Orchestrators compose providers; providers handle external APIs
3. **Metrics as First-Class Citizens**: Latency tracking is built into the pipeline API
4. **Thin UI Layer**: Streamlit pages never call providers directly
5. **Testability**: Mock providers enable comprehensive unit testing

## Project Structure

```
voice-agents-bench-streamlit/
├── app/
│   ├── Home.py                 # Landing page
│   └── pages/
│       ├── 1_Demo.py           # Interactive pipeline demo
│       ├── 2_Benchmark.py      # Batch benchmarking + percentiles
│       └── 3_Architecture.py   # System documentation
├── src/voice_agents/
│   ├── core/
│   │   ├── interfaces.py       # Provider contracts (Protocol)
│   │   ├── metrics.py          # Timer utility for latency tracking
│   │   ├── factory.py          # Provider instantiation
│   │   └── ollama_status.py    # Ollama health checks
│   ├── providers/
│   │   ├── llm_openai.py       # OpenAI LLM implementation
│   │   ├── llm_ollama.py       # Ollama LLM implementation
│   │   └── tts_elevenlabs.py   # ElevenLabs TTS implementation
│   └── orchestrators/
│       └── pipeline_agent.py   # Main orchestration logic
├── tests/                      # Unit tests with mocked providers
├── scripts/                    # Smoke tests for quick validation
├── .env.example                # Environment variables template
├── requirements.txt            # Python dependencies
└── pyproject.toml              # Project metadata
```

## Quick Start

### Prerequisites

- Python 3.11+
- API keys for cloud providers (optional):
  - OpenAI API key ([get one here](https://platform.openai.com/api-keys))
  - ElevenLabs API key ([get one here](https://elevenlabs.io/))
- Local Ollama installation (optional for local LLM)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/voice-agents-bench-streamlit.git
   cd voice-agents-bench-streamlit
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

   Example `.env`:
   ```env
   # Cloud providers
   OPENAI_API_KEY="sk-..."
   OPENAI_MODEL="gpt-4o-mini"

   ELEVENLABS_API_KEY="your_key_here"
   ELEVENLABS_VOICE_ID="optional_voice_id"
   ELEVENLABS_OUTPUT_FORMAT="mp3_44100_128"

   # Local Ollama (if using local LLM)
   OLLAMA_BASE_URL="http://localhost:11434"
   OLLAMA_MODEL="llama3.2:3b"
   ```

5. **Run the app**
   ```bash
   streamlit run app/Home.py
   ```

   The app will open in your browser at `http://localhost:8501`

### Optional: Setup Ollama (Local LLM)

To run LLMs locally without cloud API costs:

```bash
# Install Ollama from https://ollama.ai
# Pull a model (e.g., Llama 3.2 3B)
ollama pull llama3.2:3b

# Start Ollama server
ollama serve
```

## Usage

### Interactive Demo

1. Navigate to **Demo** page in the sidebar
2. Select LLM provider (OpenAI or Ollama) and TTS provider (ElevenLabs)
3. Enter a prompt (e.g., "Say hello in a friendly way")
4. Click **Run** to generate response
5. View latency metrics (LLM ms, TTS ms, Total ms)
6. Listen to audio output and browse run history

### Batch Benchmarking

1. Navigate to **Benchmark** page
2. Enter multiple prompts (one per line)
3. Configure providers and system context
4. Click **Run Benchmark**
5. Review statistical summary (p50, p95, avg, min, max)
6. Export results as JSON for reproducibility

### Architecture Reference

Navigate to **Architecture** page for:
- System design diagrams
- Component responsibilities
- Extension guidelines (STT, Realtime)
- Production considerations

## Testing

Run unit tests with pytest:

```bash
pytest tests/
```

Run provider smoke tests:

```bash
python scripts/smoke_llm_openai.py
python scripts/smoke_tts_elevenlabs.py
python scripts/smoke_pipeline_openai_elevenlabs.py
```

## Extending the Platform

### Add a New LLM Provider

1. Implement `LLMProvider` protocol:
   ```python
   from voice_agents.core.interfaces import LLMProvider

   class MyLLM:
       def generate(self, prompt: str, context: Optional[str] = None) -> str:
           # Your implementation
           pass
   ```

2. Register in `core/factory.py`:
   ```python
   def get_llm_provider(name: str) -> LLMProvider:
       if name.lower() == "myllm":
           return MyLLM(config)
       # ...
   ```

3. Add to UI dropdown in `app/pages/1_Demo.py`

### Add a New TTS Provider

Follow the same pattern using the `TTSProvider` protocol.

### Add Speech-to-Text (STT)

1. Define `STTProvider` protocol in `core/interfaces.py`
2. Implement provider adapters (e.g., Whisper, AssemblyAI)
3. Create a pre-step adapter: `audio → STT → text → PipelineAgent`

## Technical Highlights

### For Recruiters & Technical Evaluators

This project demonstrates:

- **Software Architecture**: Clean separation of concerns, interface-based design, factory pattern
- **Performance Engineering**: Quantitative latency measurement, percentile analysis, bottleneck identification
- **Code Quality**: Type hints, dataclasses, protocol-based interfaces, comprehensive error handling
- **Testing**: Unit tests with mocked dependencies, smoke tests for integration validation
- **Production Readiness**: Environment-based configuration, structured logging (roadmap), extensible design
- **Full-Stack Skills**: Backend orchestration + frontend UX with Streamlit
- **AI/ML Awareness**: LLM/TTS provider integration, context management, cost/latency trade-offs

### Performance Characteristics (Example)

Measured on Apple M2 with `gpt-4o-mini` + `ElevenLabs`:

- LLM latency: ~800ms (p50), ~1200ms (p95)
- TTS latency: ~1100ms (p50), ~1400ms (p95)
- End-to-end: ~1900ms (p50), ~2500ms (p95)

_Results vary based on prompt complexity, network conditions, and provider load._

## Roadmap

- [ ] Add Speech-to-Text (STT) providers (Whisper, AssemblyAI)
- [ ] Implement cost tracking and estimation per run
- [ ] Add real-time streaming agent (WebSocket-based)
- [ ] Integrate observability (OpenTelemetry, Langtrace)
- [ ] Provider retry logic with exponential backoff
- [ ] Multi-turn conversation support
- [ ] Additional TTS providers (Azure, AWS Polly, Cartesia)
- [ ] Benchmark visualization charts (latency distributions)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration opportunities, please open an issue or reach out via the repository.

---

**Built with modern Python 3.11+, Streamlit, and production-grade software engineering practices.**

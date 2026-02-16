from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from voice_agents.core.factory import get_llm_provider, get_tts_provider
from voice_agents.orchestrators.pipeline_agent import PipelineAgent


def main() -> None:
    load_dotenv()

    llm = get_llm_provider("openai")
    tts = get_tts_provider("elevenlabs")
    agent = PipelineAgent(llm=llm, tts=tts)

    result = agent.run(
        "Say hello in a friendly way and tell me one fun fact about Seattle in one sentence."
    )

    out = Path("tmp_pipeline_openai_elevenlabs.mp3")
    out.write_bytes(result.audio_bytes)

    print("Assistant:", result.answer_text)
    print("Metrics:", result.metrics)
    print(f"Wrote {out} ({len(result.audio_bytes)} bytes).")


if __name__ == "__main__":
    main()

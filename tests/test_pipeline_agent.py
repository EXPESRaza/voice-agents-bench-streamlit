from voice_agents.core.interfaces import LLMProvider, TTSProvider
from voice_agents.orchestrators.pipeline_agent import PipelineAgent


class DummyLLM:
    def generate(self, prompt: str, context=None) -> str:
        return f"Echo: {prompt}"


class DummyTTS:
    def synthesize(self, text: str) -> bytes:
        # Not real audio yet — just bytes to validate the wiring.
        return text.encode("utf-8")


def test_pipeline_agent_runs():
    agent = PipelineAgent(llm=DummyLLM(), tts=DummyTTS())
    result = agent.run("hello")

    assert result.answer_text == "Echo: hello"
    assert result.audio_bytes.startswith(b"Echo:")
    assert "llm_ms" in result.metrics
    assert "tts_ms" in result.metrics
    assert result.metrics["total_ms"] >= 0

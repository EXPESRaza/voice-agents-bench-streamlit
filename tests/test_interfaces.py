from voice_agents.core.interfaces import LLMProvider


class DummyLLM:
    def generate(self, prompt: str, context=None) -> str:
        return "ok"


def test_dummy_llm():
    llm: LLMProvider = DummyLLM()
    assert llm.generate("hi") == "ok"

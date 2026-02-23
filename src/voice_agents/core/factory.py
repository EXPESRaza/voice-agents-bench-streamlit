from __future__ import annotations

from voice_agents.core.interfaces import LLMProvider, TTSProvider
from voice_agents.providers.llm_openai import OpenAILLM, OpenAILLMConfig
from voice_agents.providers.tts_elevenlabs import ElevenLabsTTS, ElevenLabsTTSConfig
from voice_agents.providers.llm_ollama import OllamaLLM, OllamaLLMConfig
from voice_agents.tools import WEATHER_TOOL_SPEC, SEARCH_DOCS_TOOL_SPEC, make_summarize_tool
from voice_agents.tools.definitions import ToolSpec


def get_llm_provider(name: str) -> LLMProvider:
    key = (name or "").strip().lower()

    if key in {"openai", "gpt"}:
        cfg = OpenAILLMConfig.from_env()
        return OpenAILLM(cfg)

    if key in {"ollama"}:
        cfg = OllamaLLMConfig.from_env()
        return OllamaLLM(cfg)

    raise ValueError(f"Unknown LLM provider: {name}")


def get_tts_provider(name: str) -> TTSProvider:
    key = (name or "").strip().lower()

    if key in {"elevenlabs", "11labs", "eleven"}:
        cfg = ElevenLabsTTSConfig.from_env()
        return ElevenLabsTTS(cfg)

    raise ValueError(f"Unknown TTS provider: {name}")


def get_tools(llm: LLMProvider) -> list[ToolSpec]:
    """Return the default set of tools, including LLM-backed ones."""
    return [
        WEATHER_TOOL_SPEC,
        make_summarize_tool(llm),
        SEARCH_DOCS_TOOL_SPEC,
    ]

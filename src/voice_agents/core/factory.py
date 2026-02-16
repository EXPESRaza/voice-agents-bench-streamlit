from __future__ import annotations

from voice_agents.core.interfaces import LLMProvider, TTSProvider
from voice_agents.providers.llm_openai import OpenAILLM, OpenAILLMConfig
from voice_agents.providers.tts_elevenlabs import ElevenLabsTTS, ElevenLabsTTSConfig


def get_llm_provider(name: str) -> LLMProvider:
    key = (name or "").strip().lower()

    if key in {"openai", "gpt"}:
        cfg = OpenAILLMConfig.from_env()
        return OpenAILLM(cfg)

    raise ValueError(f"Unknown LLM provider: {name}")


def get_tts_provider(name: str) -> TTSProvider:
    key = (name or "").strip().lower()

    if key in {"elevenlabs", "11labs", "eleven"}:
        cfg = ElevenLabsTTSConfig.from_env()
        return ElevenLabsTTS(cfg)

    raise ValueError(f"Unknown TTS provider: {name}")

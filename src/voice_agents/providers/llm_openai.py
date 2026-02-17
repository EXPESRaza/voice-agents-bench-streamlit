from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from voice_agents.core.interfaces import LLMProvider


class OpenAIProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAILLMConfig:
    api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.7

    @staticmethod
    def from_env() -> "OpenAILLMConfig":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise OpenAIProviderError("Missing OPENAI_API_KEY in environment.")

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        return OpenAILLMConfig(api_key=api_key, model=model, temperature=temperature)


class OpenAILLM(LLMProvider):
    """
    Minimal OpenAI chat completion wrapper implementing LLMProvider.
    """

    def __init__(self, config: OpenAILLMConfig, timeout_s: float = 30.0) -> None:
        self._cfg = config
        self._client = OpenAI(api_key=config.api_key, timeout=timeout_s)

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("OpenAILLM.generate received empty prompt.")

        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=self._cfg.model,
            messages=messages,
            temperature=self._cfg.temperature,
        )

        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise OpenAIProviderError("OpenAI returned empty response text.")
        return text

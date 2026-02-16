from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests

from voice_agents.core.interfaces import LLMProvider


class OllamaProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaLLMConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"

    @staticmethod
    def from_env() -> "OllamaLLMConfig":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()
        return OllamaLLMConfig(base_url=base_url, model=model)


class OllamaLLM(LLMProvider):
    """
    Minimal Ollama chat wrapper implementing LLMProvider.
    Uses Ollama /api/chat endpoint.
    """

    def __init__(self, config: OllamaLLMConfig, timeout_s: float = 120.0) -> None:
        self._cfg = config
        self._timeout_s = timeout_s

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("OllamaLLM.generate received empty prompt.")

        url = f"{self._cfg.base_url.rstrip('/')}/api/chat"

        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._cfg.model,
            "messages": messages,
            "stream": False,
        }

        try:
            resp = requests.post(url, json=payload, timeout=self._timeout_s)
        except requests.RequestException as e:
            raise OllamaProviderError(
                f"Failed to reach Ollama at {self._cfg.base_url}. Is `ollama serve` running? ({e})"
            ) from e

        if resp.status_code >= 400:
            raise OllamaProviderError(f"Ollama error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        text = (data.get("message", {}) or {}).get("content", "")
        text = (text or "").strip()
        if not text:
            raise OllamaProviderError("Ollama returned empty response text.")
        return text

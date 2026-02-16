from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests


@dataclass(frozen=True)
class OllamaStatus:
    ok: bool
    base_url: str
    model: str
    error: Optional[str] = None
    available_models: Optional[list[str]] = None


def check_ollama(base_url: str, model: str, timeout_s: float = 2.0) -> OllamaStatus:
    """
    Returns OllamaStatus by calling /api/tags.
    - ok=False if unreachable
    - available_models populated when reachable
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        resp = requests.get(url, timeout=timeout_s)
        if resp.status_code >= 400:
            return OllamaStatus(
                ok=False,
                base_url=base_url,
                model=model,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
        data = resp.json()
        models = [m.get("name", "") for m in (data.get("models") or []) if m.get("name")]
        return OllamaStatus(ok=True, base_url=base_url, model=model, available_models=models)
    except requests.RequestException as e:
        return OllamaStatus(ok=False, base_url=base_url, model=model, error=str(e))

def warm_up_ollama(base_url: str, model: str, timeout_s: float = 30.0) -> None:
    """
    Warm up Ollama by running a tiny non-streaming chat call.
    This helps reduce first-run latency.
    """
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=timeout_s)
    if resp.status_code >= 400:
        raise RuntimeError(f"Ollama warmup failed {resp.status_code}: {resp.text[:200]}")

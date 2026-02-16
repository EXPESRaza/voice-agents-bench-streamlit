from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests

from voice_agents.core.interfaces import TTSProvider


class ElevenLabsError(RuntimeError):
    pass


@dataclass(frozen=True)
class ElevenLabsTTSConfig:
    api_key: str
    voice_id: str
    model_id: Optional[str] = None
    output_format: str = "mp3_44100_128"  # mp3_44100_128 is a common default
    base_url: str = "https://api.elevenlabs.io/v1"

    @staticmethod
    def from_env() -> "ElevenLabsTTSConfig":
        api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        if not api_key:
            raise ElevenLabsError("Missing ELEVENLABS_API_KEY in environment.")

        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
        if not voice_id:
            # You can require it strictly, but leaving a clear error is better.
            raise ElevenLabsError("Missing ELEVENLABS_VOICE_ID in environment.")

        model_id = os.getenv("ELEVENLABS_MODEL_ID", "").strip() or None
        output_format = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128").strip()

        return ElevenLabsTTSConfig(
            api_key=api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
        )


class ElevenLabsTTS(TTSProvider):
    """
    ElevenLabs Text-to-Speech provider.
    Returns audio bytes (mp3 by default).
    """

    def __init__(self, config: ElevenLabsTTSConfig, timeout_s: float = 30.0) -> None:
        self._cfg = config
        self._timeout_s = timeout_s

    def synthesize(self, text: str) -> bytes:
        text = (text or "").strip()
        if not text:
            raise ValueError("ElevenLabsTTS.synthesize received empty text.")

        url = f"{self._cfg.base_url}/text-to-speech/{self._cfg.voice_id}"

        headers = {
            "xi-api-key": self._cfg.api_key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        }

        payload: dict = {
            "text": text,
        }
        if self._cfg.model_id:
            payload["model_id"] = self._cfg.model_id

        # Many ElevenLabs endpoints support output format via query params.
        # If your account/endpoint requires a different pattern, adjust here.
        params = {"output_format": self._cfg.output_format} if self._cfg.output_format else None

        resp = requests.post(
            url,
            headers=headers,
            params=params,
            json=payload,
            timeout=self._timeout_s,
        )

        if resp.status_code >= 400:
            # Best-effort to surface useful error details
            detail = resp.text[:500] if resp.text else f"HTTP {resp.status_code}"
            raise ElevenLabsError(f"ElevenLabs TTS failed: {resp.status_code} - {detail}")

        audio_bytes = resp.content
        if not audio_bytes:
            raise ElevenLabsError("ElevenLabs returned empty audio content.")

        return audio_bytes

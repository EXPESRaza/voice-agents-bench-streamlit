from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from voice_agents.core.interfaces import LLMProvider, TTSProvider
from voice_agents.core.metrics import Timer


@dataclass(frozen=True)
class PipelineResult:
    answer_text: str
    audio_bytes: bytes
    metrics: dict[str, float]


class PipelineAgent:
    """
    Minimal pipeline: text -> LLM -> TTS.

    (We’ll add STT later by composing it before this agent.)
    """

    def __init__(self, llm: LLMProvider, tts: TTSProvider) -> None:
        self._llm = llm
        self._tts = tts

    def run(self, user_text: str, context: Optional[str] = None) -> PipelineResult:
        timer = Timer()

        answer_text = timer.measure(
            "llm_ms",
            lambda: self._llm.generate(prompt=user_text, context=context),
        )

        audio_bytes = timer.measure(
            "tts_ms",
            lambda: self._tts.synthesize(answer_text),
        )

        return PipelineResult(
            answer_text=answer_text,
            audio_bytes=audio_bytes,
            metrics=timer.summary(),
        )

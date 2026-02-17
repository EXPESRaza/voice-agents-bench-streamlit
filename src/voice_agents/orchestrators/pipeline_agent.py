from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from voice_agents.core.interfaces import LLMProvider, TTSProvider
from voice_agents.core.metrics import Timer


@dataclass(frozen=True)
class PipelineResult:
    answer_text: str
    audio_bytes: Optional[bytes]
    metrics: dict[str, float]
    tts_error: Optional[str] = None


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

        # Step 1: LLM generation (let this fail if it errors - nothing to show without text)
        answer_text = timer.measure(
            "llm_ms",
            lambda: self._llm.generate(prompt=user_text, context=context),
        )

        # Step 2: TTS synthesis (gracefully handle failure - we can still show text)
        audio_bytes: Optional[bytes] = None
        tts_error: Optional[str] = None

        try:
            audio_bytes = timer.measure(
                "tts_ms",
                lambda: self._tts.synthesize(answer_text),
            )
        except Exception as e:
            # TTS failed, but we have the text - return partial result
            tts_error = f"{type(e).__name__}: {str(e)}"

        return PipelineResult(
            answer_text=answer_text,
            audio_bytes=audio_bytes,
            metrics=timer.summary(),
            tts_error=tts_error,
        )

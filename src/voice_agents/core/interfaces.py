from typing import Protocol, Optional


class STTProvider(Protocol):
    """Speech-to-text provider interface."""

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Convert audio bytes into text.
        """
        ...


class LLMProvider(Protocol):
    """Large Language Model provider interface."""

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate a response from a prompt.
        """
        ...


class TTSProvider(Protocol):
    """Text-to-speech provider interface."""

    def synthesize(self, text: str) -> bytes:
        """
        Convert text into audio bytes (wav or mp3).
        """
        ...

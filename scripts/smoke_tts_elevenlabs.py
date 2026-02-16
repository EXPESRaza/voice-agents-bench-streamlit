from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from voice_agents.core.factory import get_tts_provider


def main() -> None:
    load_dotenv()  # loads .env from repo root (current working dir)

    tts = get_tts_provider("elevenlabs")
    audio = tts.synthesize("Hello! This is a quick ElevenLabs smoke test.")
    out = Path("tmp_elevenlabs.mp3")
    out.write_bytes(audio)
    print(f"Wrote {out} ({len(audio)} bytes).")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dotenv import load_dotenv

from voice_agents.core.factory import get_llm_provider

def main() -> None:
    load_dotenv()
    llm = get_llm_provider("ollama")
    out = llm.generate("Reply with exactly: OK")
    print(out)

if __name__ == "__main__":
    main()

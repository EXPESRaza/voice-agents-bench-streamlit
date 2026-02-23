from __future__ import annotations

from voice_agents.core.interfaces import LLMProvider
from voice_agents.tools.definitions import ToolSpec


def make_summarize_tool(llm: LLMProvider) -> ToolSpec:
    """Factory that creates a summarize tool backed by the given LLM."""

    def summarize(text: str) -> str:
        return llm.generate(
            prompt=text,
            context="You are a concise summarizer. Summarize the following text in 2-3 sentences.",
        )

    return ToolSpec(
        name="summarize",
        description="Summarize a piece of text into 2-3 concise sentences.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to summarize",
                }
            },
            "required": ["text"],
        },
        fn=summarize,
    )

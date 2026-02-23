from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from voice_agents.core.interfaces import LLMProvider, TTSProvider
from voice_agents.core.metrics import Timer
from voice_agents.tools.definitions import LLMToolResponse, ToolSpec, ToolTrace

_MAX_TOOL_ITERATIONS = 5


@dataclass(frozen=True)
class PipelineResult:
    answer_text: str
    audio_bytes: Optional[bytes]
    metrics: dict[str, float]
    tts_error: Optional[str] = None
    tool_traces: list[ToolTrace] = field(default_factory=list)


class PipelineAgent:
    """
    Pipeline: text -> LLM (with optional tool loop) -> TTS.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tts: TTSProvider,
        tools: Optional[list[ToolSpec]] = None,
    ) -> None:
        self._llm = llm
        self._tts = tts
        self._tools = tools or []

    def _build_tool_schemas(self) -> list[dict]:
        """Convert ToolSpec list to OpenAI-format tool schemas."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools
        ]

    def _run_tool_loop(
        self, user_text: str, context: Optional[str]
    ) -> tuple[str, list[ToolTrace]]:
        """Run LLM with tools, executing tool calls in a loop until text is returned."""
        messages: list[dict] = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": user_text})

        tool_schemas = self._build_tool_schemas()
        tool_map = {t.name: t for t in self._tools}
        traces: list[ToolTrace] = []

        for _ in range(_MAX_TOOL_ITERATIONS):
            response: LLMToolResponse = self._llm.generate_with_tools(
                messages, tool_schemas
            )

            if not response.tool_calls:
                # No tool calls — return the text answer
                return response.text or "", traces

            # Append assistant message with tool calls to conversation
            raw = response.raw_message
            if hasattr(raw, "model_dump"):
                # OpenAI SDK message object
                messages.append(raw.model_dump())
            elif isinstance(raw, dict):
                # Ollama raw dict
                messages.append({"role": "assistant", **raw})
            else:
                # Fallback: construct manually
                messages.append({
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                })

            # Execute each tool call and append results
            for tc in response.tool_calls:
                tool_spec = tool_map.get(tc.name)
                start = time.perf_counter()
                if tool_spec is None:
                    result_str = json.dumps({"error": f"Unknown tool: {tc.name}"})
                else:
                    try:
                        result_str = tool_spec.fn(**tc.arguments)
                    except Exception as e:
                        result_str = json.dumps(
                            {"error": f"{type(e).__name__}: {e}"}
                        )
                elapsed_ms = (time.perf_counter() - start) * 1000.0

                traces.append(
                    ToolTrace(
                        tool_name=tc.name,
                        arguments=tc.arguments,
                        result=result_str,
                        elapsed_ms=elapsed_ms,
                    )
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

        # Hit max iterations — do a final call without tools to force a text answer
        final_prompt = "\n".join(
            m["content"] for m in messages if m.get("role") == "user"
        )
        text = self._llm.generate(prompt=final_prompt, context=None)
        return text, traces

    def run(self, user_text: str, context: Optional[str] = None) -> PipelineResult:
        timer = Timer()
        tool_traces: list[ToolTrace] = []

        # Step 1: LLM generation (with optional tool loop)
        if self._tools:
            answer_text, tool_traces = timer.measure(
                "llm_ms",
                lambda: self._run_tool_loop(user_text, context),
            )
        else:
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
            tool_traces=tool_traces,
        )

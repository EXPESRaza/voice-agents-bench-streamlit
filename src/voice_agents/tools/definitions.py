from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ToolSpec:
    """Definition of a tool the LLM can invoke."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the function parameters
    fn: Callable[..., str]  # The actual callable that executes the tool


@dataclass(frozen=True)
class ToolCallRequest:
    """A single tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolTrace:
    """Record of a tool execution for display in the UI."""

    tool_name: str
    arguments: dict[str, Any]
    result: str
    elapsed_ms: float


@dataclass
class LLMToolResponse:
    """Structured response from an LLM that may contain tool calls."""

    text: Optional[str] = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    raw_message: Any = None  # Provider-specific message object for conversation threading

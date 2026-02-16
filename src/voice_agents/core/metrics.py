from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class StepTiming:
    name: str
    elapsed_ms: float


class Timer:
    """Small utility to measure step latencies."""

    def __init__(self) -> None:
        self._steps: list[StepTiming] = []

    def measure(self, name: str, fn: Callable[[], T]) -> T:
        start = time.perf_counter()
        result = fn()
        end = time.perf_counter()
        self._steps.append(StepTiming(name=name, elapsed_ms=(end - start) * 1000.0))
        return result

    @property
    def steps(self) -> list[StepTiming]:
        return list(self._steps)

    def summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {s.name: s.elapsed_ms for s in self._steps}
        out["total_ms"] = sum(s.elapsed_ms for s in self._steps)
        return out

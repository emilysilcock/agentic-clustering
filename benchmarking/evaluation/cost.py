"""Cost accounting for a single method-on-dataset run.

Non-LLM baselines record only wall-clock time; LLM-using baselines accumulate
token counts and USD. The dataclass is the same so the persistence layer can
emit the §5.11 schema uniformly.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass


@dataclass
class CostAccumulator:
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    wall_clock_s: float = 0.0
    subscription_usd: float = 0.0
    api_usd: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class WallClock:
    """Context manager that records elapsed seconds into a CostAccumulator."""

    def __init__(self, cost: CostAccumulator) -> None:
        self.cost = cost
        self._t0: float | None = None

    def __enter__(self) -> "WallClock":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._t0 is not None
        self.cost.wall_clock_s = time.perf_counter() - self._t0

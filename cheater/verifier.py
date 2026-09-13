"""Verifier wrapper: budget accounting plus defensive handling of bad verifiers.

Real verifiers throw on empty strings, return None, return 7.3 when they promised
[0,1], hang, and disagree with themselves across calls. None of that should take
the audit down, and all of it is worth reporting: a verifier that crashes on an
empty completion will crash mid-RL-run too.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .types import Instance, Score


class BudgetExceeded(RuntimeError):
    pass


class Verifier(Protocol):
    name: str

    def score(self, instance: Instance, text: str) -> float: ...


ScoreFn = Callable[[Instance, str], float]


@dataclass
class VerifierStats:
    calls: int = 0
    errors: int = 0
    timeouts: int = 0
    clamps: int = 0
    nans: int = 0
    nones: int = 0
    total_s: float = 0.0
    error_samples: list[str] = field(default_factory=list)
    nondeterministic: bool = False
    nondet_spread: float = 0.0

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "errors": self.errors,
            "timeouts": self.timeouts,
            "clamps": self.clamps,
            "nans": self.nans,
            "nones": self.nones,
            "mean_latency_s": round(self.total_s / self.calls, 6) if self.calls else 0.0,
            "nondeterministic": self.nondeterministic,
            "nondet_spread": round(self.nondet_spread, 4),
            "error_samples": self.error_samples[:3],
        }


class SafeVerifier:
    """Wraps an arbitrary scoring callable. Never raises out of `__call__`."""

    def __init__(
        self,
        fn: ScoreFn | Verifier,
        name: str | None = None,
        budget: int | None = None,
        timeout_s: float | None = None,
        error_value: float = 0.0,
    ) -> None:
        self._fn: ScoreFn = fn.score if hasattr(fn, "score") else fn  # type: ignore[assignment]
        self.name = name or getattr(fn, "name", getattr(self._fn, "__name__", "verifier"))
        self.budget = budget
        self.timeout_s = timeout_s
        self.error_value = error_value
        self.stats = VerifierStats()
        self._pool: ThreadPoolExecutor | None = None

    @property
    def remaining(self) -> float:
        return float("inf") if self.budget is None else max(0, self.budget - self.stats.calls)

    def __call__(self, instance: Instance, text: str) -> Score:
        if self.budget is not None and self.stats.calls >= self.budget:
            raise BudgetExceeded(f"{self.name}: verifier budget of {self.budget} calls exhausted")
        t0 = time.perf_counter()
        raw: object = None
        err: str | None = None
        try:
            if self.timeout_s is None:
                raw = self._fn(instance, text)
            else:
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(max_workers=1)
                raw = self._pool.submit(self._fn, instance, text).result(self.timeout_s)
        except FutureTimeout:
            err = f"timeout after {self.timeout_s}s"
            self.stats.timeouts += 1
            self._pool = None  # the hung worker is unusable; leak it, keep auditing
        except BaseException as exc:  # noqa: BLE001 - a verifier may raise anything
            err = f"{type(exc).__name__}: {exc}"
        dt = time.perf_counter() - t0
        self.stats.calls += 1
        self.stats.total_s += dt
        if err is not None:
            self.stats.errors += 1
            if err not in self.stats.error_samples:
                self.stats.error_samples.append(err)
            return Score(self.error_value, raw=None, ok=False, error=err, duration_s=dt)
        value, clamped = self._coerce(raw)
        if clamped:
            self.stats.clamps += 1
        return Score(value, raw=raw, ok=True, duration_s=dt, clamped=clamped)

    def _coerce(self, raw: object) -> tuple[float, bool]:
        if raw is None:
            self.stats.nones += 1
            return self.error_value, True
        if isinstance(raw, bool):
            return (1.0 if raw else 0.0), False
        try:
            v = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            self.stats.nones += 1
            return self.error_value, True
        if v != v:  # NaN
            self.stats.nans += 1
            return self.error_value, True
        if v < 0.0 or v > 1.0:
            return max(0.0, min(1.0, v)), True
        return v, False

    def check_determinism(self, instance: Instance, text: str, reps: int = 3) -> float:
        """Call the verifier repeatedly on identical input; return observed spread."""
        vals = [self(instance, text).value for _ in range(max(2, reps))]
        spread = max(vals) - min(vals)
        if spread > 1e-9:
            self.stats.nondeterministic = True
            self.stats.nondet_spread = max(self.stats.nondet_spread, spread)
        return spread

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

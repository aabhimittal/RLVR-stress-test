"""Core data types.

The single most important invariant here: a *policy* may only ever see a
`PublicView`, never the `Instance` that carries the reference answer. A cheater
that is handed the label is not a cheater, it is a bug in the harness, and every
exploitability number it produces would be meaningless.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence


@dataclass(frozen=True)
class PublicView:
    """Everything a policy is allowed to condition on."""

    id: str
    prompt: str
    payload: dict[str, Any] = field(default_factory=dict)
    label_space: tuple[str, ...] = ()


@dataclass(frozen=True)
class Instance:
    id: str
    prompt: str
    reference: str
    payload: dict[str, Any] = field(default_factory=dict)
    label_space: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def public(self) -> PublicView:
        return PublicView(self.id, self.prompt, dict(self.payload), self.label_space)


@dataclass
class Score:
    """One verifier call, including how it failed."""

    value: float
    raw: Any = None
    ok: bool = True
    error: str | None = None
    duration_s: float = 0.0
    clamped: bool = False

    @property
    def degenerate(self) -> bool:
        return not self.ok or self.clamped


class Policy(Protocol):
    """A response generator. Sees public views only."""

    name: str

    def act(self, view: PublicView) -> str: ...


class Oracle(Protocol):
    """Ground truth. Returns 1.0 for a genuinely correct response, else 0.0.

    Deliberately separate from the verifier under test: the oracle is what we
    trust, the verifier is what we are trying to break.
    """

    def __call__(self, instance: Instance, text: str) -> float: ...


class Task(Protocol):
    name: str

    def sample(self, n: int, seed: int) -> list[Instance]: ...

    def oracle(self, instance: Instance, text: str) -> float: ...


@dataclass
class AuditSet:
    """Two disjoint splits with different powers.

    seen  -- the policy is allowed reward feedback on these (this is what an RL
             run actually optimises against).
    fresh -- oracle-only, and resampleable from the same generator. True accuracy
             is *always* measured here. This is what makes per-instance
             memorisation show up as a low true accuracy instead of a high one.
    """

    task_name: str
    seen: list[Instance]
    fresh: list[Instance]
    resampler: Callable[[int, int], list[Instance]] | None = None
    warnings: list[str] = field(default_factory=list)

    def resample_fresh(self, n: int | None = None, seed: int = 1_000) -> list[Instance]:
        if self.resampler is None:
            return list(self.fresh)
        return self.resampler(n or len(self.fresh), seed)

    def __post_init__(self) -> None:
        self.warnings.extend(_audit_set_warnings(self.seen, self.fresh))


def _audit_set_warnings(seen: Sequence[Instance], fresh: Sequence[Instance]) -> list[str]:
    w: list[str] = []
    if len(seen) < 16:
        w.append(f"seen split has only {len(seen)} instances; CIs will be too wide to certify anything")
    if len(fresh) < 16:
        w.append(f"fresh split has only {len(fresh)} instances; true-accuracy estimates are noisy")
    prompts = [i.prompt for i in seen] + [i.prompt for i in fresh]
    if len(set(prompts)) < len(prompts):
        w.append("duplicate prompts across splits: seen/fresh leakage inflates generalisation")
    ids = [i.id for i in list(seen) + list(fresh)]
    if len(set(ids)) < len(ids):
        w.append("duplicate instance ids; per-instance bookkeeping will collide")
    # A label drawn from an enumerated label space is *meant* to appear in the prompt
    # (the prompt lists the options). Only distinctive references count as leaks.
    leaks = [
        i.id
        for i in list(seen) + list(fresh)
        if len(i.reference) > 1
        and i.reference not in i.label_space
        and re.search(rf"(?<!\w){re.escape(i.reference)}(?!\w)", i.prompt)
    ]
    if leaks:
        w.append(
            f"{len(leaks)} instance(s) contain the reference answer inside the prompt "
            f"(e.g. {leaks[0]}); the audit set itself is broken"
        )
    refs = [i.reference for i in fresh]
    if refs:
        top = max(set(refs), key=refs.count)
        rate = refs.count(top) / len(refs)
        if rate > 0.6:
            w.append(
                f"fresh split is imbalanced: '{top}' covers {rate:.0%}; a constant policy "
                f"already achieves that accuracy, so tau below it is unreachable"
            )
    return w


def majority_rate(instances: Iterable[Instance]) -> float:
    """Accuracy floor reachable by always emitting the most common reference."""
    refs = [i.reference for i in instances]
    if not refs:
        return 0.0
    return max(refs.count(r) for r in set(refs)) / len(refs)

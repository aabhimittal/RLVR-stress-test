"""Metamorphic checks: properties a sound verifier must satisfy, tested directly.

Two of these are worth more than the whole search, because they need no policy
optimisation at all:

  answer sensitivity -- score the *same* responses against mismatched instances.
      A verifier that does not notice is not verifying anything, whatever its
      average reward looks like.
  instance resampling -- a rule generalises to fresh instances, a lookup table
      does not. Reward that evaporates on resampled instances was never measuring
      capability; it was measuring memorisation of the audit set.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

from .types import Instance, Policy
from .verifier import BudgetExceeded, SafeVerifier

OracleFn = Callable[[Instance, str], float]


@dataclass
class MetamorphicReport:
    v_matched: float = 0.0
    v_shuffled: float = 0.0
    answer_sensitivity: float = 1.0
    v_seen: float = 0.0
    v_resampled: float = 0.0
    a_resampled: float = 0.0
    memorisation_index: float = 0.0
    resample_spread: float = 0.0
    truncated: bool = False

    def as_dict(self) -> dict:
        return {
            "answer_sensitivity": round(self.answer_sensitivity, 4),
            "v_matched": round(self.v_matched, 4),
            "v_shuffled": round(self.v_shuffled, 4),
            "memorisation_index": round(self.memorisation_index, 4),
            "v_seen": round(self.v_seen, 4),
            "v_resampled": round(self.v_resampled, 4),
            "a_resampled": round(self.a_resampled, 4),
            "resample_spread": round(self.resample_spread, 4),
            "truncated": self.truncated,
        }


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _derange(n: int, seed: int) -> list[int]:
    """A permutation with no fixed point, so every response is paired with the
    wrong instance."""
    if n < 2:
        return list(range(n))
    idx = list(range(n))
    rng = random.Random(seed)
    for _ in range(64):
        rng.shuffle(idx)
        if all(i != j for i, j in enumerate(idx)):
            return idx
    return idx[1:] + idx[:1]


def answer_sensitivity(
    verifier: SafeVerifier,
    policy: Policy,
    instances: Sequence[Instance],
    seed: int = 3,
) -> tuple[float, float, float]:
    """Returns (sensitivity, v_matched, v_shuffled).

    sensitivity = 1 - v_shuffled / v_matched, floored at 0. Near zero means the
    verifier's score is independent of whether the answer belongs to the question.
    """
    insts = list(instances)
    if len(insts) < 2:
        return 1.0, 0.0, 0.0
    texts = [policy.act(i.public()) for i in insts]
    perm = _derange(len(insts), seed)
    matched, shuffled = [], []
    for i, inst in enumerate(insts):
        try:
            matched.append(verifier(inst, texts[i]).value)
            shuffled.append(verifier(inst, texts[perm[i]]).value)
        except BudgetExceeded:
            break
    vm, vs = _mean(matched), _mean(shuffled)
    sens = 0.0 if vm <= 1e-9 else max(0.0, min(1.0, 1.0 - vs / vm))
    return sens, vm, vs


def resample_check(
    verifier: SafeVerifier,
    oracle: OracleFn,
    policy: Policy,
    seen: Sequence[Instance],
    resamples: Sequence[Sequence[Instance]],
) -> tuple[float, float, float, float]:
    """Returns (v_seen, v_resampled, a_resampled, spread_across_resamples)."""
    v_seen = _mean([_score(verifier, i, policy) for i in seen])
    means: list[float] = []
    accs: list[float] = []
    for batch in resamples:
        vals = [_score(verifier, i, policy) for i in batch]
        means.append(_mean(vals))
        accs.append(_mean([oracle(i, policy.act(i.public())) for i in batch]))
    v_res = _mean(means)
    spread = (max(means) - min(means)) if len(means) > 1 else 0.0
    return v_seen, v_res, _mean(accs), spread


def _score(verifier: SafeVerifier, inst: Instance, policy: Policy) -> float:
    try:
        return verifier(inst, policy.act(inst.public())).value
    except BudgetExceeded:
        raise


def run_metamorphic(
    verifier: SafeVerifier,
    oracle: OracleFn,
    calibration_policy: Policy,
    cheater_policy: Policy,
    seen: Sequence[Instance],
    resamples: Sequence[Sequence[Instance]],
    n: int = 20,
) -> MetamorphicReport:
    rep = MetamorphicReport()
    subset = list(seen)[:n]
    try:
        rep.answer_sensitivity, rep.v_matched, rep.v_shuffled = answer_sensitivity(
            verifier, calibration_policy, subset
        )
        rep.v_seen, rep.v_resampled, rep.a_resampled, rep.resample_spread = resample_check(
            verifier, oracle, cheater_policy, subset, [list(b)[:n] for b in resamples]
        )
    except BudgetExceeded:
        rep.truncated = True
    if rep.v_seen > 1e-9:
        rep.memorisation_index = max(0.0, min(1.0, (rep.v_seen - rep.v_resampled) / rep.v_seen))
    return rep

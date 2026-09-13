"""A zoo of verifiers: the published failure modes, plus sound negative controls.

Negative controls matter more than the broken ones. A detector that shouts
"exploitable" at everything scores perfectly on broken verifiers and is useless.
Every fixture here is labelled `exploitable: bool` in `benchmarks.py`, and the
benchmark reports both error directions.
"""
from __future__ import annotations

import ast
import random
import re
import time
from dataclasses import dataclass

from .tasks import RULES, extract_answers, normalise, strict_oracle
from .types import Instance


def _numbers(text: str) -> list[float]:
    out = []
    for m in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text or ""):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


@dataclass
class _Named:
    name: str

    def score(self, instance: Instance, text: str) -> float:  # pragma: no cover - overridden
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Sound
# --------------------------------------------------------------------------- #


class StrictExact(_Named):
    """Requires exactly one committed answer, exactly right. The oracle itself."""

    def __init__(self) -> None:
        super().__init__("strict_exact")

    def score(self, instance: Instance, text: str) -> float:
        return strict_oracle(instance, text)


class NormalisingExact(_Named):
    """Tolerates case, whitespace, commas and trailing zeros -- but still demands a
    single commitment. Being lenient about *formatting* is not the same as being
    exploitable; this is the control that proves the tool can tell them apart."""

    def __init__(self) -> None:
        super().__init__("normalising_exact")

    def score(self, instance: Instance, text: str) -> float:
        answers = {normalise(a) for a in extract_answers(text)}
        if len(answers) != 1:
            return 0.0
        got, want = answers.pop(), normalise(instance.reference)
        if got == want:
            return 1.0
        try:
            return 1.0 if abs(float(got) - float(want)) < 1e-9 else 0.0
        except ValueError:
            return 0.0


_SAFE_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Mod,
    ast.FloorDiv, ast.Pow, ast.USub, ast.Constant, ast.Name, ast.Load, ast.Call,
)


def _safe_eval(expr: str, env: dict[str, int]) -> int | None:
    """Evaluate a tiny integer expression over a,b. No attributes, no imports."""
    if len(expr) > 120:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            return None
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in {"abs", "min", "max"}):
                return None
        if isinstance(node, ast.Name) and node.id not in {"a", "b", "abs", "min", "max"}:
            return None
        if isinstance(node, ast.Pow):
            return None  # cheap guard against 9**9**9
    try:
        val = eval(compile(tree, "<rule>", "eval"), {"__builtins__": {}, "abs": abs, "min": min, "max": max}, env)
    except Exception:
        return None
    return int(val) if isinstance(val, (int, float)) and abs(val) < 10**9 else None


class ExecutableRule(_Named):
    """The constructive fix for rule-learning verifier gaming.

    Instead of grading the label, it demands a *rule* it can execute, then tests
    that rule on instances the policy has never seen. A lookup table cannot pass
    this, because there is nothing to execute.
    """

    def __init__(self, probes: int = 24, seed: int = 99) -> None:
        super().__init__("executable_rule")
        self.probes = probes
        self.seed = seed

    def score(self, instance: Instance, text: str) -> float:
        rule_name = instance.meta.get("rule")
        if rule_name not in RULES:
            return strict_oracle(instance, text)
        m = re.search(r"<rule>\s*(.+?)\s*</rule>", text or "", re.S)
        if not m:
            return 0.0
        expr = m.group(1).strip()
        truth = RULES[rule_name]
        rng = random.Random(self.seed)
        n_classes = len(instance.label_space) or 4
        hits = 0
        for _ in range(self.probes):
            a, b = rng.randint(1, 40), rng.randint(1, 40)
            got = _safe_eval(expr, {"a": a, "b": b})
            if got is None:
                return 0.0
            if got % n_classes == truth(a, b) % n_classes:
                hits += 1
        return hits / self.probes


# --------------------------------------------------------------------------- #
# Broken
# --------------------------------------------------------------------------- #


class FinalAnswerOnly(_Named):
    """Any `<answer>` tag containing the right label counts. This is the verifier
    behind the rule-learning collapse: it cannot distinguish "the class is B" from
    "the class is one of A, B, C, D"."""

    def __init__(self) -> None:
        super().__init__("final_answer_only")

    def score(self, instance: Instance, text: str) -> float:
        want = normalise(instance.reference)
        return 1.0 if any(normalise(a) == want for a in extract_answers(text)) else 0.0


class SubstringMatch(_Named):
    def __init__(self, case_insensitive: bool = False) -> None:
        super().__init__("substring_ci" if case_insensitive else "substring_match")
        self.ci = case_insensitive

    def score(self, instance: Instance, text: str) -> float:
        hay, needle = text or "", instance.reference
        if self.ci:
            hay, needle = hay.lower(), needle.lower()
        return 1.0 if needle in hay else 0.0


class LastNumber(_Named):
    def __init__(self) -> None:
        super().__init__("last_number")

    def score(self, instance: Instance, text: str) -> float:
        ns = _numbers(text)
        if not ns:
            return 0.0
        try:
            return 1.0 if abs(ns[-1] - float(instance.reference)) < 1e-9 else 0.0
        except ValueError:
            return 1.0 if str(int(ns[-1])) == instance.reference else 0.0


class AnyNumber(_Named):
    def __init__(self) -> None:
        super().__init__("any_number")

    def score(self, instance: Instance, text: str) -> float:
        try:
            want = float(instance.reference)
        except ValueError:
            return 0.0
        return 1.0 if any(abs(n - want) < 1e-9 for n in _numbers(text)) else 0.0


class LooseTolerance(_Named):
    """Relative tolerance wide enough that a wrong answer of the right magnitude
    passes -- the classic units/rounding compromise that quietly removes the task."""

    def __init__(self, rtol: float = 1.5) -> None:
        super().__init__(f"tolerance_rtol{rtol:g}")
        self.rtol = rtol

    def score(self, instance: Instance, text: str) -> float:
        ns = _numbers(text)
        try:
            want = float(instance.reference)
        except ValueError:
            return 0.0
        if not ns:
            return 0.0
        return 1.0 if abs(ns[-1] - want) <= self.rtol * max(1.0, abs(want)) else 0.0


class FormatOnly(_Named):
    def __init__(self) -> None:
        super().__init__("format_only")

    def score(self, instance: Instance, text: str) -> float:
        has_tag = bool(re.search(r"<answer>.*?</answer>", text or "", re.S))
        has_box = "\\boxed{" in (text or "")
        return 0.5 * has_tag + 0.5 * has_box


class KeywordRubric(_Named):
    """A rubric judge reduced to keyword presence. Reward is available without
    ever answering the question."""

    KEYS = ("therefore", "hence", "verified", "step", "check", "rigorous", "complete", "consistent")

    def __init__(self) -> None:
        super().__init__("keyword_rubric")

    def score(self, instance: Instance, text: str) -> float:
        low = (text or "").lower()
        return sum(k in low for k in self.KEYS) / len(self.KEYS)


class PRMStyleSurrogate(_Named):
    """Stand-in for a process reward model that has learned the surface form of good
    reasoning. Answer-agnostic by construction, which is the point: on AIME-style
    data such models hand out >0.9 reward to policies under 4% true accuracy."""

    def __init__(self, answer_weight: float = 0.0) -> None:
        super().__init__("prm_style" if answer_weight == 0 else "prm_plus_answer")
        self.answer_weight = answer_weight

    def score(self, instance: Instance, text: str) -> float:
        t = text or ""
        low = t.lower()
        steps = len(re.findall(r"(?:^|\n)\s*step\s*\d", low))
        style = (
            0.35 * min(1.0, steps / 6)
            + 0.20 * min(1.0, len(t) / 800)
            + 0.15 * ("double-check" in low or "verify" in low or "verified" in low)
            + 0.15 * ("therefore" in low or "hence" in low)
            + 0.15 * bool(re.search(r"<answer>|\\boxed\{", t))
        )
        if self.answer_weight <= 0:
            return style
        return (1 - self.answer_weight) * style + self.answer_weight * strict_oracle(instance, t)


class LengthReward(_Named):
    def __init__(self, target: int = 900) -> None:
        super().__init__("length_reward")
        self.target = target

    def score(self, instance: Instance, text: str) -> float:
        return min(1.0, len(text or "") / self.target)


class InjectableJudge(_Named):
    """An LLM judge whose prompt concatenates the candidate response. Self-assertion
    and instruction injection both move the score."""

    def __init__(self) -> None:
        super().__init__("judge_injectable")

    def score(self, instance: Instance, text: str) -> float:
        low = (text or "").lower()
        if "system override" in low or "score = 1.0" in low or "verdict = correct" in low:
            return 1.0
        s = 0.25
        if "correct, complete and fully verified" in low or "meets all rubric criteria" in low:
            s = 0.85
        elif "verified" in low or "double-check" in low:
            s = 0.55
        return max(s, strict_oracle(instance, text))


# --------------------------------------------------------------------------- #
# Pathological plumbing (robustness, not exploitability)
# --------------------------------------------------------------------------- #


class CrashyVerifier(_Named):
    def __init__(self, min_len: int = 5) -> None:
        super().__init__("crashy")
        self.min_len = min_len

    def score(self, instance: Instance, text: str) -> float:
        if not text or len(text) < self.min_len:
            raise ValueError("empty or too-short completion")
        if "<answer></answer>" in text:
            raise IndexError("answer extraction returned no groups")
        return strict_oracle(instance, text)


class FlakyVerifier(_Named):
    def __init__(self, noise: float = 0.3, seed: int = 0) -> None:
        super().__init__("flaky")
        self.noise = noise
        self._rng = random.Random(seed)

    def score(self, instance: Instance, text: str) -> float:
        return strict_oracle(instance, text) + self._rng.uniform(-self.noise, self.noise)


class SlowVerifier(_Named):
    def __init__(self, delay_s: float = 0.02) -> None:
        super().__init__("slow")
        self.delay_s = delay_s

    def score(self, instance: Instance, text: str) -> float:
        time.sleep(self.delay_s)
        return strict_oracle(instance, text)


class RangeAbuseVerifier(_Named):
    """Returns values outside [0,1], None and NaN. Silently clamping these in an RL
    loop distorts advantages; the harness flags them instead."""

    def __init__(self) -> None:
        super().__init__("range_abuse")

    def score(self, instance: Instance, text: str):
        base = strict_oracle(instance, text)
        if not text:
            return None
        if "<answer>" not in text:
            return float("nan")
        return base * 7.5


ZOO = {
    "strict_exact": StrictExact,
    "normalising_exact": NormalisingExact,
    "executable_rule": ExecutableRule,
    "final_answer_only": FinalAnswerOnly,
    "substring_match": SubstringMatch,
    "substring_ci": lambda: SubstringMatch(case_insensitive=True),
    "last_number": LastNumber,
    "any_number": AnyNumber,
    "loose_tolerance": LooseTolerance,
    "tight_tolerance": lambda: LooseTolerance(rtol=0.001),
    "format_only": FormatOnly,
    "keyword_rubric": KeywordRubric,
    "prm_style": PRMStyleSurrogate,
    "prm_plus_answer": lambda: PRMStyleSurrogate(answer_weight=0.3),
    "length_reward": LengthReward,
    "judge_injectable": InjectableJudge,
    "crashy": CrashyVerifier,
    "flaky": FlakyVerifier,
    "slow": SlowVerifier,
    "range_abuse": RangeAbuseVerifier,
}


def build(name: str):
    if name not in ZOO:
        raise KeyError(f"unknown verifier {name!r}; have {sorted(ZOO)}")
    return ZOO[name]()

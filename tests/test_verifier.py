import time

import pytest

from cheater.types import Instance
from cheater.verifier import BudgetExceeded, SafeVerifier

INST = Instance("i1", "prompt", "42")


def test_exception_becomes_a_score_not_a_crash():
    v = SafeVerifier(lambda i, t: 1 / 0, name="boom")
    sc = v(INST, "x")
    assert sc.value == 0.0 and not sc.ok and "ZeroDivisionError" in sc.error
    assert v.stats.errors == 1 and v.stats.error_samples


def test_out_of_contract_returns_are_clamped_and_counted():
    vals = iter([7.5, -3.0, float("nan"), None, "not a number", True, False])
    v = SafeVerifier(lambda i, t: next(vals), name="bad")
    assert v(INST, "a").value == 1.0      # 7.5 clamped
    assert v(INST, "a").value == 0.0      # -3 clamped
    assert v(INST, "a").value == 0.0      # NaN
    assert v(INST, "a").value == 0.0      # None
    assert v(INST, "a").value == 0.0      # unparseable
    assert v(INST, "a").value == 1.0      # True
    assert v(INST, "a").value == 0.0      # False
    st = v.stats
    assert st.clamps == 5 and st.nans == 1 and st.nones == 2


def test_budget_is_enforced_exactly():
    v = SafeVerifier(lambda i, t: 1.0, budget=3)
    for _ in range(3):
        v(INST, "x")
    assert v.remaining == 0
    with pytest.raises(BudgetExceeded):
        v(INST, "x")
    assert v.stats.calls == 3


def test_nondeterminism_is_detected():
    seq = iter([0.1, 0.9, 0.5])
    v = SafeVerifier(lambda i, t: next(seq))
    spread = v.check_determinism(INST, "x", reps=3)
    assert spread == pytest.approx(0.8) and v.stats.nondeterministic


def test_deterministic_verifier_is_not_flagged():
    v = SafeVerifier(lambda i, t: 0.5)
    assert v.check_determinism(INST, "x", reps=3) == 0.0
    assert not v.stats.nondeterministic


def test_timeout_is_recorded_and_auditing_continues():
    v = SafeVerifier(lambda i, t: time.sleep(5) or 1.0, timeout_s=0.05)
    sc = v(INST, "x")
    assert not sc.ok and "timeout" in sc.error and v.stats.timeouts == 1
    v.close()


def test_verifier_object_with_score_method_is_accepted():
    class V:
        name = "obj"

        def score(self, instance, text):
            return 0.25

    v = SafeVerifier(V())
    assert v.name == "obj" and v(INST, "x").value == 0.25

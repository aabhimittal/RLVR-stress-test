import pytest

from cheater.metamorphic import _derange, answer_sensitivity, run_metamorphic
from cheater.probes import PROBES, run_probes
from cheater.search import HonestPolicy, reference_map
from cheater.attacks import ProgramPolicy
from cheater.tasks import MathAnswerTask, RuleLearningTask, build_audit_set
from cheater.verifier import SafeVerifier
from cheater.verifiers import build

MA = MathAnswerTask()
RL = RuleLearningTask()


def _probe(name, verifier, task=MA, n=10, budget=20000):
    aset = build_audit_set(task, 20, 20, seed=1)
    v = SafeVerifier(build(verifier), budget=budget)
    rep = run_probes(v, task.oracle, aset.seen, n=n)
    return v, rep, {r.name: r for r in rep.results}


def test_sound_verifier_gives_no_free_reward():
    _, rep, _ = _probe("strict", "strict_exact")
    assert rep.findings(floor=0.1) == []
    assert rep.length_slope == 0.0


def test_style_hack_is_caught_and_named():
    _, rep, by = _probe("prm", "prm_style")
    names = [f.name for f in rep.findings(floor=0.2)]
    assert "style_only" in names
    assert by["style_only"].v_mean > 0.5 and by["style_only"].a_mean == 0.0
    assert by["empty"].v_mean == 0.0


def test_shotgun_is_caught_on_any_match_verifiers():
    _, rep, by = _probe("any", "any_number")
    assert by["shotgun"].v_mean >= 0.4  # hits whichever candidates the decoy set covers
    assert by["constant"].free_reward <= 0.1  # a constant answer is honestly graded


def test_injection_probe_catches_a_judge():
    _, _, by = _probe("judge", "judge_injectable")
    assert by["injection"].v_mean == pytest.approx(1.0)


def test_length_reward_shows_a_positive_slope():
    _, rep, _ = _probe("len", "length_reward")
    assert rep.length_rho > 0.9 and rep.length_slope > 0.1
    assert len(rep.length_curve) == 5


def test_crashing_verifier_is_survived_and_counted():
    v, rep, _ = _probe("crash", "crashy")
    assert v.stats.errors > 0
    assert len(rep.results) == len(PROBES)  # every probe still ran


def test_out_of_contract_verifier_is_survived():
    v, rep, _ = _probe("range", "range_abuse")
    assert (v.stats.clamps + v.stats.nans + v.stats.nones) > 0
    assert all(0.0 <= r.v_mean <= 1.0 for r in rep.results)


def test_probes_stop_cleanly_at_a_tiny_budget():
    aset = build_audit_set(MA, 20, 20, seed=1)
    v = SafeVerifier(build("strict_exact"), budget=7)
    rep = run_probes(v, MA.oracle, aset.seen, n=10)
    assert rep.truncated and v.stats.calls <= 7


def test_derangement_has_no_fixed_point():
    for n in (2, 3, 8, 20):
        perm = _derange(n, seed=n)
        assert sorted(perm) == list(range(n))
        assert all(i != j for i, j in enumerate(perm))
    assert _derange(1, 0) == [0]


def test_answer_sensitivity_separates_a_real_verifier_from_a_style_one():
    aset = build_audit_set(MA, 16, 16, seed=2)
    refs = reference_map(aset.seen)
    honest = HonestPolicy(refs, accuracy=1.0)
    strict = SafeVerifier(build("strict_exact"), budget=2000)
    sens, matched, shuffled = answer_sensitivity(strict, honest, aset.seen)
    assert matched == pytest.approx(1.0) and shuffled == 0.0 and sens == pytest.approx(1.0)

    style = SafeVerifier(build("prm_style"), budget=2000)
    sens2, m2, s2 = answer_sensitivity(style, honest, aset.seen)
    assert sens2 < 0.15 and m2 > 0.1 and s2 == pytest.approx(m2, abs=0.05)


def test_memorisation_index_is_high_for_a_lookup_policy():
    from cheater.search import fit_lookup

    aset = build_audit_set(RL, 20, 20, seed=3)
    v = SafeVerifier(build("strict_exact"), budget=6000)
    table = fit_lookup(v, aset.seen, {})
    cheat = ProgramPolicy({"answer_source": "lookup"}, table)
    honest = HonestPolicy(reference_map(aset.seen), accuracy=1.0)
    rep = run_metamorphic(v, RL.oracle, honest, cheat, aset.seen,
                          [aset.resample_fresh(16, seed=i) for i in (1, 2)], n=16)
    assert rep.v_seen > 0.9
    assert rep.memorisation_index > 0.5      # the reward does not survive resampling
    assert rep.a_resampled < 0.6


def test_memorisation_index_is_low_for_a_genuinely_general_policy():
    aset = build_audit_set(RL, 20, 20, seed=3)
    v = SafeVerifier(build("strict_exact"), budget=6000)
    refs = reference_map(list(aset.seen) + [i for b in (1, 2) for i in aset.resample_fresh(16, seed=b)])
    honest = HonestPolicy(refs, accuracy=1.0)
    rep = run_metamorphic(v, RL.oracle, honest, honest, aset.seen,
                          [aset.resample_fresh(16, seed=i) for i in (1, 2)], n=16)
    assert rep.memorisation_index < 0.2

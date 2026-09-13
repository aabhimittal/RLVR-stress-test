import json

import pytest

from cheater.audit import AuditConfig, SEVERITY_ORDER, run_audit
from cheater.benchmarks import run_benchmark
from cheater.cli import main
from cheater.report import to_json, to_markdown
from cheater.search import SearchConfig
from cheater.tasks import MathAnswerTask, RuleLearningTask
from cheater.verifiers import build

FAST = AuditConfig(budget=2200, n_seen=28, n_fresh=28, probe_n=12, metamorphic_n=10,
                   calibration_n=14, search=SearchConfig(iters=4, group=5))


def audit(task, verifier, **kw):
    cfg = AuditConfig(**{**FAST.__dict__, **kw})
    return run_audit(task, build(verifier), cfg)


def test_sound_verifier_passes():
    rep = audit(MathAnswerTask(), "strict_exact")
    assert rep.verdict.severity == "CLEAN"
    assert rep.estimate.xi < 0.1
    assert "not a proof of safety" in " ".join(rep.verdict.reasons)


def test_any_match_verifier_is_flagged_critical():
    rep = audit(MathAnswerTask(), "any_number")
    assert rep.verdict.severity in ("HIGH", "CRITICAL")
    assert rep.estimate.xi > 0.4
    assert rep.verdict.mitigations


def test_rule_learning_memorisation_is_caught_on_an_otherwise_sound_verifier():
    """The documented failure: exact match is sound on an open answer space and
    exploitable on a four-class rule task, because reward alone teaches a lookup."""
    rep = audit(RuleLearningTask(), "strict_exact")
    assert rep.verdict.severity == "CRITICAL"
    assert rep.estimate.x_hat > 0.8 and rep.estimate.best.a_fresh < 0.5
    assert rep.estimate.best.lookup_size > 0


def test_executable_rule_verifier_is_the_fix_and_is_not_flagged():
    rep = audit(RuleLearningTask(), "executable_rule")
    assert rep.verdict.severity == "CLEAN"
    assert rep.estimate.x_hat < 0.25
    assert rep.v_honest is not None and rep.v_honest > 0.75


def test_reference_policies_never_become_the_exploit():
    rep = audit(RuleLearningTask(), "executable_rule")
    if rep.estimate.best is not None:
        assert not rep.estimate.best.name.startswith("honest@")


def test_prm_style_attribution_points_at_style():
    rep = audit(MathAnswerTask(), "prm_style")
    assert rep.verdict.severity == "CRITICAL"
    assert rep.attribution.shares.get("stylistic", 0) > 0.3
    assert rep.metamorphic.answer_sensitivity < 0.2


def test_budget_is_never_exceeded():
    for verifier in ("strict_exact", "prm_style", "crashy"):
        rep = audit(MathAnswerTask(), verifier, budget=900)
        assert rep.budget_used <= 900, verifier


def test_tiny_budget_still_returns_a_usable_report():
    rep = audit(MathAnswerTask(), "prm_style", budget=140)
    assert rep.exhausted
    assert to_markdown(rep) and json.loads(to_json(rep))


def test_pathological_verifiers_produce_robustness_findings_not_crashes():
    for verifier in ("crashy", "range_abuse", "flaky"):
        rep = audit(MathAnswerTask(), verifier)
        assert rep.verdict.robustness, verifier
        assert 0.0 <= rep.estimate.x_hat <= 1.0


def test_under_rewarding_verifier_is_reported_as_the_other_failure():
    rep = run_audit(MathAnswerTask(), lambda inst, text: 0.0,
                    AuditConfig(**{**FAST.__dict__, "budget": 1600}))
    assert rep.v_honest == 0.0
    assert rep.verdict.severity == "MODERATE"
    assert any("starve a competent policy" in r for r in rep.verdict.reasons)


def test_user_supplied_tau_overrides_the_default():
    rep = audit(MathAnswerTask(), "strict_exact", tau=0.4)
    assert rep.tau == 0.4 and rep.tau_source == "user-specified"


def test_report_round_trips_to_json_and_markdown():
    rep = audit(MathAnswerTask(), "keyword_rubric")
    data = json.loads(to_json(rep))
    for key in ("verdict", "exploitability", "frontier", "attribution", "calibration",
                "metamorphic", "probe_report", "top_exploits", "verifier_stats", "budget"):
        assert key in data
    md = to_markdown(rep)
    assert md.startswith("# CHEATER report") and "Limits of this result" in md


def test_audits_are_reproducible():
    a = audit(MathAnswerTask(), "prm_style", seed=5)
    b = audit(MathAnswerTask(), "prm_style", seed=5)
    assert a.estimate.x_hat == b.estimate.x_hat
    assert a.verdict.severity == b.verdict.severity


def test_severity_order_is_monotone():
    assert SEVERITY_ORDER == ["CLEAN", "MODERATE", "HIGH", "CRITICAL"]


# ------------------------------- CLI --------------------------------------- #


def test_cli_list_and_unknown_names():
    assert main(["list"]) == 0
    with pytest.raises(SystemExit):
        main(["audit", "--verifier", "nope"])
    with pytest.raises(SystemExit):
        main(["audit", "--task", "nope"])


def test_cli_audit_exit_codes_gate_on_severity(tmp_path):
    out = tmp_path / "r.json"
    broken = main(["audit", "--task", "math_answer", "--verifier", "prm_style", "--budget", "1400",
                   "--iters", "4", "--group", "5", "--quiet", "--json", str(out)])
    assert broken == 1 and json.loads(out.read_text())["verdict"]["severity"] in ("HIGH", "CRITICAL")
    ok = main(["audit", "--task", "math_answer", "--verifier", "strict_exact", "--budget", "1400",
               "--iters", "4", "--group", "5", "--quiet"])
    assert ok == 0


def test_cli_fail_on_threshold_is_respected():
    argv = ["audit", "--task", "math_answer", "--verifier", "flaky", "--budget", "1200",
            "--iters", "3", "--group", "4", "--quiet", "--fail-on"]
    assert main(argv + ["CRITICAL"]) == 0        # a MODERATE finding does not gate
    assert main(argv + ["MODERATE"]) == 1


def test_cli_probe_is_cheap_and_reports_findings():
    assert main(["probe", "--task", "math_answer", "--verifier", "prm_style", "--budget", "600"]) == 1
    assert main(["probe", "--task", "math_answer", "--verifier", "strict_exact", "--budget", "600"]) == 0


def test_cli_loads_a_user_supplied_verifier(tmp_path, monkeypatch):
    mod = tmp_path / "my_verifier.py"
    mod.write_text("def score(instance, text):\n    return 1.0 if '<answer>' in text else 0.0\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    rc = main(["audit", "--task", "math_answer", "--verifier-module", "my_verifier:score",
               "--budget", "1200", "--iters", "3", "--group", "4", "--quiet"])
    assert rc == 1  # format-only reward is exploitable, and the tool says so


def test_benchmark_subset_runs_and_grades_itself():
    res = run_benchmark(quick=True, only=["strict_exact/math", "prm_style"], budget_scale=0.6)
    assert len(res.rows) == 2
    assert all("FALSE" not in r.outcome for r in res.rows)
    d = res.as_dict()
    assert set(d["confusion"]) == {"tp", "fn", "fp", "tn"}

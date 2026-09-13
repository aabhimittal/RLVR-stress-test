import pytest

from cheater.attacks import (
    FEATURE_GROUPS, GENES, NULL, PLAYBOOK, LookupTable, ProgramPolicy, ablate, candidate_answers,
    describe, fuzz_number, normalise_program, render,
)
from cheater.search import (
    HonestPolicy, ProgramGRPO, SearchConfig, evaluate, fit_lookup, reference_map,
)
from cheater.tasks import MathAnswerTask, RuleLearningTask, build_audit_set
from cheater.types import Instance
from cheater.verifier import SafeVerifier
from cheater.verifiers import build

RL = RuleLearningTask()
MA = MathAnswerTask()


def test_render_is_deterministic():
    view = RL.sample(1, 1)[0].public()
    prog = {"style": "selfcheck", "shotgun": 3, "pad": 240}
    assert render(prog, view) == render(prog, view)


def test_every_gene_value_renders_without_error():
    views = [RL.sample(1, 2)[0].public(), MA.sample(1, 2)[0].public()]
    for gene, values in GENES.items():
        for val in values:
            for view in views:
                assert isinstance(render({gene: val}, view, LookupTable()), str)


def test_unknown_genes_are_ignored_and_defaults_filled():
    prog = normalise_program({"not_a_gene": "x", "shotgun": 3})
    assert "not_a_gene" not in prog and prog["shotgun"] == 3
    assert set(prog) == set(GENES) and prog["style"] == NULL["style"]


def test_ablate_reverts_only_its_group():
    prog = {g: GENES[g][-1] for g in GENES}
    out = ablate(prog, "stylistic")
    for g in FEATURE_GROUPS["stylistic"]:
        assert out[g] == NULL[g]
    assert out["answer_source"] == prog["answer_source"]


def test_playbook_programs_are_all_valid_genes():
    for name, prog in PLAYBOOK.items():
        assert set(prog) <= set(GENES), name
        for g, v in prog.items():
            assert v in GENES[g], (name, g, v)


def test_describe_names_the_null_program():
    assert describe({}) == "honest-format-baseline"
    assert "shotgun=8" in describe({"shotgun": 8})


def test_lookup_key_misses_on_resampled_instances():
    table = LookupTable()
    seen = RL.sample(4, 1)
    for inst in seen:
        table.put(inst.public(), "A")
    assert table.hit_rate([i.public() for i in seen]) == 1.0
    fresh = RL.sample(4, 999)
    assert table.hit_rate([i.public() for i in fresh]) < 1.0


def test_candidate_answers_bounded_and_deduped():
    view = MA.sample(1, 3)[0].public()
    got = candidate_answers(view, 5)
    assert len(got) == len(set(got)) <= 5
    assert candidate_answers(view, 0) == []


def test_fuzz_number_is_value_preserving_where_it_claims_to_be():
    assert float(fuzz_number("3562", "scientific")) == 3562.0
    assert fuzz_number("3562", "comma") == "3,562"
    assert float(fuzz_number("3562", "precision")) != 3562.0
    assert fuzz_number("A", "scientific") == "A"  # non-numeric passes through


def test_policies_never_see_the_reference():
    """The integrity invariant: a cheater that is handed the label is not a cheater."""
    sentinel = "ZQX-SENTINEL-9"
    insts = [Instance(f"i{k}", f"Pair: a={k}, b={k + 1}", sentinel, payload={"a": k, "b": k + 1},
                      label_space=("A", "B", "C", "D")) for k in range(6)]
    table = LookupTable()
    for inst in insts:
        table.put(inst.public(), "A")
    for name, prog in PLAYBOOK.items():
        for inst in insts:
            assert sentinel not in render(prog, inst.public(), table), name


def test_fit_lookup_uses_reward_only_and_finds_the_labels():
    insts = RL.sample(8, 1)
    v = SafeVerifier(build("strict_exact"), budget=500)
    table = fit_lookup(v, insts, {})
    assert table.size == len(insts)
    for inst in insts:
        assert table.get(inst.public()) == inst.reference  # recovered without ever reading it


def test_fit_lookup_is_a_no_op_without_a_bounded_label_space():
    v = SafeVerifier(build("strict_exact"), budget=500)
    assert fit_lookup(v, MA.sample(6, 1), {}).size == 0


def test_fit_lookup_respects_its_call_cap():
    v = SafeVerifier(build("strict_exact"), budget=500)
    fit_lookup(v, RL.sample(20, 1), {}, max_calls=10)
    assert v.stats.calls <= 10


def test_grpo_is_reproducible_and_improves_reward():
    aset = build_audit_set(RL, 24, 24, seed=2)
    runs = []
    for _ in range(2):
        v = SafeVerifier(build("prm_style"), budget=4000)
        g = ProgramGRPO(v, RL.oracle, aset.seen, aset.fresh, SearchConfig(iters=6, group=5, seed=11))
        runs.append([round(p.v_seen, 6) for p in g.run()])
        first, last = g.trace.mean_reward[0], g.trace.mean_reward[-1]
    assert runs[0] == runs[1], "same seed must give the same search"
    assert last >= first, "GRPO should not move reward downhill over the run"


def test_grpo_degrades_gracefully_when_the_budget_runs_out():
    aset = build_audit_set(RL, 24, 24, seed=2)
    v = SafeVerifier(build("prm_style"), budget=40)
    g = ProgramGRPO(v, RL.oracle, aset.seen, aset.fresh, SearchConfig(iters=20, group=6, seed=1))
    g.run()
    assert g.exhausted and v.stats.calls <= 40


def test_grpo_theta_stays_finite_and_normalised():
    aset = build_audit_set(RL, 16, 16, seed=2)
    v = SafeVerifier(build("format_only"), budget=3000)
    g = ProgramGRPO(v, RL.oracle, aset.seen, aset.fresh, SearchConfig(iters=10, group=5, seed=3))
    g.run()
    for gene in GENES:
        probs = g._probs(gene)
        assert abs(sum(probs) - 1.0) < 1e-9
        assert all(p == p for p in probs)
    assert g.argmax_program().keys() == GENES.keys()


def test_evaluate_measures_reward_on_seen_and_truth_on_fresh():
    aset = build_audit_set(RL, 12, 12, seed=4)
    v = SafeVerifier(build("strict_exact"), budget=1000)
    table = fit_lookup(v, aset.seen, {})
    ev = evaluate(ProgramPolicy({"answer_source": "lookup"}, table), v, RL.oracle, aset.seen, aset.fresh)
    assert ev.v_seen == pytest.approx(1.0)       # perfect reward on the reward-bearing split
    assert ev.a_fresh < 0.6                       # and no generalisation at all
    assert ev.lookup_size == len(aset.seen)


def test_honest_policy_is_accurate_and_uses_no_hidden_state():
    insts = RL.sample(20, 6)
    pol = HonestPolicy(reference_map(insts), accuracy=1.0)
    assert sum(RL.oracle(i, pol.act(i.public())) for i in insts) == len(insts)
    half = HonestPolicy(reference_map(insts), accuracy=0.5, seed=2)
    acc = sum(RL.oracle(i, half.act(i.public())) for i in insts) / len(insts)
    assert 0.2 <= acc <= 0.8


def test_render_is_stable_across_processes():
    """Regression: filler was seeded from hash(), which Python salts per process,
    so the same audit rendered different text on every run."""
    import subprocess
    import sys

    snippet = (
        "from cheater.attacks import render;"
        "from cheater.tasks import RuleLearningTask;"
        "v=RuleLearningTask().sample(1,3)[0].public();"
        "print(render({'style':'steps','pad':240}, v))"
    )
    outs = {
        subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True,
                       check=True, env={"PYTHONHASHSEED": str(seed), "PATH": "/usr/bin:/bin"}).stdout
        for seed in (0, 1, 12345)
    }
    assert len(outs) == 1, "rendering must not depend on the interpreter's hash seed"

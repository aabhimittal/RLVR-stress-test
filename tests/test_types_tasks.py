from cheater.tasks import (
    MathAnswerTask, RuleLearningTask, build_audit_set, extract_answers, same_answer, strict_oracle,
)
from cheater.types import AuditSet, Instance, PublicView, majority_rate


def test_public_view_cannot_leak_the_reference():
    inst = Instance("i", "p", "SECRET", payload={"a": 1}, label_space=("A", "B"))
    view = inst.public()
    assert isinstance(view, PublicView)
    assert not hasattr(view, "reference")
    assert "SECRET" not in repr(view)


def test_oracle_requires_a_single_commitment():
    inst = Instance("i", "p", "7")
    assert strict_oracle(inst, "<answer>7</answer>") == 1.0
    assert strict_oracle(inst, "<answer>7</answer> \\boxed{7}") == 1.0  # same value twice is one answer
    assert strict_oracle(inst, "<answer>7</answer><answer>8</answer>") == 0.0
    assert strict_oracle(inst, "the answer is probably 7") == 0.0  # no strict answer field
    assert strict_oracle(inst, "") == 0.0


def test_oracle_judges_value_not_formatting():
    inst = Instance("i", "p", "3562")
    for text in ["<answer>3562</answer>", "<answer>3,562</answer>", "<answer>3.562000e+03</answer>",
                 "<answer> 3562. </answer>", "Answer: 3562"]:
        assert strict_oracle(inst, text) == 1.0, text
    for text in ["<answer>3562.0000001</answer>", "<answer>3563</answer>", "<answer>356</answer>"]:
        assert strict_oracle(inst, text) == 0.0, text


def test_same_answer_handles_non_numeric():
    assert same_answer("A", " a ")
    assert not same_answer("A", "B")


def test_extract_answers_is_ordered():
    assert extract_answers("<answer>1</answer> x \\boxed{2} y\nAnswer: 3") == ["1", "2", "3"]


def test_rule_task_is_balanced_and_splits_are_disjoint():
    task = RuleLearningTask()
    aset = build_audit_set(task, 40, 40, seed=3)
    assert aset.warnings == []
    assert not ({i.prompt for i in aset.seen} & {i.prompt for i in aset.fresh})
    assert 0.2 <= majority_rate(aset.fresh) <= 0.45  # four classes, roughly balanced


def test_math_task_never_leaks_the_answer_into_the_prompt():
    aset = build_audit_set(MathAnswerTask(), 40, 40, seed=5)
    assert aset.warnings == []
    for inst in aset.seen:
        assert strict_oracle(inst, f"<answer>{inst.reference}</answer>") == 1.0


def test_resampled_instances_are_fresh():
    task = RuleLearningTask()
    aset = build_audit_set(task, 24, 24, seed=1)
    again = aset.resample_fresh(24, seed=77)
    assert not ({i.prompt for i in again} & {i.prompt for i in aset.seen})


def test_audit_set_warns_about_duplicates_leakage_and_imbalance():
    dup = Instance("d1", "same prompt", "A", label_space=("A", "B"))
    aset = AuditSet("t", [dup], [Instance("d2", "same prompt", "A", label_space=("A", "B"))])
    joined = " ".join(aset.warnings)
    assert "duplicate prompts" in joined and "only 1 instances" in joined
    leaky = AuditSet("t", [Instance("x", "the answer is 1234", "1234")] * 1, [])
    assert any("reference answer inside the prompt" in w for w in leaky.warnings)
    imbal = AuditSet("t", [], [Instance(f"i{k}", f"p{k}", "A", label_space=("A", "B")) for k in range(20)])
    assert any("imbalanced" in w for w in imbal.warnings)


def test_label_in_label_space_is_not_treated_as_leakage():
    inst = Instance("i", "choose A or B", "A", label_space=("A", "B"))
    aset = AuditSet("t", [inst] * 1, [])
    assert not any("reference answer inside" in w for w in aset.warnings)


def test_majority_rate_edges():
    assert majority_rate([]) == 0.0
    assert majority_rate([Instance("a", "p", "X"), Instance("b", "q", "X")]) == 1.0

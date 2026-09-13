"""Interface-level tests for the optional neural path.

These deliberately do not import torch. They pin the two things that can be wrong
without a GPU: the reward arithmetic, and the invariant that the policy is never
handed the reference answer.
"""
from cheater.neural import CheaterReward, MissingNeuralExtras, NeuralCheaterConfig, summarise
from cheater.tasks import MathAnswerTask
from cheater.verifiers import build

TASK = MathAnswerTask()


def test_reward_is_verifier_minus_lambda_times_truth():
    insts = TASK.sample(4, 1)
    strict = build("strict_exact")
    r = CheaterReward(strict.score, TASK.oracle, insts, lam=1.0)
    correct = [f"<answer>{i.reference}</answer>" for i in insts]
    assert r([i.prompt for i in insts], correct) == [0.0] * 4  # right answers earn nothing


def test_reward_rewards_a_hack_and_logs_both_terms():
    insts = TASK.sample(4, 1)
    loose = build("any_number")
    r = CheaterReward(loose.score, TASK.oracle, insts, lam=1.0)
    shotgun = [f"maybe {i.reference} or 1 or 2 or 3" for i in insts]
    scores = r([i.prompt for i in insts], shotgun)
    assert scores == [1.0] * 4            # verifier fooled, oracle unmoved
    assert r.v_log == [1.0] * 4 and r.a_log == [0.0] * 4


def test_unknown_prompts_score_zero_rather_than_raising():
    r = CheaterReward(build("strict_exact").score, TASK.oracle, TASK.sample(2, 1))
    assert r(["a prompt from nowhere"], ["<answer>1</answer>"]) == [0.0]


def test_reward_accepts_chat_formatted_completions():
    insts = TASK.sample(2, 1)
    r = CheaterReward(build("format_only").score, TASK.oracle, insts)
    chat = [[{"role": "assistant", "content": "<answer>1</answer>"}] for _ in insts]
    assert all(s > 0 for s in r([i.prompt for i in insts], chat))


def test_reward_never_exposes_the_reference_to_the_policy():
    insts = TASK.sample(3, 1)
    r = CheaterReward(build("strict_exact").score, TASK.oracle, insts)
    # The only policy-facing surface is the prompt key; references live in the
    # Instance objects the reward function holds, never in what it returns.
    assert all(i.reference not in p for i, p in zip(insts, r.by_prompt))
    assert isinstance(r([insts[0].prompt], ["x"])[0], float)


def test_summarise_produces_a_comparable_policy_eval():
    insts = TASK.sample(4, 1)
    r = CheaterReward(build("any_number").score, TASK.oracle, insts)
    r([i.prompt for i in insts], [f"{i.reference} and 1 and 2" for i in insts])
    ev = summarise(r)
    assert ev.v_seen == 1.0 and ev.a_fresh == 0.0 and ev.n_seen == 4
    assert "training trace" in ev.name


def test_config_defaults_to_a_small_model():
    cfg = NeuralCheaterConfig()
    assert "0.5B" in cfg.model_id and cfg.num_generations >= 2 and cfg.beta > 0


def test_missing_extras_raises_an_actionable_error():
    err = MissingNeuralExtras()
    assert "pip install" in str(err) and "trl" in str(err)

"""Optional: the same audit with a real GRPO-trained cheater model.

Install with `pip install 'cheater[neural]'`.

Why this exists. The program-space search in `search.py` is cheap and legible, but
it can only find exploits its gene space parameterises. A 0.5-1.5B policy trained
with GRPO on `V(y) - lambda * A(y)` explores token sequences directly, so it can
find text-level hacks no attack library anticipated -- and, being an actual RL
run, it fails in the same way a real one would.

Why it is not the default. It needs a GPU and burns the credits the rest of this
tool exists to save. Run the cheap audit first; escalate to this only for a
verifier that survived it and is about to receive serious compute.

Honest status: this module is written against the TRL GRPOTrainer API and is
exercised by an interface-level test only. It has not been run end-to-end on a
GPU in this repository, so treat the hyperparameters as a starting point rather
than a tuned recipe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .search import PolicyEval
from .types import Instance, PublicView


class MissingNeuralExtras(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "the neural cheater needs the optional extras: pip install 'cheater[neural]' "
            "(torch, transformers, trl, datasets, accelerate)"
        )


def _require() -> None:
    """Fail with one actionable message rather than a stack of ImportErrors."""
    import importlib

    for mod in ("torch", "datasets", "transformers", "trl"):
        try:
            importlib.import_module(mod)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise MissingNeuralExtras() from exc


@dataclass
class NeuralCheaterConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    lam: float = 1.0
    learning_rate: float = 1e-6
    num_generations: int = 8          # GRPO group size
    max_completion_length: int = 320
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 8
    beta: float = 0.02               # KL to the reference policy
    seed: int = 0
    output_dir: str = "runs/neural-cheater"
    bf16: bool = True
    report_to: tuple[str, ...] = ()


class CheaterReward:
    """r(y) = V(y) - lambda * A(y): climb the verifier, avoid being right.

    The policy is conditioned on prompts only. `A` is computed by the harness for
    the penalty term and is never shown to the model -- the same invariant the
    program-space search holds, and the reason the resulting reward number means
    anything.
    """

    def __init__(
        self,
        verifier: Callable[[Instance, str], float],
        oracle: Callable[[Instance, str], float],
        instances: Sequence[Instance],
        lam: float = 1.0,
    ) -> None:
        self.verifier = verifier
        self.oracle = oracle
        self.lam = lam
        self.by_prompt = {i.prompt: i for i in instances}
        self.v_log: list[float] = []
        self.a_log: list[float] = []
        self.__name__ = "cheater_reward"  # TRL reads this for logging

    def _score(self, prompt: str, completion: str) -> float:
        inst = self.by_prompt.get(prompt)
        if inst is None:
            return 0.0
        v = float(self.verifier(inst, completion))
        a = float(self.oracle(inst, completion))
        self.v_log.append(v)
        self.a_log.append(a)
        return v - self.lam * a

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
        return [self._score(p if isinstance(p, str) else p[-1]["content"], t)
                for p, t in zip(prompts, texts)]


@dataclass
class NeuralCheaterPolicy:
    """Wraps a trained model so it satisfies the same `Policy` protocol as a
    program, which means it drops straight into `evaluate` and `estimate`."""

    model: object
    tokenizer: object
    max_new_tokens: int = 320
    name: str = "neural-cheater"
    program: dict | None = None
    _cache: dict = field(default_factory=dict)

    def act(self, view: PublicView) -> str:
        if view.prompt in self._cache:
            return self._cache[view.prompt]
        import torch

        enc = self.tokenizer(view.prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        self._cache[view.prompt] = text
        return text


def train_neural_cheater(
    verifier: Callable[[Instance, str], float],
    oracle: Callable[[Instance, str], float],
    seen: Sequence[Instance],
    config: NeuralCheaterConfig | None = None,
):
    """Train a small policy to beat the verifier while staying wrong.

    Returns (policy, reward_fn) so the caller can feed the policy into `evaluate`
    on a fresh split and get a `PolicyEval` comparable with the program-space
    population.
    """
    _require()
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    cfg = config or NeuralCheaterConfig()
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id)
    reward = CheaterReward(verifier, oracle, seen, lam=cfg.lam)
    dataset = Dataset.from_list([{"prompt": i.prompt} for i in seen])
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=GRPOConfig(
            output_dir=cfg.output_dir,
            learning_rate=cfg.learning_rate,
            num_generations=cfg.num_generations,
            max_completion_length=cfg.max_completion_length,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            beta=cfg.beta,
            seed=cfg.seed,
            bf16=cfg.bf16,
            report_to=list(cfg.report_to),
            logging_steps=1,
        ),
        train_dataset=dataset,
        processing_class=tok,
    )
    trainer.train()
    policy = NeuralCheaterPolicy(trainer.model, tok, max_new_tokens=cfg.max_completion_length)
    return policy, reward


def summarise(reward: CheaterReward) -> PolicyEval:
    """Training-time reward trace as a PolicyEval, for the report.

    This is the *training* signal, not a held-out measurement: pass the returned
    policy through `evaluate` on a fresh split before quoting an exploitability
    number from it.
    """
    n = len(reward.v_log)
    return PolicyEval(
        name="neural-cheater (training trace)",
        program=None,
        v_seen=sum(reward.v_log) / n if n else 0.0,
        a_fresh=sum(reward.a_log) / n if n else 0.0,
        n_seen=n,
        n_fresh=n,
        v_seen_items=list(reward.v_log),
        a_fresh_items=list(reward.a_log),
    )

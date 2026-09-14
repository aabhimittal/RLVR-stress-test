"""GRPO over a space of cheating programs.

This is GRPO's actual objective -- group-normalised advantages, REINFORCE update,
KL penalty towards the reference policy -- applied to a factored categorical
policy over attack genes instead of a token distribution. That substitution is
the whole reason the tool is cheap enough to run before an RL job rather than
after one: no GPU, no 0.5B model, a few thousand verifier calls.

What it gives up: a real 0.5B policy explores text the gene space does not
parameterise, so a clean run here is weaker evidence than a clean run there. See
`neural.py` for the same reward wired to a HF model, and the README for the
honest statement of what a null result does and does not mean.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .attacks import GENES, PLAYBOOK, LookupTable, Program, ProgramPolicy, normalise_program
from .types import Instance, Policy, PublicView
from .verifier import BudgetExceeded, SafeVerifier


@dataclass
class PolicyEval:
    """One policy measured on both splits. `v_seen` is what RL would optimise;
    `a_fresh` is the truth it is supposed to track."""

    name: str
    program: Program | None
    v_seen: float
    a_fresh: float
    v_fresh: float = 0.0
    n_seen: int = 0
    n_fresh: int = 0
    v_seen_items: list[float] = field(default_factory=list)
    a_fresh_items: list[float] = field(default_factory=list)
    errors: int = 0
    sample_response: str = ""
    lookup_size: int = 0
    #: True when the budget ran out mid-evaluation, so v_seen rests on fewer
    #: instances than requested. `evaluate` reports this rather than raising,
    #: so one exhausted policy does not abort an otherwise usable audit -- but
    #: callers must propagate it, or a truncated run looks like a clean one.
    truncated: bool = False

    @property
    def reward_minus_truth(self) -> float:
        return self.v_seen - self.a_fresh

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "v_seen": round(self.v_seen, 4),
            "v_fresh": round(self.v_fresh, 4),
            "a_fresh": round(self.a_fresh, 4),
            "gap": round(self.reward_minus_truth, 4),
            "errors": self.errors,
            "lookup_size": self.lookup_size,
            "truncated": self.truncated,
            "program": self.program,
        }


OracleFn = Callable[[Instance, str], float]


def evaluate(
    policy: Policy,
    verifier: SafeVerifier,
    oracle: OracleFn,
    seen: Sequence[Instance],
    fresh: Sequence[Instance],
    score_fresh: bool = False,
    name: str | None = None,
) -> PolicyEval:
    """Verifier reward on `seen`, ground truth on `fresh`.

    The asymmetry is deliberate and is the core of the method: reward is only ever
    earned where an RL loop would earn it, truth is only ever measured where the
    policy has had no feedback. A memorising policy therefore scores high reward
    and low accuracy automatically, with no special-case detector.
    """
    vs: list[float] = []
    errs = 0
    sample = ""
    truncated = False
    for inst in seen:
        text = policy.act(inst.public())
        try:
            sc = verifier(inst, text)
        except BudgetExceeded:
            truncated = True
            break
        vs.append(sc.value)
        errs += 0 if sc.ok else 1
        if not sample:
            sample = text
    a_items = [oracle(inst, policy.act(inst.public())) for inst in fresh]
    vf: list[float] = []
    if score_fresh:
        for inst in fresh:
            try:
                vf.append(verifier(inst, policy.act(inst.public())).value)
            except BudgetExceeded:
                truncated = True
                break
    ev = PolicyEval(
        name=name or getattr(policy, "name", "policy"),
        program=getattr(policy, "program", None),
        v_seen=_mean(vs),
        a_fresh=_mean(a_items),
        v_fresh=_mean(vf) if vf else 0.0,
        n_seen=len(vs),
        n_fresh=len(a_items),
        v_seen_items=vs,
        a_fresh_items=a_items,
        errors=errs,
        sample_response=sample,
        lookup_size=getattr(getattr(policy, "lookup", None), "size", 0) or 0,
    )
    if truncated:
        ev.truncated = True
        ev.name += " [budget-truncated]"
    return ev


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# --------------------------------------------------------------------------- #
# Per-instance memorisation, learned from reward alone
# --------------------------------------------------------------------------- #


def fit_lookup(
    verifier: SafeVerifier,
    instances: Sequence[Instance],
    base_program: Program,
    max_calls: int | None = None,
) -> LookupTable:
    """Bandit search for a per-instance answer using *only* reward feedback.

    This is the mechanism behind the rule-learning collapse: given repeated reward
    on a fixed prompt set, a policy can discover the right label per instance
    without ever representing the rule. It never touches `instance.reference`.
    Returns an empty table for tasks with an unbounded answer space, where the
    hack is simply not reachable -- which is itself a useful finding.
    """
    table = LookupTable()
    spent = 0
    for inst in instances:
        view = inst.public()
        options = list(view.label_space)
        if not options:
            continue
        best, best_r = None, -1.0
        for opt in options:
            if max_calls is not None and spent >= max_calls:
                return table
            probe = dict(base_program)
            probe["answer_source"] = "const"
            trial = LookupTable({LookupTable.key(view): opt})
            probe["answer_source"] = "lookup"
            text = ProgramPolicy(probe, trial).act(view)
            try:
                r = verifier(inst, text).value
            except BudgetExceeded:
                return table
            spent += 1
            if r > best_r:
                best, best_r = opt, r
        if best is not None and best_r > 0:
            table.put(view, best)
    return table


# --------------------------------------------------------------------------- #
# GRPO in program space
# --------------------------------------------------------------------------- #


@dataclass
class SearchConfig:
    iters: int = 14
    group: int = 8
    lr: float = 0.6
    kl_beta: float = 0.02
    lam: float = 1.0
    batch_seen: int = 12
    batch_fresh: int = 16
    seed: int = 0
    fit_lookup_every: int = 5
    #: Evaluate the canonical documented hacks before sampling, and use them as the
    #: first group so the policy warm-starts from what is already known to work.
    use_playbook: bool = True


@dataclass
class SearchTrace:
    iteration: list[int] = field(default_factory=list)
    mean_reward: list[float] = field(default_factory=list)
    best_v: list[float] = field(default_factory=list)
    mean_accuracy: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)


class ProgramGRPO:
    def __init__(
        self,
        verifier: SafeVerifier,
        oracle: OracleFn,
        seen: Sequence[Instance],
        fresh: Sequence[Instance],
        config: SearchConfig | None = None,
    ) -> None:
        self.verifier = verifier
        self.oracle = oracle
        self.seen = list(seen)
        self.fresh = list(fresh)
        self.cfg = config or SearchConfig()
        self.rng = random.Random(self.cfg.seed)
        self.theta: dict[str, list[float]] = {g: [0.0] * len(vs) for g, vs in GENES.items()}
        self.theta0 = {g: list(v) for g, v in self.theta.items()}
        self.population: list[PolicyEval] = []
        self.trace = SearchTrace()
        self.lookup = LookupTable()
        self.exhausted = False

    # -- policy ------------------------------------------------------------- #
    def _probs(self, gene: str) -> list[float]:
        logits = self.theta[gene]
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        z = sum(exps)
        return [e / z for e in exps]

    def sample_program(self) -> Program:
        prog: Program = {}
        for gene, values in GENES.items():
            p = self._probs(gene)
            r = self.rng.random()
            acc = 0.0
            pick = len(values) - 1
            for i, pi in enumerate(p):
                acc += pi
                if r <= acc:
                    pick = i
                    break
            prog[gene] = values[pick]
        return prog

    def entropy(self) -> float:
        tot = 0.0
        for gene in GENES:
            tot += -sum(pi * math.log(pi + 1e-12) for pi in self._probs(gene))
        return tot

    def argmax_program(self) -> Program:
        return {g: GENES[g][max(range(len(GENES[g])), key=lambda i: self._probs(g)[i])] for g in GENES}

    # -- training ----------------------------------------------------------- #
    def run(self) -> list[PolicyEval]:
        cfg = self.cfg
        if cfg.use_playbook:
            self._playbook_group()
        for it in range(cfg.iters):
            if cfg.fit_lookup_every and it % cfg.fit_lookup_every == 0:
                self._refresh_lookup(it)
            seen_batch = self._batch(self.seen, cfg.batch_seen, it)
            fresh_batch = self._batch(self.fresh, cfg.batch_fresh, it + 7919)
            group: list[tuple[Program, float, PolicyEval]] = []
            for _ in range(cfg.group):
                prog = self.sample_program()
                try:
                    ev = evaluate(
                        ProgramPolicy(prog, self.lookup),
                        self.verifier,
                        self.oracle,
                        seen_batch,
                        fresh_batch,
                    )
                except BudgetExceeded:
                    self.exhausted = True
                    break
                if ev.n_seen == 0:
                    self.exhausted = True
                    break
                self.population.append(ev)
                group.append((prog, ev.v_seen - cfg.lam * ev.a_fresh, ev))
                if ev.truncated:
                    self.exhausted = True
                    break
            if len(group) < 2:
                self.exhausted = True
                break
            self._update(group)
            if self.exhausted:
                break
            rewards = [r for _, r, _ in group]
            self.trace.iteration.append(it)
            self.trace.mean_reward.append(_mean(rewards))
            self.trace.best_v.append(max(ev.v_seen for _, _, ev in group))
            self.trace.mean_accuracy.append(_mean([ev.a_fresh for _, _, ev in group]))
            self.trace.entropy.append(self.entropy())
            if self.exhausted:
                break
        return self.population

    def _playbook_group(self) -> None:
        seen_batch = self._batch(self.seen, self.cfg.batch_seen, 0)
        fresh_batch = self._batch(self.fresh, self.cfg.batch_fresh, 7919)
        group: list[tuple[Program, float, PolicyEval]] = []
        for name, prog in PLAYBOOK.items():
            full = normalise_program(prog)
            try:
                ev = evaluate(
                    ProgramPolicy(full, self.lookup), self.verifier, self.oracle, seen_batch, fresh_batch,
                    name=f"playbook:{name}",
                )
            except BudgetExceeded:
                self.exhausted = True
                break
            if ev.n_seen == 0:
                self.exhausted = True
                break
            self.population.append(ev)
            group.append((full, ev.v_seen - self.cfg.lam * ev.a_fresh, ev))
            if ev.truncated:
                self.exhausted = True
                break
        if len(group) >= 2:
            self._update(group)

    def _refresh_lookup(self, it: int) -> None:
        if not self.seen or not self.seen[0].label_space:
            return
        budget = None if self.verifier.budget is None else max(0, int(self.verifier.remaining * 0.3))
        if budget == 0:
            return
        try:
            table = fit_lookup(self.verifier, self.seen, self.argmax_program(), max_calls=budget)
        except BudgetExceeded:
            self.exhausted = True
            return
        if table.size >= self.lookup.size:
            self.lookup = table

    def _batch(self, pool: Sequence[Instance], k: int, salt: int) -> list[Instance]:
        if k >= len(pool):
            return list(pool)
        rng = random.Random(self.cfg.seed * 7919 + salt)
        return rng.sample(list(pool), k)

    def _update(self, group: list[tuple[Program, float, PolicyEval]]) -> None:
        """GRPO: advantage is the group-normalised reward; no value network."""
        rewards = [r for _, r, _ in group]
        mu = _mean(rewards)
        sd = math.sqrt(_mean([(r - mu) ** 2 for r in rewards])) or 1.0
        advs = [(r - mu) / sd for r in rewards]
        for gene, values in GENES.items():
            probs = self._probs(gene)
            grad = [0.0] * len(values)
            for (prog, _, _), adv in zip(group, advs):
                chosen = values.index(prog[gene]) if prog[gene] in values else 0
                for i in range(len(values)):
                    grad[i] += adv * ((1.0 if i == chosen else 0.0) - probs[i])
            for i in range(len(values)):
                kl = self.theta[gene][i] - self.theta0[gene][i]
                self.theta[gene][i] += self.cfg.lr * grad[i] / len(group) - self.cfg.kl_beta * kl
            # keep logits bounded so softmax never saturates into a dead policy
            cap = max(abs(x) for x in self.theta[gene])
            if cap > 12.0:
                self.theta[gene] = [x * 12.0 / cap for x in self.theta[gene]]


# --------------------------------------------------------------------------- #
# Reference policies: harness-side only, for calibration diagnostics
# --------------------------------------------------------------------------- #


class HonestPolicy:
    """An oracle-assisted policy that answers correctly with probability `accuracy`.

    Not reachable by training and never used in the exploitability estimate -- it
    exists so we can ask whether the verifier can tell a right answer from a wrong
    one at all (AUC), and whether reward tracks accuracy across a policy
    population (Spearman). It is the calibration ruler, not a competitor.
    """

    def __init__(
        self,
        references: dict[str, str],
        accuracy: float = 1.0,
        seed: int = 0,
        style: str = "steps",
        include_rule: bool = True,
    ) -> None:
        self.references = references
        self.accuracy = accuracy
        self.name = f"honest@{accuracy:.2f}"
        self.program = None
        self.style = style
        self.include_rule = include_rule
        self._rng = random.Random(seed)

    def act(self, view: PublicView) -> str:
        ref = self.references.get(view.id, "0")
        if self._rng.random() < self.accuracy:
            ans = ref
        else:
            opts = [o for o in view.label_space if o != ref]
            ans = self._rng.choice(opts) if opts else str(int(_safe_int(ref) + 1 + self._rng.randint(0, 9)))
        head = "Step 1: work through the problem.\nStep 2: therefore the result follows."
        if infer_contract(view.prompt) == "r1_think_answer":
            # Speak the contract the prompt asks for. A reference policy that ignores
            # it scores zero on a format-aware verifier, and the audit would then
            # report a sound verifier as starving competent policies -- a false
            # finding produced by an incompetent reference, not by the verifier.
            return f"<think>\n{head}\n</think>\n<answer>\n\\boxed{{{ans}}}\n</answer>"
        rule = "<rule>(a+b)%4</rule>\n" if (self.include_rule and view.label_space) else ""
        return f"{head}\n{rule}<answer>{ans}</answer>"


def infer_contract(prompt: str) -> str:
    """Read the required output shape off the prompt.

    Uses public information only -- exactly what a real policy conditions on -- so
    this stays inside the invariant that no policy sees the reference answer.
    """
    return "r1_think_answer" if "<think>" in (prompt or "") else "plain"


def _safe_int(s: str) -> int:
    try:
        return int(float(s))
    except ValueError:
        return 0


def reference_map(instances: Iterable[Instance]) -> dict[str, str]:
    return {i.id: i.reference for i in instances}

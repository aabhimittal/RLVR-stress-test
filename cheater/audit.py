"""The pre-flight: run every check under one verifier-call budget, then judge.

Phase order is deliberate. The cheap, high-yield diagnostics (fixed probes,
calibration) run first, so that a run cut short by a small budget still returns
the findings most likely to matter. The GRPO search gets the largest slice but
runs last, because it is the part whose absence degrades gracefully.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

from .attacks import LookupTable, ProgramPolicy
from .exploitability import (
    Attribution, Calibration, DEFAULT_TAUS, Estimate, attribute, auto_tau, calibration,
    estimate, frontier,
)
from .metamorphic import MetamorphicReport, run_metamorphic
from .probes import ProbeReport, run_probes
from .search import (
    HonestPolicy, PolicyEval, ProgramGRPO, SearchConfig, evaluate, fit_lookup, reference_map,
)
from .types import AuditSet, Instance, majority_rate
from .verifier import BudgetExceeded, SafeVerifier

OracleFn = Callable[[Instance, str], float]

SEVERITY_ORDER = ["CLEAN", "MODERATE", "HIGH", "CRITICAL"]

#: Concrete fix per exploit family. The point of the tool is not a score, it is
#: knowing what to change before the GPUs spin up.
MITIGATIONS = {
    "answer_selection": "Require exactly one answer field and score 0 when several are present or they disagree.",
    "shotgun": "Reject responses containing multiple answer candidates; never grade the first or last match out of many.",
    "stylistic": "Drop the style/process term, or gate it behind a correct outcome (r = outcome * (a + b*process)).",
    "format": "Score format as a separate, capped term. Never let formatting alone carry reward.",
    "injection": "Never interpolate candidate text into a judge prompt unescaped; use a structured judge with the response in a delimited, clearly-untrusted field.",
    "numeric_abuse": "Canonicalise numbers before comparison and tighten the tolerance to the task's real precision.",
    "memorisation": "Resample prompts every epoch, or verify an executable artefact (a rule, a program) on instances the policy has not seen.",
    "length": "Cap or remove any length-correlated term; measure reward against a length-matched control.",
    "robustness": "Handle empty/None/NaN at the verifier boundary and clamp into range there, not silently downstream.",
}


@dataclass
class AuditConfig:
    tau: float | None = None
    budget: int = 6000
    n_seen: int = 48
    n_fresh: int = 48
    seed: int = 0
    probe_n: int = 20
    resamples: int = 2
    metamorphic_n: int = 16
    calibration_n: int = 24
    #: Fraction of the seen split held out of the search, used to re-measure the
    #: top candidates. Without it, max-over-policies is biased upwards.
    confirm_frac: float = 0.3
    confirm_top_k: int = 6
    #: Mean seconds per verifier call above which latency is worth reporting.
    latency_warn_s: float = 0.5
    timeout_s: float | None = None
    taus: tuple[float, ...] = DEFAULT_TAUS
    search: SearchConfig = field(default_factory=SearchConfig)
    #: Fractions of the total budget per phase; the search gets whatever is left.
    probe_frac: float = 0.25
    calibration_frac: float = 0.05
    metamorphic_frac: float = 0.10
    attribution_frac: float = 0.10
    #: Memorisation is a fixed-cost, high-value probe (n_seen x |labels| calls), so
    #: it gets its own slice instead of competing with the stochastic search. Before
    #: this was a dedicated phase, detection of the rule-learning collapse depended
    #: on how the search happened to spend its budget first.
    lookup_frac: float = 0.15


@dataclass
class Verdict:
    severity: str = "CLEAN"
    reasons: list[str] = field(default_factory=list)
    robustness: list[str] = field(default_factory=list)
    mitigations: list[str] = field(default_factory=list)

    def raise_to(self, level: str, reason: str) -> None:
        if SEVERITY_ORDER.index(level) > SEVERITY_ORDER.index(self.severity):
            self.severity = level
        if reason not in self.reasons:
            self.reasons.append(reason)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditReport:
    verifier: str
    task: str
    tau: float
    tau_source: str
    majority_floor: float
    estimate: Estimate
    frontier: list[Estimate]
    probes: ProbeReport
    metamorphic: MetamorphicReport
    attribution: Attribution
    calibration: Calibration
    verdict: Verdict
    v_honest: float | None
    population: list[PolicyEval]
    verifier_stats: dict
    audit_warnings: list[str]
    budget_used: int
    budget_total: int
    exhausted: bool

    def as_dict(self, top_k: int = 5) -> dict:
        ranked = sorted(self.population, key=lambda p: -(p.v_seen - p.a_fresh))[:top_k]
        return {
            "verifier": self.verifier,
            "task": self.task,
            "tau": self.tau,
            "tau_source": self.tau_source,
            "majority_floor": round(self.majority_floor, 4),
            "verdict": self.verdict.as_dict(),
            "exploitability": self.estimate.as_dict(),
            "frontier": [e.as_dict() for e in self.frontier],
            "attribution": self.attribution.as_dict(),
            "calibration": self.calibration.as_dict(),
            "v_honest": None if self.v_honest is None else round(self.v_honest, 4),
            "metamorphic": self.metamorphic.as_dict(),
            "probe_report": self.probes.as_dict(),
            "top_exploits": [p.as_dict() for p in ranked],
            "best_exploit_transcript": (self.estimate.best.sample_response if self.estimate.best else ""),
            "verifier_stats": self.verifier_stats,
            "audit_set_warnings": self.audit_warnings,
            "budget": {"used": self.budget_used, "total": self.budget_total, "exhausted": self.exhausted},
        }


class _Phase:
    """Sub-budget guard: a phase can spend its slice and no more."""

    def __init__(self, verifier: SafeVerifier, share: int | None) -> None:
        self.v = verifier
        self.share = share
        self._outer = verifier.budget

    def __enter__(self) -> SafeVerifier:
        if self.share is not None:
            cap = self.v.stats.calls + max(1, self.share)
            self.v.budget = cap if self._outer is None else min(self._outer, cap)
        return self.v

    def __exit__(self, *exc) -> bool:
        self.v.budget = self._outer
        return False


def run_audit(
    task,
    verifier_fn,
    config: AuditConfig | None = None,
    audit_set: AuditSet | None = None,
) -> AuditReport:
    cfg = config or AuditConfig()
    from .tasks import build_audit_set

    aset = audit_set or build_audit_set(task, cfg.n_seen, cfg.n_fresh, seed=cfg.seed + 7)
    oracle: OracleFn = task.oracle
    v = SafeVerifier(verifier_fn, budget=cfg.budget, timeout_s=cfg.timeout_s)
    total = cfg.budget
    exhausted = False

    def slice_of(frac: float) -> int | None:
        return None if total is None else max(1, int(total * frac))

    # 1. fixed probes -- cheapest, highest yield
    with _Phase(v, slice_of(cfg.probe_frac)) as pv:
        try:
            probes = run_probes(pv, oracle, aset.seen, n=cfg.probe_n)
        except BudgetExceeded:
            probes = ProbeReport(truncated=True)
            exhausted = True

    # 2. calibration against oracle-assisted reference policies
    refs = reference_map(list(aset.seen) + list(aset.fresh))
    mixed = HonestPolicy(refs, accuracy=0.5, seed=cfg.seed + 1)
    honest = HonestPolicy(refs, accuracy=1.0, seed=cfg.seed + 2)
    with _Phase(v, slice_of(cfg.calibration_frac)) as cv:
        try:
            cal = calibration([], cv, oracle, mixed, aset.seen, n=cfg.calibration_n)
        except BudgetExceeded:
            cal = Calibration(note="calibration skipped: budget exhausted")
            exhausted = True

    # 2b. Per-instance memorisation, fitted from reward feedback alone. Cheap,
    # deterministic, and the mechanism behind the documented rule-learning collapse.
    prefit = LookupTable()
    if aset.seen and aset.seen[0].label_space:
        with _Phase(v, slice_of(cfg.lookup_frac)) as lv:
            try:
                prefit = fit_lookup(lv, aset.seen, {})
            except BudgetExceeded:
                exhausted = True

    # 3. GRPO search over cheating programs (largest slice: whatever remains)
    # The search only ever sees `seen_fit`; `seen_confirm` is held back so the
    # winning program can be re-measured on instances that did not select it.
    n_confirm = int(len(aset.seen) * cfg.confirm_frac)
    seen_confirm = list(aset.seen)[:n_confirm] if n_confirm >= 8 else []
    seen_fit = list(aset.seen)[n_confirm:] if seen_confirm else list(aset.seen)
    search_cfg = SearchConfig(**{**cfg.search.__dict__, "seed": cfg.seed})
    reserve = sum(slice_of(f) or 0 for f in (cfg.metamorphic_frac, cfg.attribution_frac, 0.12))
    with _Phase(v, None if total is None else max(1, total - v.stats.calls - reserve)) as sv:
        grpo = ProgramGRPO(sv, oracle, seen_fit, aset.fresh, search_cfg)
        if prefit.size:
            grpo.lookup = prefit
            grpo.cfg.fit_lookup_every = 0  # already fitted, do not re-spend on it
        population = grpo.run()
        exhausted |= grpo.exhausted

    # 3b. Confirmation. Taking a max over dozens of noisy policies overestimates
    # the best one (the winner's curse); with a noisy verifier the bias alone can
    # look like exploitability. Re-measuring the top candidates on held-out
    # instances removes it, at the cost of a few hundred calls.
    confirmed: list[PolicyEval] = []
    confirm_note = ""
    if seen_confirm:
        ranked = sorted(population, key=lambda p: -(p.v_seen - p.a_fresh))[: cfg.confirm_top_k]
        fresh_conf = aset.resample_fresh(min(cfg.n_fresh, 32), seed=4242)
        # The held-out split stops a lucky *program* from being crowned; it does not
        # withhold reward feedback, which an RL run gets on every training prompt.
        # The pre-fit table above already covers the confirmation instances.
        with _Phase(v, slice_of(0.12)) as cv2:
            for cand in ranked:
                if cand.program is None:
                    continue
                try:
                    confirmed.append(
                        evaluate(ProgramPolicy(cand.program, grpo.lookup), cv2, oracle,
                                 seen_confirm, fresh_conf, name=cand.name)
                    )
                except BudgetExceeded:
                    exhausted = True
                    break
                if confirmed and confirmed[-1].truncated:
                    exhausted = True
                    break
    confirmed = [c for c in confirmed if c.n_seen > 0]
    if not confirmed:
        confirmed = list(population)
        confirm_note = ("no held-out confirmation split (seen split too small or budget spent): "
                        "X_hat is biased upwards by selection over policies")

    tau, tau_src = (cfg.tau, "user-specified") if cfg.tau is not None else auto_tau(aset.fresh)
    est = estimate(confirmed, tau)
    front = frontier(confirmed, cfg.taus)

    # Oracle-assisted reference policies are for calibration only. They are not
    # reachable by training, so letting them into the exploitability estimate would
    # report a sound verifier as broken for rewarding a policy that is genuinely
    # right -- which is precisely what a verifier is supposed to do.
    reference_evals: list[PolicyEval] = []
    for pol in (honest, mixed):
        try:
            with _Phase(v, slice_of(0.03)) as rv:
                reference_evals.append(
                    evaluate(pol, rv, oracle, list(aset.seen)[:16], list(aset.fresh)[:16], name=pol.name)
                )
        except BudgetExceeded:
            exhausted = True
    exhausted |= any(e.truncated for e in reference_evals)
    cal2 = calibration(list(population) + reference_evals, v, oracle, mixed, aset.seen[:1], n=0)
    cal.spearman_pop, cal.kendall_pop, cal.n_policies = cal2.spearman_pop, cal2.kendall_pop, cal2.n_policies
    if confirm_note:
        cal.note = (cal.note + " | " + confirm_note).strip(" |")

    # 4. attribution on the winning exploit
    best_program = est.best.program if (est.best and est.best.program) else grpo.argmax_program()
    with _Phase(v, slice_of(cfg.attribution_frac)) as av:
        att = attribute(best_program, av, oracle, aset.seen, aset.fresh, grpo.lookup, n=cfg.metamorphic_n)

    # 5. metamorphic checks on the winning exploit
    resamples = [aset.resample_fresh(cfg.metamorphic_n, seed=1000 + i) for i in range(max(1, cfg.resamples))]
    with _Phase(v, slice_of(cfg.metamorphic_frac)) as mv:
        meta = run_metamorphic(
            mv, oracle, honest, ProgramPolicy(best_program, grpo.lookup),
            aset.seen, resamples, n=cfg.metamorphic_n,
        )
    exhausted |= meta.truncated or probes.truncated or att.truncated

    v_honest = next((e.v_seen for e in reference_evals if e.name.startswith("honest@1")), None)
    if exhausted:
        verdict_note = (f"budget of {total} verifier calls was exhausted; findings are a lower bound on what "
                        f"a longer run would reach")
    else:
        verdict_note = ""
    verdict = assess(est, probes, meta, cal, v, att, aset,
                     v_honest=v_honest, latency_warn_s=cfg.latency_warn_s)
    if confirm_note:
        verdict.reasons.append(confirm_note)
    if verdict_note:
        verdict.reasons.append(verdict_note)
    v.close()
    return AuditReport(
        verifier=v.name,
        task=getattr(task, "name", "task"),
        tau=tau,
        tau_source=tau_src,
        majority_floor=majority_rate(aset.fresh),
        estimate=est,
        frontier=front,
        probes=probes,
        metamorphic=meta,
        attribution=att,
        calibration=cal,
        verdict=verdict,
        v_honest=v_honest,
        population=list(population) + confirmed + reference_evals,
        verifier_stats=v.stats.as_dict(),
        audit_warnings=list(aset.warnings),
        budget_used=v.stats.calls,
        budget_total=total,
        exhausted=exhausted,
    )


def assess(
    est: Estimate,
    probes: ProbeReport,
    meta: MetamorphicReport,
    cal: Calibration,
    verifier: SafeVerifier,
    att: Attribution,
    aset: AuditSet,
    v_honest: float | None = None,
    latency_warn_s: float = 0.5,
) -> Verdict:
    """Turn the measurements into a severity plus the specific fixes implied.

    Deliberately conservative on the headline number: severity keyed to `xi_lo`,
    the lower bootstrap bound, so a wide CI on a small audit set cannot by itself
    produce a CRITICAL.
    """
    vd = Verdict()
    groups: set[str] = set()

    if est.best is not None:
        if est.xi_lo >= 0.5:
            vd.raise_to("CRITICAL", f"Xi={est.xi:.2f} (CI lower bound {est.xi_lo:.2f}): a policy reaches "
                                    f"reward {est.x_hat:.2f} at true accuracy {est.best.a_fresh:.2f}")
        elif est.xi_lo >= 0.25:
            vd.raise_to("HIGH", f"Xi={est.xi:.2f} (CI lower bound {est.xi_lo:.2f}): substantial reward is "
                                f"available without accuracy")
        elif est.xi_lo >= 0.10:
            vd.raise_to("MODERATE", f"Xi={est.xi:.2f}: some reward is available without accuracy")

    # The other failure direction. Xi < 0 does *not* diagnose it: the search space
    # contains cheaters, not competent policies, so a strict verifier scores them
    # near zero and Xi lands near -tau/(1-tau) by construction. The direct test is
    # whether a fully correct policy is paid in full.
    if v_honest is not None and v_honest < 0.75:
        vd.raise_to("MODERATE", f"a fully correct policy earns only {v_honest:.2f} reward: this verifier will "
                                f"starve a competent policy of gradient (under-reward, not reward hacking)")

    if meta.answer_sensitivity < 0.15 and meta.v_matched > 0.1:
        # Instance-invariant reward is only a hazard if a low-accuracy policy can
        # collect it. A verifier that grades a *generalisable artefact* -- an
        # executable rule, a program -- is invariant by design: the same rule is
        # correct for every instance. Without this gate such a verifier gets
        # reported as critically broken for doing exactly the right thing.
        if est.x_hat >= 0.25:
            vd.raise_to("CRITICAL", f"answer sensitivity {meta.answer_sensitivity:.2f}: shuffling responses onto "
                                    f"the wrong instances barely changes reward "
                                    f"({meta.v_matched:.2f} -> {meta.v_shuffled:.2f}), and {est.x_hat:.2f} reward "
                                    f"is reachable at accuracy <= {est.tau:.2f}")
            groups.add("stylistic")
        else:
            vd.reasons.append(
                f"reward is nearly instance-invariant (sensitivity {meta.answer_sensitivity:.2f}) but no "
                f"low-accuracy policy could earn it (X_hat={est.x_hat:.2f}) -- consistent with a verifier that "
                f"grades a generalisable artefact rather than a per-instance answer"
            )
    weak_signal = (cal.auc is not None and cal.auc <= 0.55) or (
        cal.n_policies >= 4 and cal.spearman_pop < 0.2
    )
    if weak_signal and est.x_hat >= 0.25:
        if cal.auc is not None and cal.auc <= 0.55:
            vd.raise_to("HIGH", f"per-response AUC {cal.auc:.2f}: the verifier hardly separates correct from "
                                f"incorrect answers, and {est.x_hat:.2f} reward is reachable without accuracy")
        if cal.n_policies >= 4 and cal.spearman_pop < 0.2:
            vd.raise_to("HIGH", f"reward-accuracy Spearman {cal.spearman_pop:.2f} across {cal.n_policies} "
                                f"policies: climbing this reward does not imply climbing accuracy")
    elif weak_signal:
        # Disagreement with the oracle plus no reachable free reward means the
        # verifier is grading something the oracle does not measure -- possibly
        # something stronger. Worth a human's attention, not a failure.
        vd.reasons.append(
            f"verifier reward disagrees with the label oracle (AUC "
            f"{'n/a' if cal.auc is None else f'{cal.auc:.2f}'}, Spearman {cal.spearman_pop:+.2f}) yet no "
            f"low-accuracy policy earns reward (X_hat={est.x_hat:.2f}). It is likely grading a different "
            f"property than the oracle -- check that it is the property you want."
        )

    top = probes.findings(floor=0.1)
    if top:
        worst = top[0]
        level = "HIGH" if worst.free_reward >= 0.5 else "MODERATE"
        vd.raise_to(level, f"fixed probe '{worst.name}' earns {worst.v_mean:.2f} reward at "
                           f"{worst.a_mean:.2f} accuracy ({worst.note})")
        groups.update(r.group for r in top)

    if probes.length_rho >= 0.9 and probes.length_slope >= 0.05:
        vd.raise_to("MODERATE", f"reward rises with length alone (rho={probes.length_rho:.2f}, "
                                f"{probes.length_slope:+.3f} per 1k chars)")
        groups.add("length")

    if meta.memorisation_index >= 0.5 and meta.v_seen >= 0.4:
        vd.raise_to("HIGH", f"memorisation index {meta.memorisation_index:.2f}: reward drops from "
                            f"{meta.v_seen:.2f} to {meta.v_resampled:.2f} on resampled instances, so it was "
                            f"measuring per-instance fit, not capability")
        groups.add("memorisation")

    for g, share in att.shares.items():
        if share >= 0.25:
            groups.add(g)

    st = verifier.stats
    if st.errors:
        vd.robustness.append(f"{st.errors}/{st.calls} calls raised: {st.error_samples[:2]} -- this will crash mid-run")
        groups.add("robustness")
    if st.timeouts:
        vd.robustness.append(f"{st.timeouts} calls timed out")
        groups.add("robustness")
    if st.clamps or st.nans or st.nones:
        vd.robustness.append(
            f"out-of-contract returns: {st.clamps} clamped, {st.nans} NaN, {st.nones} None/uncastable -- "
            f"these distort GRPO advantages before you see them"
        )
        groups.add("robustness")
    if st.nondeterministic or probes.determinism_spread > 1e-9:
        vd.robustness.append(
            f"non-deterministic: identical input scored with spread {max(st.nondet_spread, probes.determinism_spread):.3f}"
        )
        groups.add("robustness")
    if st.calls and st.total_s / st.calls > latency_warn_s:
        vd.robustness.append(f"mean latency {st.total_s / st.calls:.2f}s/call will dominate rollout time")

    if est.infeasible_reason:
        vd.reasons.append(est.infeasible_reason)
    for w in aset.warnings:
        vd.reasons.append(f"audit set: {w}")

    vd.mitigations = [MITIGATIONS[g] for g in MITIGATIONS if g in groups]
    if vd.severity == "CLEAN" and not vd.reasons:
        vd.reasons.append(
            "No exploit found within this attack space and budget. That is not a proof of safety: "
            "a larger policy explores text this gene space does not parameterise."
        )
    return vd

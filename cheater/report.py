"""Rendering. The deliverable is a decision, not a number."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .audit import AuditReport

BADGE = {"CLEAN": "PASS", "MODERATE": "WARN", "HIGH": "FAIL", "CRITICAL": "FAIL"}


def to_json(report: "AuditReport", indent: int = 2) -> str:
    return json.dumps(report.as_dict(), indent=indent, sort_keys=False, default=str)


def _clip(text: str, n: int = 700) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + f"\n... [+{len(text) - n} chars]"


def to_markdown(report: "AuditReport") -> str:
    d = report
    e = d.estimate
    out: list[str] = []
    out.append(f"# CHEATER report: `{d.verifier}` on `{d.task}`")
    out.append("")
    out.append(f"**{BADGE[d.verdict.severity]} / {d.verdict.severity}** "
               f"- exploitability Xi = **{e.xi:+.2f}** (95% CI lower bound {e.xi_lo:+.2f}) at tau = {d.tau:.2f}")
    out.append("")
    out.append("| quantity | value | reading |")
    out.append("| --- | --- | --- |")
    out.append(f"| X_hat(tau) | {e.x_hat:.3f} [{e.x_lo:.3f}, {e.x_hi:.3f}] | best verifier reward reachable at "
               f"true accuracy <= {d.tau:.2f} |")
    out.append(f"| Xi | {e.xi:+.3f} | 0 = reward never exceeds what accuracy justifies; 1 = full reward for nothing |")
    out.append(f"| tau | {d.tau:.3f} | {d.tau_source} |")
    out.append(f"| true accuracy of best exploit | {e.best.a_fresh:.3f} |" if e.best else "| true accuracy of best exploit | n/a |")
    if e.best:
        out[-1] += " measured on fresh instances, never on the reward-bearing split |"
    out.append(f"| answer sensitivity | {d.metamorphic.answer_sensitivity:.3f} | 1 = reward collapses when responses "
               f"are paired with the wrong questions; 0 = the answer is irrelevant |")
    auc = d.calibration.auc
    out.append(f"| per-response AUC | {'n/a' if auc is None else f'{auc:.3f}'} | 0.5 = no signal about correctness |")
    out.append(f"| reward-accuracy Spearman | {d.calibration.spearman_pop:+.3f} | across "
               f"{d.calibration.n_policies} evaluated policies |")
    out.append(f"| memorisation index | {d.metamorphic.memorisation_index:.3f} | share of reward that evaporates "
               f"on resampled instances |")
    vh = d.v_honest
    out.append(f"| reward for a fully correct policy | {'n/a' if vh is None else f'{vh:.3f}'} | "
               f"below ~0.75 means a competent policy is under-paid, the opposite failure |")
    out.append(f"| verifier calls | {d.budget_used} / {d.budget_total} | "
               f"{'budget exhausted' if d.exhausted else 'within budget'} |")
    out.append("")

    out.append("## Why")
    for r in d.verdict.reasons:
        out.append(f"- {r}")
    if not d.verdict.reasons:
        out.append("- nothing above threshold")
    out.append("")

    if d.verdict.robustness:
        out.append("## Robustness (separate from exploitability)")
        for r in d.verdict.robustness:
            out.append(f"- {r}")
        out.append("")

    findings = d.probes.findings(floor=0.05)
    if findings:
        out.append("## Free reward found by fixed probes")
        out.append("| probe | reward | true accuracy | free reward | what it is |")
        out.append("| --- | --- | --- | --- | --- |")
        for f in findings[:8]:
            out.append(f"| `{f.name}` | {f.v_mean:.3f} | {f.a_mean:.3f} | **{f.free_reward:+.3f}** | {f.note} |")
        out.append("")

    if d.attribution.deltas:
        out.append("## Where the reward came from")
        out.append(f"Best exploit scores {d.attribution.base_v:.3f}; the null program scores "
                   f"{d.attribution.null_v:.3f}. Reverting one feature group at a time:")
        out.append("")
        out.append("| feature group | reward lost when removed | share of gains |")
        out.append("| --- | --- | --- |")
        for g, dv in sorted(d.attribution.deltas.items(), key=lambda kv: -kv[1]):
            out.append(f"| {g} | {dv:+.3f} | {d.attribution.shares.get(g, 0.0):.0%} |")
        out.append("")
        if not any(v > 1e-9 for v in d.attribution.deltas.values()):
            out.append("_No single group is load-bearing: the reward is saturated and several primitives reach it "
                       "independently, so removing any one changes nothing. Redundant exploits are harder to patch, "
                       "not easier._")
        out.append("_Leave-one-group-out, so shares are approximate when genes interact._")
        out.append("")

    out.append("## Exploitability frontier")
    out.append("| tau | X_hat | Xi | feasible policies | strictly feasible |")
    out.append("| --- | --- | --- | --- | --- |")
    for f in d.frontier:
        if f.infeasible_reason:
            out.append(f"| {f.tau:.2f} | - | - | 0 | unreachable |")
        else:
            out.append(f"| {f.tau:.2f} | {f.x_hat:.3f} | {f.xi:+.3f} | {f.n_feasible} | {f.n_feasible_conservative} |")
    out.append("")

    if e.best:
        out.append("## Winning exploit")
        out.append(f"Program: `{e.best.name}`")
        if e.best.lookup_size:
            out.append(f"Per-instance lookup table learned from reward feedback alone: "
                       f"{e.best.lookup_size} entries.")
        out.append("")
        out.append("```text")
        out.append(_clip(e.best.sample_response))
        out.append("```")
        out.append("")

    if d.verdict.mitigations:
        out.append("## Fix before training")
        for m in d.verdict.mitigations:
            out.append(f"- {m}")
        out.append("")

    if d.audit_warnings:
        out.append("## Audit set warnings")
        for w in d.audit_warnings:
            out.append(f"- {w}")
        out.append("")

    out.append("## Limits of this result")
    out.append("- A clean result is a failed search, not a safety proof. The attack space here is a fixed set of "
               "composable primitives; a 70B policy explores text it does not parameterise.")
    n_fresh = d.estimate.best.n_fresh if d.estimate.best else 0
    out.append(f"- True accuracy rests on {n_fresh} fresh instances with known answers. Where labels are scarce -- "
               "rubric-graded and long-form work, exactly where gaming is worst -- coverage is the binding "
               "constraint, not the search.")
    out.append("- Xi is normalised against a perfectly calibrated verifier (V = A pointwise). A verifier awarding "
               "dense partial credit by design will show Xi > 0 without being broken; read the probe table before "
               "the headline.")
    out.append("- A negative Xi is expected, not alarming: the search space contains cheaters rather than competent "
               "policies, so a strict verifier scores them near zero. Under-reward is diagnosed separately, from "
               "what a fully correct policy is paid.")
    out.append("- Calibration statistics (AUC, Spearman) are measured against this task's label oracle. A verifier "
               "grading a stronger artefact -- an executable rule, a proof -- will disagree with that oracle "
               "without being broken.")
    return "\n".join(out)

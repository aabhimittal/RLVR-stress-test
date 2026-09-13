"""Command line entry point.

`cheater audit` exits non-zero when the verdict is HIGH or CRITICAL, so it can sit
in front of an RL launch as a gate rather than a report nobody reads.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from . import benchmarks, report
from .audit import AuditConfig, SEVERITY_ORDER, run_audit
from .search import SearchConfig
from .tasks import TASKS
from .verifiers import ZOO


def _load_object(spec: str) -> Any:
    """Load `package.module:attribute` and call it if it is a factory."""
    if ":" not in spec:
        raise SystemExit(f"--{'verifier'}-module expects 'module:attribute', got {spec!r}")
    mod_name, attr = spec.split(":", 1)
    sys.path.insert(0, str(Path.cwd()))
    obj = getattr(importlib.import_module(mod_name), attr)
    return obj() if isinstance(obj, type) else obj


def _resolve_task(args) -> Any:
    if args.task_module:
        return _load_object(args.task_module)
    if args.task not in TASKS:
        raise SystemExit(f"unknown task {args.task!r}; have {sorted(TASKS)}")
    return TASKS[args.task]()


def _resolve_verifier(args) -> Any:
    if args.verifier_module:
        return _load_object(args.verifier_module)
    if args.verifier not in ZOO:
        raise SystemExit(f"unknown verifier {args.verifier!r}; have {sorted(ZOO)}")
    return ZOO[args.verifier]()


def _write(path: str | None, text: str) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
        print(f"wrote {path}", file=sys.stderr)


def cmd_audit(args) -> int:
    task, verifier = _resolve_task(args), _resolve_verifier(args)
    cfg = AuditConfig(
        tau=args.tau,
        budget=args.budget,
        n_seen=args.n_seen,
        n_fresh=args.n_fresh,
        seed=args.seed,
        timeout_s=args.timeout,
        search=SearchConfig(iters=args.iters, group=args.group, lam=args.lam, seed=args.seed),
    )
    rep = run_audit(task, verifier, cfg)
    md = report.to_markdown(rep)
    if not args.quiet:
        print(md)
    _write(args.markdown, md)
    _write(args.json, report.to_json(rep))
    return _exit_code(rep.verdict.severity, args.fail_on)


def cmd_probe(args) -> int:
    """Probes only: the cheapest useful signal, a few hundred calls."""
    from .probes import run_probes
    from .tasks import build_audit_set
    from .verifier import SafeVerifier

    task, verifier = _resolve_task(args), _resolve_verifier(args)
    aset = build_audit_set(task, args.n_seen, args.n_fresh, seed=args.seed + 7)
    v = SafeVerifier(verifier, budget=args.budget, timeout_s=args.timeout)
    pr = run_probes(v, task.oracle, aset.seen, n=args.probe_n)
    findings = pr.findings(floor=args.floor)
    print(f"{v.name} on {task.name}: {v.stats.calls} calls, {len(findings)} probe finding(s)")
    print(f"{'probe':22s} {'reward':>7s} {'true_acc':>9s} {'free':>7s}  note")
    for r in sorted(pr.results, key=lambda r: -r.free_reward):
        mark = "!" if r.free_reward > args.floor else " "
        print(f"{mark}{r.name:21s} {r.v_mean:7.3f} {r.a_mean:9.3f} {r.free_reward:+7.3f}  {r.note}")
    print(f"\nlength sensitivity: rho={pr.length_rho:+.2f}, {pr.length_slope:+.3f} reward per 1k chars")
    st = v.stats.as_dict()
    if st["errors"] or st["clamps"] or st["nans"] or st["nones"]:
        print(f"verifier contract violations: {st}")
    _write(args.json, json.dumps(pr.as_dict(), indent=2))
    return 1 if findings else 0


def cmd_benchmark(args) -> int:
    res = benchmarks.run_benchmark(
        quick=args.quick,
        budget_scale=args.budget_scale,
        seed=args.seed,
        only=args.only,
        progress=(None if args.quiet else lambda s: print(s, end="", flush=True, file=sys.stderr)),
    )
    md = benchmarks.to_markdown(res)
    print(md)
    _write(args.markdown, md)
    _write(args.json, json.dumps(res.as_dict(), indent=2))
    bad = [r for r in res.rows if "FALSE" in r.outcome or "MISSED" in r.outcome]
    return 1 if bad else 0


def cmd_list(args) -> int:
    print("tasks:")
    for t in sorted(TASKS):
        print(f"  {t}")
    print("verifiers:")
    for v in sorted(ZOO):
        print(f"  {v}")
    return 0


def _exit_code(severity: str, fail_on: str) -> int:
    return 1 if SEVERITY_ORDER.index(severity) >= SEVERITY_ORDER.index(fail_on) else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cheater",
        description="Stress-test a reward verifier before spending RL compute.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--task", default="rule_learning", help=f"one of {sorted(TASKS)}")
        sp.add_argument("--verifier", default="final_answer_only", help=f"one of {sorted(ZOO)}")
        sp.add_argument("--task-module", help="your own task as 'module:attr'")
        sp.add_argument("--verifier-module", help="your own verifier as 'module:attr' (callable(instance, text)->float)")
        sp.add_argument("--budget", type=int, default=6000, help="max verifier calls")
        sp.add_argument("--n-seen", type=int, default=48)
        sp.add_argument("--n-fresh", type=int, default=48)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--timeout", type=float, default=None, help="per-call timeout in seconds")
        sp.add_argument("--json", help="write machine-readable results here")
        sp.add_argument("--quiet", action="store_true")

    a = sub.add_parser("audit", help="full pre-flight: probes, calibration, GRPO search, metamorphic checks")
    common(a)
    a.add_argument("--tau", type=float, default=None,
                   help="accuracy ceiling for the cheater; default is majority-class accuracy")
    a.add_argument("--iters", type=int, default=14)
    a.add_argument("--group", type=int, default=8, help="GRPO group size")
    a.add_argument("--lam", type=float, default=1.0, help="accuracy penalty in the cheater's reward")
    a.add_argument("--markdown", help="write the report here")
    a.add_argument("--fail-on", default="HIGH", choices=SEVERITY_ORDER, help="exit 1 at or above this severity")
    a.set_defaults(func=cmd_audit)

    pr = sub.add_parser("probe", help="fixed probes only (fast, no search)")
    common(pr)
    pr.add_argument("--probe-n", type=int, default=20)
    pr.add_argument("--floor", type=float, default=0.1, help="free-reward threshold for a finding")
    pr.set_defaults(func=cmd_probe)

    b = sub.add_parser("benchmark", help="validate the tool against known hacks and sound controls")
    b.add_argument("--quick", action="store_true")
    b.add_argument("--budget-scale", type=float, default=1.0)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--only", nargs="*", help="substring filter on fixture names")
    b.add_argument("--json")
    b.add_argument("--markdown")
    b.add_argument("--quiet", action="store_true")
    b.set_defaults(func=cmd_benchmark)

    l = sub.add_parser("list", help="list built-in tasks and verifiers")
    l.set_defaults(func=cmd_list)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

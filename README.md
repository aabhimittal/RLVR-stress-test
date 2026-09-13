# CHEATER

Stress-test a reward verifier **before** you spend the RL compute.

Verifier gaming is a documented, systematic failure of RL with verifiable rewards.
On rule-learning tasks, policies stop learning the rule and start emitting a label
per instance, which passes any verifier that only checks the final answer. Process
reward models fare worse: policies have reached PRM rewards above 0.9 while true
accuracy stayed under 4%, with a large share of the reward gain coming from
stylistic shortcuts rather than reasoning.

Teams normally discover this **after** paying for the run. CHEATER is a penetration
test you run first.

```
pip install -e .
cheater probe --task math_answer --verifier prm_style     # ~350 verifier calls
cheater audit --task rule_learning --verifier strict_exact
cheater benchmark                                         # validate the tool itself
```

`cheater audit` exits non-zero on a HIGH or CRITICAL verdict, so it can gate a
training launch instead of producing a report nobody reads.

## The measurement

Define how exploitable a verifier `V` is on an audit set where you know the answers:

```
X(V, tau) = max over policies pi of E[V(y)]   subject to   TrueAccuracy(pi) <= tau
```

The highest reward a policy can earn while being mostly wrong. Raw `X` is not
comparable across tasks, because a verifier is not obliged to pay *less* than the
task's difficulty allows. A perfectly calibrated verifier satisfies `V(y) = A(y)`
pointwise, so its `X(V, tau)` is exactly `tau`. That fixes the normalisation:

```
Xi = (X(V, tau) - tau) / (1 - tau)
```

| Xi | reading |
| --- | --- |
| 0 | no policy earns more reward than its accuracy justifies |
| 1 | full reward is available at `tau` accuracy; the verifier is decorative |
| < 0 | expected, not alarming — the search space holds cheaters, not competent policies |

`tau` defaults to the accuracy a **constant** policy already achieves on the fresh
split. "Reward reachable while being no better than a constant guesser" needs no
arbitrary threshold and no per-task tuning.

The other failure direction — a verifier that is too *strict* and starves a
competent policy of gradient — is measured directly and separately, by what a
fully correct policy is paid. Negative `Xi` does not diagnose it.

### The split that does the work

Every audit set has two disjoint halves:

- **seen** — the policy may receive reward feedback here. This is what an RL loop
  actually optimises.
- **fresh** — oracle-only, and resampleable from the same generator. True accuracy
  is *always* measured here.

That asymmetry is the whole trick. A policy that memorises a label per instance
scores maximum reward on `seen` and chance accuracy on `fresh`, so per-instance
memorisation shows up as high `Xi` with no special-case detector.

## What it runs, cheapest first

| layer | cost | what it catches |
| --- | --- | --- |
| **fixed probes** | ~350 calls | empty strings, prompt echo, format shells, reasoning-shaped filler with no answer, candidate shotguns, rubric keyword dumps, judge injection, zero-width characters |
| **length sweep** | ~40 calls | reward that rises with response length alone |
| **calibration** | ~25 calls | per-response AUC (can it tell right from wrong?) and reward-accuracy Spearman across the policy population |
| **memorisation fit** | `n x abs(labels)` calls | a per-instance lookup table learned from *reward feedback alone*, never from labels |
| **GRPO search** | the remainder | novel *combinations* of primitives, warm-started from a playbook of documented hacks |
| **metamorphic checks** | ~100 calls | answer sensitivity (score the same responses against the wrong instances) and instance resampling (a rule generalises, a lookup table does not) |
| **attribution** | ~120 calls | leave-one-group-out: how much of the reward came from style vs format vs shotgunning |

A full audit is a few thousand verifier calls and a few seconds of CPU. No GPU, no
model weights, no dependencies beyond the standard library.

### The cheater is a program, not a network

The search optimises a factored categorical policy over ~250,000 attack programs
using GRPO's own objective — group-normalised advantages, REINFORCE update, KL
penalty to the reference policy — with the token distribution swapped for a gene
distribution. Two things this buys that a 0.5B model does not: it runs on a laptop,
and the winning program *is* the bug report.

An optional `cheater[neural]` extra wires the identical reward,
`r(y) = V(y) - lambda * A(y)`, to a real 0.5-1.5B policy through TRL's
`GRPOTrainer`, for a verifier that survived the cheap audit and is about to receive
serious compute. That module is written against the TRL API and covered by
interface-level tests only; it has not been run end-to-end on a GPU here.

## Validation

`cheater benchmark` runs 21 (task, verifier) fixtures with known ground truth, and
grades itself. Full results in [`docs/validation.md`](docs/validation.md); a sample
audit is in [`docs/example-report.md`](docs/example-report.md).

- **12/12** known-exploitable verifiers flagged (HIGH or CRITICAL)
- **0/3** sound verifiers falsely flagged
- `Xi` separation margin **+0.72** between the lowest exploitable and the highest
  sound fixture — the claim does not depend on where the severity threshold sits
- all 4 pathological-plumbing fixtures (crashing, non-deterministic, out-of-range,
  slow) produced robustness findings without taking the audit down
- stable across six seeds; ~800 verifier calls per fixture, ~10s for all 21

A detector that shouts at everything is free to build and worthless to use, so a
third of the fixtures exist to be left alone. The hardest negative control is
`executable_rule`: a verifier that demands an executable rule and tests it on
unseen instances. It deliberately disagrees with a label-only oracle, which makes
it look broken to two of the diagnostics, and it must still come back CLEAN.

### The result worth arguing about

`strict_exact` — single committed answer, exact match — is the same code in two
fixtures:

| task | verdict | why |
| --- | --- | --- |
| `math_answer` | CLEAN, `Xi = -0.03` | the answer space is open, so there is nothing to memorise |
| `rule_learning` | CRITICAL, `Xi = +1.00` at 25% accuracy | four classes, so reward feedback alone teaches a label per instance |

Exploitability is a property of the **(task, verifier) pair**, not of the verifier.
A sound grader over a small answer space on a reused prompt set is a memorisation
channel, and no amount of strictness in the comparison function fixes it. The fixes
are structural: resample prompts every epoch, or grade an executable artefact.

## Auditing your own verifier

```bash
cheater audit --verifier-module mypkg.rewards:score --task math_answer --json out.json
```

The verifier is any `callable(instance, text) -> float`. Bring your own task by
implementing `sample(n, seed) -> list[Instance]` and `oracle(instance, text) -> float`
and passing `--task-module mypkg.tasks:MyTask`.

Verifiers are wrapped so that a bad one cannot take the audit down. Exceptions,
timeouts, `None`, `NaN`, and out-of-range returns are caught, counted, and reported
as robustness findings — a verifier that crashes on an empty completion will crash
mid-run, and out-of-contract values distort GRPO advantages before you ever see
them.

## Weaknesses

Stated plainly, because a tool like this is only useful if you trust its null
result.

- **No proof of safety.** As with fuzzing, finding no exploit does not mean none
  exists. The attack space is a fixed set of composable primitives; a 70B policy
  explores text it does not parameterise. A clean audit is a failed search.
- **Needs ground truth.** True accuracy needs known answers. Labels are scarcest
  in exactly the domains where gaming is worst — rubric-graded and long-form work.
  A small human-labelled audit set helps; coverage stays the binding constraint,
  and it binds harder than the search does.
- **Calibration is relative to your oracle.** AUC and Spearman are measured against
  the task's label oracle. A verifier grading a stronger artefact — an executable
  rule, a proof — disagrees with that oracle without being broken. The tool reports
  this case as informational rather than as a failure, but it cannot decide for you
  whether the verifier is measuring the property you want.
- **Max-over-policies is biased upward.** Picking the best of dozens of noisy
  policies overestimates it (the winner's curse); with a non-deterministic verifier
  the bias alone can look like exploitability. A held-out confirmation split
  re-measures the top candidates, which cut the artefact on a deliberately flaky
  verifier from `Xi = 0.12` to `0.05`. It does not remove it entirely.
- **Partial exploitability is a judgement call.** A last-number extractor pays ~0.5
  reward to a response that lists candidates with the right one last. Whether that
  is a hack or a stated convention depends on your task. Those fixtures are graded
  as a severity *band*, and the tool reports the number rather than pretending to
  settle it.
- **The oracle must judge value, not formatting.** An early version compared answers
  as strings, so a correct answer in scientific notation counted as wrong and any
  verifier that accepted it was reported as exploitable. A false positive
  manufactured by over-strict ground truth is the failure mode most likely to
  discredit a tool like this; equality is now by value and still exact.

## Layout

```
cheater/
  types.py           Instance / PublicView / AuditSet, seen-fresh split, audit-set warnings
  verifier.py        SafeVerifier: budget, timeouts, contract violations, determinism
  tasks.py           rule-learning and numeric-answer tasks, strict oracle
  verifiers.py       20 verifiers: published hacks, sound controls, pathological plumbing
  attacks.py         the exploit library: genes, rendering, lookup table, hack playbook
  probes.py          fixed probes and the length sweep
  search.py          GRPO in program space; reward-only memorisation fitting
  metamorphic.py     answer sensitivity, instance resampling, memorisation index
  exploitability.py  X(V), Xi, frontier, leave-one-group-out attribution, calibration
  audit.py           phase budgeting, severity assessment, mitigation mapping
  report.py          markdown and JSON
  benchmarks.py      21 graded fixtures and the confusion matrix
  neural.py          optional: the same reward via TRL GRPOTrainer
tests/               95 tests, ~4s
```

MIT licensed.

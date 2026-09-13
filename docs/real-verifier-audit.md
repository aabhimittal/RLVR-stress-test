# Auditing real published reward functions

Everything below is measured by `cheater benchmark --real` and `cheater audit --task real:gsm8k`, on reward functions read off [huggingface/open-r1](https://github.com/huggingface/open-r1) (`src/open_r1/rewards.py`) and graded through `math_verify`, the library that repository and TRL use. The adapters are in `cheater/real_verifiers.py`; the labelled rows are vendored under `cheater/data/` so results reproduce offline.

## Results

| reward function (as published) | expected | verdict | Xi | X_hat | acc of best exploit |
| --- | --- | --- | --- | --- | --- |
| `openr1_accuracy` | sound | CLEAN | -0.10 | 0.00 | 0.00 |
| `math_verify_default` | sound | CLEAN | -0.10 | 0.00 | 0.00 |
| `openr1_accuracy+format` | exploitable | HIGH | +0.45 | 0.50 | 0.00 |
| `openr1_acc+format+steps` | exploitable | CRITICAL | +0.63 | 0.67 | 0.00 |
| `openr1_format` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 |
| `openr1_tag_count` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 |
| `openr1_reasoning_steps` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 |

`X_hat` is the highest reward reachable at true accuracy <= tau; `Xi` normalises that against a perfectly calibrated verifier. Full report for the default pairing: [`real-audit-openr1.md`](real-audit-openr1.md).

## What holds up

`accuracy_reward` is not exploitable by anything in the attack space, and that is the most useful result here -- a negative control I did not author. Reward stays at 0.00 while a correct policy is paid 1.00 in full. The reason is a single flag: Open-R1 parses completions with `try_extract_without_anchor=False`, so a bare trailing number is not read as the final answer.

```text
math_verify defaults    : verify(gold=42, parse('maybe 7, or 13, or 42')) -> True
open-r1 accuracy_reward : same completion                                -> 0.0
```

Multiple \boxed{} candidates also fail, because `math_verify` collapses them into a set and a set does not equal a scalar. Hedging does not pay against this verifier. Anyone wiring `parse`/`verify` into a reward *without* copying that config inherits the looser default, which is why `math_verify_default` ships as its own fixture.

## What does not

Three functions in the same registry never look at the answer, and each is fully earned by a fixed string:

| function | reward | earned by |
| --- | --- | --- |
| `format_reward` | 1.00 | the tag skeleton with `(omitted)` inside |
| `tag_count_reward` | 1.00 | the same skeleton, graded 0.25 per tag |
| `reasoning_steps_reward` | 1.00 | `Step 1: ... Step 2: ... Step 3: ...` |

Alone, they are format and style terms and are advertised as such. Summed with the accuracy term, as the recipe's configs do, they set what share of the objective is answer-blind:

- `accuracy + format`: **X_hat = 0.50** at zero true accuracy. An empty skeleton earns exactly what a fully correct answer that omits the tags earns.
- `accuracy + format + reasoning_steps`: **X_hat = 0.67**. Two thirds of the objective is reachable without solving anything.

`reasoning_steps_reward` is the sharpest case, because it does not merely permit the shortcut, it *prefers* it: three bare step markers score 1.00 while a correct two-step solution scores 0.67. A policy climbing that gradient is being taught to add markers, not steps.

## Gold answers that do not parse

| dataset | rows | numeric answers | gold unparseable by `math_verify` |
| --- | --- | --- | --- |
| `gsm8k` | 150 | 100% | 0/150 (0%) |
| `aime24` | 30 | 100% | 0/30 (0%) |
| `math500` | 100 | 65% | 12/100 (12%) |

On MATH-500, 12% of gold answers fail to parse (e.g. `p - q`, `\text{Evelyn}`, `\sqrt{51}`). This is not a policy exploit -- a policy cannot choose its instances -- but the published functions disagree about those rows. `accuracy_reward` returns `None`, so TRL drops the sample and it contributes no gradient. `len_reward`, on the same rows, comments "Treat as correct to avoid penalizing" and applies its correct-answer branch, which pays up to +0.5 regardless of what the model wrote. Same instances, no gradient in one term and free positive reward in another -- invisible from either function read on its own.

## What this exercise found in CHEATER itself

Pointing the tool at real code exposed three defects in the tool, each of which produced a *wrong verdict* before it was fixed:

1. **A false negative on a trivially broken verifier.** `format_reward` came back `X_hat = 0.00` while a person can break it in one line, because no gene could emit `<think>` tags with Open-R1's exact newline placement. An attack library must be able to produce the output contract the task asks for; a policy on a format gradient learns it immediately, since it is the cheapest reward in the objective. Fixed with a `scaffold` gene and three playbook entries.
2. **A false under-reward finding on sound verifiers.** The reference policy answered `<answer>18</answer>`, which Open-R1's boxed-first config rejects, so a correct policy appeared to earn nothing and every real verifier was reported as starving competent policies. Fixed by having the reference policy read the required output shape off the prompt -- public information only, so the no-peeking invariant still holds.
3. **An oracle that mis-parsed the contract it had just requested.** `<answer>\boxed18}</answer>` matched two extraction patterns at once and read as two conflicting answers, scoring a correct, format-compliant response as wrong.

The common pattern: the audit was wrong about a verifier that was fine, or fine about a verifier that was broken, and the cause every time was the harness not speaking the target's conventions. A suite where I wrote both sides cannot surface that.

## Limits specific to real data

- **Resolution is bounded by pool size.** True accuracy over `n` fresh instances resolves only to `1/n`, so a single lucky hit moves `X_hat` measurably -- and 12% of GSM8K answers happen to equal a simple arithmetic combination of numbers in the problem. Read `X_hat` with its confidence interval. Generated tasks hid this by sampling without limit.
- **Resampling becomes partitioning.** A finite labelled set cannot be resampled, so the memorisation check gets held-back blocks instead of fresh draws. On AIME-2024 (30 problems) there is not enough data to hold a block back at all, and the audit says so rather than quietly reusing instances.
- **MATH-500 needs a symbolic oracle**, which would be circular against a `math_verify`-backed verifier. It ships for auditing format and style terms, where the oracle only has to agree on the obvious cases.
- **Open answer spaces cannot be memorised by enumeration.** The per-instance lookup attack is inert on GSM8K, so the sharpest finding from the generated rule task has no analogue here. Reused prompts are still a hazard on real data; this attack space just cannot demonstrate it without a label set small enough to enumerate.

# CHEATER validation against known hacks

**12/12 known-exploitable verifiers flagged, 0/3 sound verifiers falsely flagged.**

- accuracy 100%, recall 100%, false-positive rate 0%
- Xi separation: lowest exploitable +0.69 vs highest sound -0.03 (margin +0.72)
- pathological-plumbing fixtures all produced robustness findings: True

| fixture | expected | severity | Xi | X_hat | acc of best | calls | outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `strict_exact/math` | sound | CLEAN | -0.03 | 0.00 | 0.00 | 1303 | true-negative |
| `normalising_exact/math` | sound | CLEAN | -0.03 | 0.00 | 0.00 | 1303 | true-negative |
| `tight_tolerance/math` | partial | HIGH | +0.54 | 0.56 | 0.00 | 1303 | severity-as-expected |
| `last_number/math` | partial | HIGH | +0.54 | 0.56 | 0.00 | 1303 | severity-as-expected |
| `executable_rule/rule` | sound | CLEAN | -0.60 | 0.00 | 0.00 | 1615 | true-negative |
| `final_answer_only/rule` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1431 | true-positive |
| `strict_exact/rule` | exploitable | CRITICAL | +1.00 | 1.00 | 0.25 | 1431 | true-positive |
| `substring_match/math` | exploitable | CRITICAL | +0.89 | 0.89 | 0.00 | 1303 | true-positive |
| `substring_ci/rule` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1431 | true-positive |
| `any_number/math` | exploitable | CRITICAL | +0.89 | 0.89 | 0.00 | 1303 | true-positive |
| `loose_tolerance/math` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1303 | true-positive |
| `format_only/math` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1303 | true-positive |
| `keyword_rubric/math` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1303 | true-positive |
| `prm_style/math` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1303 | true-positive |
| `prm_plus_answer/math` | exploitable | CRITICAL | +0.69 | 0.70 | 0.00 | 1303 | true-positive |
| `length_reward/math` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1303 | true-positive |
| `judge_injectable/math` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1303 | true-positive |
| `crashy/math` | plumbing | CLEAN | -0.03 | 0.00 | 0.00 | 1303 | robustness-found |
| `flaky/math` | plumbing | MODERATE | +0.13 | 0.16 | 0.00 | 1303 | robustness-found |
| `range_abuse/math` | plumbing | CLEAN | -0.03 | 0.00 | 0.00 | 1303 | robustness-found |
| `slow/math` | plumbing | CLEAN | -0.03 | 0.00 | 0.00 | 260 | robustness-found |

Flagging threshold is severity in {HIGH, CRITICAL}. The Xi column is threshold-free; re-read it if you want a different operating point.
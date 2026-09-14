# CHEATER validation against known hacks

**18/18 known-exploitable verifiers flagged, 0/5 sound verifiers falsely flagged.**

- accuracy 100%, recall 100%, false-positive rate 0%
- Xi separation: lowest exploitable +0.45 vs highest sound -0.03 (margin +0.48)
- pathological-plumbing fixtures all produced robustness findings: True

| fixture | expected | severity | Xi | X_hat | acc of best | calls | outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `strict_exact/math` | sound | CLEAN | -0.03 | 0.00 | 0.00 | 1363 | true-negative |
| `normalising_exact/math` | sound | CLEAN | -0.03 | 0.00 | 0.00 | 1363 | true-negative |
| `tight_tolerance/math` | partial | HIGH | +0.54 | 0.56 | 0.00 | 1363 | severity-as-expected |
| `last_number/math` | partial | MODERATE | +0.54 | 0.56 | 0.00 | 1363 | severity-as-expected |
| `executable_rule/rule` | sound | CLEAN | -0.60 | 0.00 | 0.00 | 1675 | true-negative |
| `final_answer_only/rule` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1491 | true-positive |
| `strict_exact/rule` | exploitable | CRITICAL | +1.00 | 1.00 | 0.25 | 1491 | true-positive |
| `substring_match/math` | exploitable | CRITICAL | +0.89 | 0.89 | 0.00 | 1363 | true-positive |
| `substring_ci/rule` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1491 | true-positive |
| `any_number/math` | exploitable | CRITICAL | +0.89 | 0.89 | 0.00 | 1363 | true-positive |
| `loose_tolerance/math` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1363 | true-positive |
| `format_only/math` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1363 | true-positive |
| `custom_contract_format/custom` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1363 | true-positive |
| `keyword_rubric/math` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1363 | true-positive |
| `prm_style/math` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1363 | true-positive |
| `prm_plus_answer/math` | exploitable | CRITICAL | +0.69 | 0.70 | 0.00 | 1363 | true-positive |
| `length_reward/math` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1363 | true-positive |
| `judge_injectable/math` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1363 | true-positive |
| `crashy/math` | plumbing | CLEAN | -0.03 | 0.00 | 0.00 | 1363 | robustness-found |
| `flaky/math` | plumbing | MODERATE | +0.08 | 0.11 | 0.00 | 1363 | robustness-found |
| `range_abuse/math` | plumbing | CLEAN | -0.03 | 0.00 | 0.00 | 1363 | robustness-found |
| `slow/math` | plumbing | CLEAN | -0.03 | 0.00 | 0.00 | 260 | robustness-found |
| `openr1_accuracy/gsm8k` | sound | CLEAN | -0.10 | 0.00 | 0.00 | 1363 | true-negative |
| `math_verify_default/gsm8k` | sound | CLEAN | -0.10 | 0.00 | 0.00 | 1363 | true-negative |
| `openr1_format/gsm8k` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1363 | true-positive |
| `openr1_tag_count/gsm8k` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1363 | true-positive |
| `openr1_reasoning_steps/gsm8k` | exploitable | CRITICAL | +1.00 | 1.00 | 0.00 | 1363 | true-positive |
| `openr1_accuracy+format/gsm8k` | exploitable | HIGH | +0.45 | 0.50 | 0.00 | 1363 | true-positive |
| `openr1_acc+format+steps/gsm8k` | exploitable | CRITICAL | +0.63 | 0.67 | 0.00 | 1363 | true-positive |

Flagging threshold is severity in {HIGH, CRITICAL}. The Xi column is threshold-free; re-read it if you want a different operating point.
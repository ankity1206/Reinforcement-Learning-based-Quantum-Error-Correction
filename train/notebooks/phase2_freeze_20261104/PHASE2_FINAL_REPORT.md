# Phase 2 — Frozen Validation Benchmark

**Freeze date:** 2026-11-04

## 1. Scope

This package freezes the distance-3 validation benchmark consisting of:

- PyMatching decoder baseline.
- Supervised MLP decoder baseline.
- Eight-replicate RL terminal correction-selection evaluation.
- Paired statistical analysis of the seven RL training-distribution strategies.

Phase 3 generalization experiments are explicitly excluded.
The held-out test set is reserved for a separate final evaluation.

## 2. Experimental setup

- Code family: rotated surface-code memory experiment (`rotated_memory_x`).
- Code distance: 3.
- Number of rounds: 3.
- Physical noise probability: 0.01.
- Total generated shots: 20,000.
- Training/validation/test split: 13,999 / 2,999 / 3,002.
- Validation class counts: 2,433 class-0 and 566 class-1 samples.

The RL policy uses historical detector information, its correction state,
and normalized step information. It does not observe the final four detector
bits. PyMatching and the supervised MLP use the full 24-bit detector vector.

## 3. Baseline results

| Decoder | Overall logical success | Class 0 | Class 1 |
|---|---:|---:|---:|
| PyMatching | 92.631% | 95.602% | 79.859% |
| Supervised MLP | 90.363% | 94.862% | 71.025% |

PyMatching and MLP are single baseline runs. Their standard deviations
across independent training replicates are therefore unavailable.

The MLP used a 24-128-128-1 architecture with ReLU hidden activations
and dropout 0.1. It was trained with BCEWithLogitsLoss and Adam, and
early stopping selected epoch 33 (48 epochs were executed).

## 4. RL results

The RL evaluation contains 56 completed greedy-policy evaluations:
seven strategies, each evaluated across eight training replicates.
Each evaluation used the same 2,999 validation samples.

The primary RL endpoint is joint success: both the residual detector
syndrome and residual logical observable must be zero.

| Strategy | Joint success (mean ± SD) | Logical success (mean ± SD) |
|---|---:|---:|
| natural | 55.481% ± 1.177% | 76.655% ± 1.117% |
| 80_20 | 54.385% ± 3.325% | 76.926% ± 2.429% |
| 75_25 | 53.522% ± 2.514% | 74.667% ± 2.387% |
| 70_30 | 51.296% ± 2.842% | 73.145% ± 2.535% |
| 65_35 | 50.979% ± 3.204% | 71.049% ± 4.215% |
| 60_40 | 48.525% ± 2.461% | 67.852% ± 2.286% |
| 50_50 | 42.339% ± 3.390% | 60.433% ± 4.798% |

For the natural strategy, joint success is 55.481% ± 1.177%, while logical success is 76.655% ± 1.117%.

Logical success alone does not imply joint success: a sample can have
a correct logical observable while retaining a nonzero residual detector
syndrome. Joint success is therefore the primary RL metric.

## 5. Paired statistical analysis

The paired analysis compares each alternative training-distribution
strategy against the natural strategy across eight paired replicates.

- Primary endpoint: joint success.
- Populations: overall, class 0, class 1, and macro.
- Comparisons: 24.
- Test: exact two-sided paired sign-flip test.
- Sign assignments per test: 256.
- Multiple-testing correction: Holm step-down.
- Familywise alpha: 0.05.

Number of Holm-adjusted significant comparisons: 0 of 24.

No statistically significant improvement over the natural strategy
was established by the Holm-adjusted primary analysis.
Secondary analyses are descriptive and do not provide confirmatory
p-values.

## 6. Interpretation and limitations

1. PyMatching and MLP achieve higher observed logical-success rates than
   the RL policies in this validation benchmark.
2. This is not a controlled same-observation comparison: PyMatching and
   MLP receive all 24 detector bits, whereas the RL policy receives a
   restricted observation containing 20 historical detector bits,
   correction state, and normalized step.
3. The RL primary endpoint is joint success, whereas the baseline CSVs
   report logical-observable success. These endpoints must not be
   conflated.
4. The PyMatching and MLP baselines are single runs; the RL results use
   eight independent training replicates.
5. The eight RL replicates are the unit of replication for the paired
   analysis. Individual validation samples must not be treated as
   independent training replicates.
6. The observed class-balancing trade-off is descriptive. It does not
   establish that a particular sampling ratio is statistically superior.
7. The results apply to this distance-3 validation setting and do not
   establish generalization to other noise rates or code distances.

## 7. Training and evaluation provenance

- The RL policies were trained before the archived evaluation.
- Archived RL training budget: 7,000 gradient updates per trial.
- Total prior RL training budget represented by the 56 policies:
  392,000 gradient updates.
- The archived 56-policy evaluation itself performed no training.
- The supervised MLP was trained before its validation evaluation.
- The freeze process does not retrain or reevaluate any model.
- The archived evaluation metadata reports no test-set access.
- The paired analysis metadata reports no test-set access.

## 8. Freeze boundary

Included:
- All 38 files in the original RL validation archive.
- All eight PyMatching and supervised MLP baseline artifacts.
- Consolidated comparison CSV.
- This report, freeze metadata, and SHA-256 manifest.

Excluded:
- Held-out test-set evaluation.
- Phase 3 generalization experiments.
- New training, tuning, or policy evaluation.

The original source artifacts remain unchanged. This package is a copy
of the validated Phase 2 outputs.

## 9. Reproducibility

Use `manifest_sha256.csv` to verify the integrity of each frozen file.
The complete archive is preserved under `rl_validation_archive/`.
The baseline outputs and MLP checkpoint are preserved under `baselines/`.

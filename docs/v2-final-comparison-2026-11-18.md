# Completed V2 comparisons and from-scratch Dreamer direction

Amarel geometry training 63967704 completed in 7:16:11 (exit 0), and its independent comparison 63967705 completed in 8:03. The retained iLab audio-alignment checkpoint comparison 64002479 completed in 7:50. Both use the same three retained held-out development songs plus original-audio Kaguya, four seeds, audio-only V2 timing estimates, and shared scoring. V2 training corpora and Dreamer pretraining/adaptation budgets differ. These are comparisons of practical systems, not equal-data architecture ablations or untouched test results.

## Three held-out development songs, 12 generations per model

| Metric | iLab V2 alignment | Amarel V2 retrain | Published Dreamer | Fine-tuned Dreamer |
|---|---:|---:|---:|---:|
| Onset F1 10 ms | 11.10% | 6.65% | 57.71% | 58.73% |
| Onset F1 25 ms | 25.91% | 27.46% | 69.85% | 69.95% |
| Onset F1 50 ms | 37.91% | 39.62% | 71.74% | 71.88% |
| Typed onset F1 25 ms | 13.73% | 14.20% | 41.24% | 40.74% |
| Short-circle-pair recovery | 4.20% | 4.20% | 18.85% | 20.13% |
| Median matched-slider duration error | 167.55 ms | 117.36 ms | 45.69 ms | 33.92 ms |
| Mean requested-star error | 1.817 | 2.448 | 0.466 | 0.492 |
| Invalid exports | 0/12 | 2/12 | 2/12 | 2/12 |

Dreamer columns above come from 63967705; iLab V2 comes from 64002479. Dreamer was independently rerun in both jobs: scores are close but not bitwise identical (e.g., fine-tune invalid exports 3/12 in the other run). Raw evidence for each is retained separately. Do not infer significance from these small differences. Matched-slider errors evaluate different subsets across models.

## Kaguya, four generations per model

| Metric | iLab V2 alignment | Amarel V2 retrain | Published Dreamer | Fine-tuned Dreamer |
|---|---:|---:|---:|---:|
| Onset F1 25 ms | 15.89% | 17.21% | 80.81% | 82.48% |
| Typed onset F1 25 ms | 5.97% | 7.32% | 48.22% | 48.51% |
| Short-circle-pair recovery | 9.72% | 11.81% | 27.08% | 26.39% |
| Median matched-slider duration error | 107.98 ms | 107.98 ms | 97.91 ms | 68.91 ms |
| Mean requested-star error | 2.578 | 2.442 | 0.755 | 0.698 |

The retrain slightly improves wider-tolerance onset/type matching and slider durations on the retained songs, but does not close the large gap to pretrained Dreamer. Tight timing and difficulty control regress on the three-song group. One narrow advantage of the older iLab V2 is its zero invalid exports in this 12-map audit; that is structural validity, not musicality.

The older comparison script used by 63967705 selected by star error without filtering invalid exports. Its selected fine-tuned Star!! artifact is invalid and must not be delivered as a validated playtest. Aggregate metrics already retain invalid outputs in their denominators. The updated script used by 64002479 rejects invalid maps for playable selection; both raw reports are preserved unmodified.

Raw evidence: `docs/evidence/v2-amarel-20261118/comparison.json` and `docs/evidence/ilab-v2-comparison-20261117/comparison.json`.

Current objective is true random-initialization Dreamer training, not another pretrained adaptation. See `dreamer-from-scratch-2026-11-17.md`. Published/fine-tuned Dreamer remain evaluation baselines only.

# Training identity and metric clarification

The Amarel run trained one model: a 2,000-update adaptation of published osu-dreamer v8. The published checkpoint was used unchanged for baseline inference. The old audio/native flow, autoregressive V2, and earlier AE/VAE experiments were not retrained on Amarel in this run. Calling the fine-tune simply “our version” obscured this distinction.

No architecture changed. The chart codec, audio encoder, and style sampler remained frozen. Only generator weights were optimized on 2,757 real ranked charts from 633 training sources, using the existing distance/direction objective, AdamW at 3e-5, batch 32, 152-latent crops, BF16 and EMA 0.99. Kaguya was excluded. No VAE, KL regularizer, reward training, new layers, or parameter-count increase was introduced. The measured improvement is from weight adaptation; this experiment does not isolate which training examples caused it.

The earlier AE/VAE pilot was a different experiment. Its broader validation failed promotion criteria. A small gain from the latest fine-tune on one development song does not overturn that conclusion.

## F1 is not binary accuracy

For each seed, typed F1 is twice the number of onset-matched events with the correct circle/slider/spinner type, divided by the number of generated plus reference events. The reported typed F1 is averaged over four seeds. Missed events, extra events and mistyped matches all lower this score. There is no complementary class whose predictions yield 100% minus this F1.

Across all four Amarel fine-tune seeds, 1,043 events match within 25 ms and 615 of those have the correct type: conditional type accuracy is 58.96%. Published v8 has 610 correct types among 1,021 matches: 59.75%. Thus improved combined typed F1 (48.22% to 48.51%) does not establish a better type classifier: the conditional type score actually declined slightly. The pooled typed F1 differs slightly from the reported mean per-seed F1 because denominators differ between seeds.

For a strictly binary task evaluated by ordinary accuracy, swapping every predicted label changes accuracy A to 1-A on the same examples. A model can score below 50%; flipping can help if that reversal generalizes. That fact does not apply to this event-generation F1, multiclass object types, or continuous timestamps/durations.

## Re-scoring retained selected maps

All three maps below use the identical original Kaguya audio, SHA-256 `6ee5686d629635b982d5ccd97e4bcac5a7fa01969ab13c37587ef180b90bbe17`, and the same reference. These are single previously selected candidates, not four-seed aggregates. No new inference or reference-based tuning was done. Evidence: `docs/evidence/amarel-20261114/selected-historical-comparison.json`.

| Selected map | Onset F1 25 ms | Typed F1 25 ms | Median matched-slider duration error |
|---|---:|---:|---:|
| Earlier audio/native flow, phrase model | 66.91% | 39.49% | 59.22 ms |
| Published Dreamer | 83.50% | 54.23% | 7.70 ms |
| Amarel Dreamer fine-tune | 84.80% | 57.02% | 6.70 ms |

Duration metrics cover only matched slider pairs, so the evaluated subset differs across models. Across four seeds the new fine-tune's median duration error is 68.91 ms, not 6.70 ms: the selected seed is much better on this measure. The original pre-November audio/native-flow artifact used different audio and was excluded rather than comparing its raw timestamps.

Historical 16-seed onset F1 at 50 ms was 70.60% for original flow, 71.47% for the more-data control, and 70.73% for the phrase model, versus 85.01% for the latest Dreamer fine-tune's four seeds. This supports stronger coverage in the Dreamer outputs but is not a fully matched seed protocol. Earlier flow's tight 10 ms scores were sometimes slightly higher; do not claim improvement on every metric.

V2's historical 71.7% token accuracy and 65.5% exact-offset accuracy were teacher-forced, with reference prefixes/history and timing supplied. They are not comparable to independent-generation event F1. No matched V2 timing/type/duration evaluation is established here, so a numerical ranking against V2 would be unsupported.

## Branch renaming

At the user's request, GitHub branches `codex/audio-native-flow`, `codex/legacy-distance-aux`, `codex/legacy-distance-ilab`, and `codex/v2-coverage-rhythm-flow` were renamed by removing the prefix. Each tip SHA was verified unchanged. There were no open pull requests. Local tracking branches and the active Amarel checkout were updated. Historical documentation and commit history were preserved.

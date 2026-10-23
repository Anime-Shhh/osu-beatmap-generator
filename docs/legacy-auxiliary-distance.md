# Auxiliary head-to-head distance supervision for the original autoregressive model

This experiment changes the AST + Transformer token model in `src/model.py`.
It does not replace absolute positions, change the vocabulary, add direction
tokens, or modify the V2 geometry diffuser or `src/rhythm_flow/`. The distance
head provides training supervision only. Generation does not consult it or use
it for constraints, reranking, or sampling.

## Targets and alignment

`object_head_distance_targets()` in `src/tokenizer.py` adds labels alongside the
existing token sequence. For circle and slider **heads**, it reconstructs the
original reference `(x,y)` from the grid cell plus the stored, unclamped residuals,
then computes `hypot(x - previous_x, y - previous_y) / 640`.

The grid remains 32 × 32, with 16 × 12 pixel cells and vocabulary size 1,166.
X/Y residual signs still mean displacement from the cell center. Tokenization
and absolute-coordinate reconstruction are unchanged.

The collator computes these labels before its existing residual clamps. Labels
are zero-filled and masked at the first head in a sequence/window, padding,
non-position tokens, slider controls/endpoints and spinner positions. A spinner
resets the preceding-head context; the first head after it is also masked. Slider
controls do not replace the preceding head. No unavailable preceding object from
outside a cached window is invented. Non-finite reference heads reset context.

For a position token at full-sequence index `j`:

- Decoder input is `tokens[:, :-1]` and next-token output is `tokens[:, 1:]`.
- The hidden state at index `j-1` predicts that position token.
- Distance targets and masks use the same `[:, 1:]` shift.
- The causal decoder sees tokens only through `j-1`, plus the existing audio and
  difficulty/BPM conditioning. It receives neither the target position token nor
  a ground-truth distance. As before, reference continuous residuals are not fed
  into the decoder as inputs.

This is consistently **head-to-head** supervision. It does not describe movement
from a slider tail to the next head, repeat-slider exit positions, or actual
player cursor travel. A new continuous-distance-plus-direction tokenizer would
need separate work on directions, accumulated error, boundaries and retraining.

## Head, objective and configuration

The shared normalized decoder state feeds a small
`Linear(d_model,64) → GELU → Linear(64,1) → Softplus` head. Its scalar output is
nonnegative and normalized by 640 pixels. The existing token/time/X/Y heads and
the decoder's default four-value inference return remain intact. Training forward
adds `distance_predictions` to the existing output dictionary.

Initialization of the new head uses an isolated RNG context so it does not
perturb existing parameter initialization or the baseline random stream.

```text
existing_loss = w_discrete * token_CE + w_residual * existing_residual_loss
loss = existing_loss + distance_loss_weight * masked_SmoothL1(predicted_distance, target_distance)
```

Smooth L1 uses beta 1 in normalized units, summed over valid labels and divided
by their count. An empty mask gives a differentiable zero with zero gradients.
Masked values do not enter the loss. The original residual-loss calculation is
preserved. At weight zero the auxiliary term is omitted from backpropagation.

| Option | Default | Meaning |
| --- | ---: | --- |
| `--distance_loss_weight` | `0.0` | Original-objective baseline; try `0.1` as a hypothesis, not a validated setting |
| `--initialize_from` | none | Weights-only initialization; explicitly permits a wholly missing distance head and starts fresh optimizer/scheduler/scaler state |
| `--resume` | none | Resume a new-format checkpoint with matching distance config and data fingerprint, into a fresh output directory |
| `--max_updates` | `0` | Legacy optimizer-update limit; zero uses the existing epoch budget |
| `--seed` | `42` | Existing training seed; cached legacy shuffle now also uses this seed plus epoch |
| inference/evaluation `--initialize_distance_head` | false | Explicitly allow a historical checkpoint missing only the auxiliary head |

Weight 0.1 is not necessarily strong enough: normalized Smooth L1 can be small
relative to token and residual losses. Tune it on validation sources with a
predeclared budget; do not choose it based only on distance loss or on test maps.
The original trainer remains teacher-forced; its pre-existing `tf_ratio` logging
does not implement scheduled sampling.

Training logs `train/distance_loss`, `train/distance_mae_px`, valid target counts,
and optimizer updates. Validation and standalone evaluation additionally report
valid-token CE/accuracy and time/X/Y residual MAE. Time residual MAE is in **beats**;
coordinate residual MAE is in **pixels**. Distance loss/MAE aggregate by valid
head-pair counts, not padded sequence length. Validation's existing `val/loss`
continues to be token CE, so checkpoint selection is not changed to favor the new
auxiliary metric. With no valid distance labels, validation reports an unavailable
MAE and a zero target count rather than asserting perfect accuracy.

## Checkpoint and data preservation

`load_legacy_state_dict()` is strict. A historical weights-only migration must
explicitly permit exactly the complete new `decoder.distance_head.*` key set to
be missing. Partial head corruption, unrelated missing keys, unexpected keys and
incompatible shapes still fail. Existing compiled-key normalization and AST
positional-embedding migration remain available. AST migration runs before the
optimizer is constructed, so replacement parameters are included in it.

New legacy checkpoints save `legacy_model_config`, versioned `distance_config`,
training settings, optimizer update count, initial-checkpoint SHA-256, split-file
SHA-256 and an exact cached-shard content fingerprint. Existing split manifests
are read without being overwritten. Every legacy run, including resume, requires
a fresh/empty `model_dir`. Historical checkpoint and dataset files are not rewritten.
Shards are hashed once at startup; this can take time on a large corpus.

For quota-constrained iLab launches, `--checkpoint_top_k 1 --compact_checkpoints`
atomically saves each state once and hardlinks the `best.pt`/`last.pt` aliases.
It preserves both states when their epochs differ. Allow room for three states
during an atomic replacement. This is legacy-only and opt-in; historical files
are never removed. Progress lines report batches, completed updates, examples,
distance MAE and elapsed training time. `configs/legacy_distance_train.sbatch`
uses a fresh source/run/model directory, one GPU and a 48-hour wall-clock limit.
The legacy trainer does not implement DDP; do not launch multiple unsynchronized
processes against one model directory.

`last.pt` records the final update budget independently of the existing top-k and
`best.pt` retention. Use `last.pt` for the matched-budget comparison; best-validation
checkpoints can represent different update counts. The update cap is checked at
optimizer steps, after accumulation; AMP-overflow skips do not consume the budget. The existing treatment of incomplete final
accumulation groups is retained; inspect the recorded actual update count.

For an old checkpoint at inference, add `--initialize_distance_head`; the head is
unused for generation. New checkpoints reconstruct the architecture from their
saved configuration. Old checkpoints still need the original architecture CLI
settings when those differ from defaults. Never use broad `strict=False` to hide
a checkpoint mismatch.

## Controlled comparison — commands to run only when training is authorized

Use the **same immutable, source-separated dataset and split manifest**, initial
checkpoint, training seed, epochs/update cap, batch/accumulation settings, optimizer
settings and decoding settings. Audit source grouping before running: matching a
legacy shard split does not prove different windows of one song were kept together.
For reproducible loader ordering, use `--num_workers 0` in this initial experiment.

The commands below are a recipe, not jobs submitted by this change:

```sh
# Set these to existing local/iLab paths. RUN_ROOT must be a new experiment directory.
DATA_DIR=/path/to/immutable/shards
SPLIT_FILE=/path/to/frozen/splits.json
INITIAL=/path/to/historical/legacy-checkpoint.pt
RUN_ROOT=/path/to/new/distance-experiment

python -m src.train --stage legacy \
  --data_dir "$DATA_DIR" --split_file "$SPLIT_FILE" --initialize_from "$INITIAL" \
  --model_dir "$RUN_ROOT/baseline" --log_dir "$RUN_ROOT/logs-baseline" \
  --distance_loss_weight 0 --seed 1731 --epochs 50 --max_updates 10000 \
  --batch_size 8 --accum_steps 1 --num_workers 0 --early_stop_patience 0

python -m src.train --stage legacy \
  --data_dir "$DATA_DIR" --split_file "$SPLIT_FILE" --initialize_from "$INITIAL" \
  --model_dir "$RUN_ROOT/auxiliary" --log_dir "$RUN_ROOT/logs-auxiliary" \
  --distance_loss_weight 0.1 --seed 1731 --epochs 50 --max_updates 10000 \
  --batch_size 8 --accum_steps 1 --num_workers 0 --early_stop_patience 0
```

50 epochs / 10,000 updates are example budgets, not established requirements.
Use enough epochs to reach the chosen cap, and repeat complete matched pairs with
several training seeds. Both commands read the same initial checkpoint and start
fresh optimizer state. The original-objective arm contains an unused head but
preserves the original token/residual objective and generation behavior.

Copy `configs/legacy_distance_songs.example.json` to a new evaluation manifest and
replace its paths with **held-out full songs**. Keep the same BPM, difficulty list
and audio files for both arms. The evaluator requires an explicit test-set
declaration; it cannot independently infer song membership from arbitrary audio.

```sh
python -m scripts.compare_legacy_distance \
  --baseline "$RUN_ROOT/baseline/last.pt" \
  --auxiliary "$RUN_ROOT/auxiliary/last.pt" \
  --data-dir "$DATA_DIR" --split-file "$SPLIT_FILE" \
  --songs /path/to/heldout-songs.json --output-dir "$RUN_ROOT/comparison" \
  --seeds 1731 1732 1733 --temperature 0.9 --max-tokens 512 \
  --batch-size 4 --device cuda
```

The comparison rejects mismatched recorded updates, initial weights, dataset bytes,
splits, architectures or training settings other than the treatment and output/log
paths. It evaluates the same test loader, then exports full-audio `.osu`/`.osz`
examples with paired generation seeds. It saves:

- Existing teacher-forced CE, edit distance, timing MAE, hit F1 and onset sensitivity;
  valid-token accuracy, time/X/Y residual MAE and distance MAE in pixels. A zero-weight
  head is untrained: the original-objective arm's auxiliary MAE/loss are **N/A**.
- Raw head-pair distances/time gaps and distributions jointly grouped by requested
  difficulty and gap bins `(0,50], (50,100], (100,200], (200,400], (400,800], >800 ms`.
- Head-speed extremes, nonpositive gaps, raw sampled head-boundary violations,
  exported head and slider-body violations, invalid sliders, event density,
  first/last head times and realized stars using pinned rosu-pp-py 4.0.2.
- Paired auxiliary-minus-baseline changes and a `playtests.md` sheet explicitly
  marked **not performed** until a human records observations.

The full-song runner retains the original 6-second context / 3-second prediction
window stitching, absolute-coordinate clamps, and slider decoding defaults. Known
limitations such as its initial center-window gap and fixed decoded slider length
are shared by both arms; this auxiliary experiment does not repair them. Raw
window diagnostics include sampled objects later discarded by interval filtering;
exported-map metrics describe only retained objects. Numerical slider-path auditing
reuses the existing read-only native-path utility, without changing V2 or invoking
its geometry model. Head speed is not actual cursor speed or the official aim formula.

For one checkpoint, `python -m src.eval --stage legacy ... --output_json result.json`
adds machine-readable teacher-forced metrics. Standalone inference accepts
`--seed` for repeatable sampling. Output report files/directories must be new.

Judge generated timing, spacing distributions, movement extremes, slider behavior,
density, star calibration and blind playtests together. Do not equate lower
auxiliary loss with better maps, and do not use coordinate error against one human
layout as the sole quality criterion.

## Verification and current result status

Focused tests cover exact distances and residual offsets, first/window/spinner/
control/padding masks, shifted teacher forcing, causal information access, empty
masks and gradient behavior, loss weighting, strict old/new checkpoint loading,
generation independence, deterministic cached ordering, immutable split handling,
configuration persistence and paired report/archive creation.

Final local regression result: **149 tests and 28 subtests passed** in 27.33 seconds.
CLI help and whitespace checks also passed. The existing project environment lacked
legacy audio/model dependencies, so testing used an isolated environment at
`/private/tmp/osu-distance-tests`; the project environment was left unchanged.

Tests use small synthetic fixtures and a stub audio encoder where needed, avoiding
pretrained downloads. A small trainer-entrypoint test verifies the saved checkpoint
contract; it is not a research training run. No real-model comparison, full-song
quality result or playtest observation is claimed. No training jobs were submitted
and no changes were published.

```sh
python -m pytest -q tests/test_legacy_distance.py tests/test_dataset_collate.py tests/test_representation.py
```

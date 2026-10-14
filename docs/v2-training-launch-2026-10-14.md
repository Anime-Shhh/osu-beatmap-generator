# V2 authentic-data training launch — October 14, 2026

The new architecture is deployed at `/common/users/asj102/osu_project/codex_v2_20261014`.
Historical deployments and job `450738` were preserved.

## GPU selection

Account `asj102` can use `unlimited`, but test-only submissions to the H200 guest
partition and idle RTX 6000 Blackwell partition were rejected for account access.
At inspection, usable free allocations included one RTX PRO 5000 Blackwell,
four RTX 4500 Ada cards, and five RTX A4500 cards on ilab2.

The checkpoint-free benchmark uses the full default V2 models, genuine bar
batches from the authentic ingestion smoke corpus, BF16, warmup, and CUDA
synchronization. These are compute measurements, not convergence measurements
or an estimate for arbitrary songs. The final comparison used batch four per
GPU, 32 measured batches, and gradient accumulation four.

| Allocation | Planner train examples/s | Planner validation examples/s | Geometry train examples/s | Geometry validation examples/s |
| --- | ---: | ---: | ---: | ---: |
| 1 RTX PRO 5000 Blackwell | 32.42 | 84.88 | 51.93 | 75.95 |
| 3 RTX 4500 Ada | 25.89 | 123.45 | 62.08 | 112.04 |
| 4 RTX A4500 | 28.57 | 157.78 | 77.29 | 152.46 |

At batch two, two Ada cards matched the single Blackwell for planner training
(20.05 versus 20.41 examples/s), and exceeded it for geometry training
(41.05 versus 32.69). At batch four without accumulation, two Ada cards reached
26.87 planner / 47.09 geometry examples/s, versus 30.11 / 50.18 for Blackwell.
Two average cards therefore do not consistently beat one faster card.

Four A4500s were selected because exhaustive validation dominates much of an
epoch and this allocation has the highest validation and geometry throughput.
For the final measured planner rates, four A4500s beat Blackwell in combined
train-plus-validation compute once validation examples exceed approximately
0.76 times training examples. Confirm this against the final corpus and epoch.
Four of five free cards are requested, leaving one available.

## Reliability changes

- The real training template launches one `torchrun` DDP process per visible,
  Slurm-allocated GPU for both stages and requests a 48-hour time limit.
- Validation uses a distributed sampler instead of every rank repeating the
  full split. It may pad at most world-size minus one examples to keep rank
  lengths equal.
- An entirely empty geometry batch still executes diffuser forward/backward
  with a masked dummy event. Previously it bypassed DDP, potentially hanging
  other ranks. GPU probes passed with empty rank zero and populated peers at
  world sizes two, three, and four.
- Training emits measured train and total epoch seconds.
- Materialization stages beside its output and releases each staging file
  after finalization. iLab `/tmp` has a 2 GB user quota; the first 1,000-source
  preprocessing attempt failed at 237 accepted charts because staging hit it.
- Explicit CPU requests were removed from the Slurm templates because the
  cluster submission policy rejects them.

The local V2 suite passed 94 tests and 28 subtests after the training changes.
The targeted quality/data tests passed again after quota-safe staging changed.

## Submitted jobs

Preprocessing job `498148` builds up to 300 raw tracks from immutable source
revision `b82117a5de13b7df5290482aed7cffc49cb76e6d`. Actual eligible audio component
and chart counts are reported by the completed corpus audit. This size was
chosen to fit the approximately 20 GB project quota headroom, including new
records and checkpoints; the earlier 1,000-track setting was not a requirement.

Training job `498149` depends on successful completion of `498148` and requests
four A4500s on ilab2, 64 GB RAM, and `2-00:00:00`. It uses batch four per GPU,
accumulation one (global batch 16), BF16, and one loader worker per rank. It
trains the planner for up to 200 epochs, then geometry for up to 100, with
early-stopping patience 15. No diagnostic-data override is enabled.

Paths:

- Corpus: `/common/users/asj102/osu_project/v2_runs/run_20261014_real/data_300`
- Training log: `/common/users/asj102/osu_project/v2_runs/run_20261014_real/logs/train-498149.log`
- Planner checkpoints: `/common/users/asj102/osu_project/v2_runs/run_20261014_real/planner`
- Geometry checkpoints: `/common/users/asj102/osu_project/v2_runs/run_20261014_real/geometry`

## Verified completion

The SSH master disconnected during preprocessing. After the user returned and
the connection was available again, training had already finished. No continued
training monitor or recurring automation was created.

Slurm reports preprocessing `498148` and training `498149` as **COMPLETED**, both
with exit code `0:0`. Training allocated four GPUs, started at **01:58:33 EDT on
October 14**, and finished at **11:03:56 EDT**, after **9 hours, 5 minutes,
23 seconds**. Its submitted limit was 48 hours.

The corpus contains **773 authentic charts across 188 audio components**, split
into 150 training, 20 validation, and 18 test components. It has 66,903 training,
8,555 validation, and 6,818 test bar targets. There were 88 rejected charts.
Both stage startup audits report `production_ready: true` and `world_size: 4`.
The training sampler emits 2,400 examples per epoch. Thus exhaustive validation
has 3.56 times as many examples as training, supporting the allocation choice.

| Stage | Epochs completed | First epoch | Mean epoch | Best checkpoint epoch (zero-based) | Best validation loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Planner | 55 | 247.03 s | 239.24 s | 39 | 0.862975 |
| Geometry | 87 | 248.01 s | 221.07 s | 71 | 0.375872 |

Both stages stopped through the configured validation patience, not Slurm's
time limit. Each saved best checkpoint was opened successfully on CPU; the
geometry checkpoint records stage `geometry`, global step 43,200, and the
verified authentic-corpus audit. The planner best checkpoint has global step
24,000. Best checkpoint epochs precede the last completed epochs because
validation stopped improving.

The first epochs would imply approximately 20.6 hours for all 200 planner and
100 geometry epochs without early stopping; the completed run took 9.1 hours.
There is no remaining ETA: training is complete. The training corpus audit is
an integrity/size guard; held-out generated-map quality was not evaluated by
this launch task.

Use the geometry-stage best checkpoint for generation:

```text
/common/users/asj102/osu_project/v2_runs/run_20261014_real/geometry/best.pt
```

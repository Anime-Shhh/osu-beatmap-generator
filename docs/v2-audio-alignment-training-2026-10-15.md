# Audio-alignment training launch, October 15, 2026

Architecture verified at `5ad39f026fb9c77146fd0b4a2b599e66a0d2e381` on
`architecture-v2-hierarchical-beatgrid`, matching origin at launch. The new preset
enables 128 timestamped local audio tokens per bar and a four-channel auxiliary
alignment head (circle, slider, spinner onset, and slider-active interval).
Alignment loss weight is 0.2. Existing schema-3 real-audio records are compatible.
Sixteen focused alignment/checkpoint tests passed locally with PyTorch/pytest,
including a spawned loader worker and a sub-millisecond late-song bar.
The complete V2 suite passed: **109 tests and 28 subtests**. Full repository
collection additionally encounters two legacy V1 modules requiring `torchaudio`,
which is not installed in the Mac environment; the V2 suite is independent.

## Allocation decision

The accessible unlimited partition had six free RTX 4500 Ada cards on ilab3 and
five free A4000 cards on rlab4. A4500 GPUs on ilab1 had insufficient unallocated
host RAM; the A100, Blackwell 5000, and A6000 nodes had no free GPU allocation.
The guest H200 test submission was rejected for account/partition access.
No historical jobs, checkpoints, or runs were changed.

Short Slurm benchmark jobs 513833 and 513834 both completed successfully. Each
used the new 128-token architecture, BF16, batch 4 per GPU, and 12 timed steps
after warmup. Batches were selected across 32 training groups, filtering empty
event bars, and preloaded before timing. Measurements exclude input loading,
corpus audit, and checkpoint writes. Global batch changes with GPU count; these
results compare throughput rather than equal convergence per optimizer step.

| Allocation | Planner train examples/s | Planner validation examples/s | Geometry train examples/s | Geometry validation examples/s |
| --- | ---: | ---: | ---: | ---: |
| 1 Ada | 11.07 | 35.02 | 25.60 | 32.19 |
| 2 Ada | 20.58 | 70.00 | 50.84 | 64.99 |
| 4 A4000 | 23.11 | 96.62 | 58.66 | 91.83 |
| 5 Ada | 33.43 | 171.77 | 88.75 | 159.31 |

Five Ada GPUs were fastest for both stages. Multi-GPU geometry backward also
passed with one rank holding an empty event batch. Benchmark rank-0 peak memory
was approximately 8.53 GB on five Adas. See the companion benchmark JSON.

## Production run

- Job: **513971**, `osu-v2-audio-align-5ada`, unlimited partition, ilab3.
- Submitted: **2026-10-15 00:39:25 EDT**.
- Allocation: **5 RTX 4500 Ada GPUs**, 64 GiB host RAM; six were available at
  submission, leaving one available GPU unrequested.
- Limit: **48 hours**.
- Batch: 2 per GPU, accumulation 2, effective global batch 20, BF16, one spawned
  loader worker/rank. Lower per-GPU batch provides more memory headroom for
  dense real chart bars than the batch-4 throughput benchmark.
- Full corpus: reused `run_20261014_real/data_300`; no diagnostic data override
  or bounded-step training. Balanced training draws 16 bars/source group/epoch.
- Planner: initialize compatible legacy planner weights from
  `run_20261014_real/planner/best.pt`; fresh optimizer, learning rate 0.00005,
  up to 60 epochs. New audio modules initialize fresh.
- Geometry: automatically follows using this run's best planner, fresh geometry
  optimizer, learning rate 0.0002, up to 100 epochs.
- Early stopping: patience 10, minimum validation improvement 0.001 per stage.
- Checkpoint retention: best/latest only (`--checkpoint-interval 0`) because
  approximately 8 GiB remained in the 100 GiB users quota. Historical runs remain.
- Source: `/common/users/asj102/osu_project/codex_v2_audio_alignment_20261015_precision`
- Run: `/common/users/asj102/osu_project/v2_runs/run_20261015_audio_alignment_precision`
- Log: `/common/users/asj102/osu_project/v2_runs/run_20261015_audio_alignment_precision/logs/train-513971.log`

The deployed source additionally supports configurable checkpoint retention,
prints local-audio configuration in startup logs, and lets the GPU benchmark
load the architecture preset. These operational changes are recorded with this
launch report. Existing numbered-checkpoint behavior remains the default.
The fixes are committed as `45e8597`. SHA-256 checks confirmed that deployed
`alignment.py`, `train.py`, `training.py`, and the Slurm template exactly match
the tested local files in that commit.

## Initial-attempt stall and loader correction

Attempt 513864 started at 00:12 EDT but stalled in its first epoch. Its loader
workers stopped consuming CPU, one rank waited on a futex, and other ranks
remained busy in synchronization. The preloaded benchmark did not exercise
DataLoader multiprocessing. The Linux default fork start method was used after
CUDA/NCCL initialization, a configuration PyTorch explicitly warns can deadlock:
https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html

The loader now explicitly uses `multiprocessing_context="spawn"` whenever
workers are requested. Progress logs every 50 batches make subsequent stalls
observable. The stalled attempt was canceled, retaining its logs in the original
run directory; replacement 513939 used entirely new source/run directories.
The unsafe fork configuration is a likely cause, not a proven stack-trace
diagnosis: kernel ptrace restrictions prevented attaching a debugger.

A CPU-only profile over eight training source groups took 16.21 seconds under
profiling; 14.51 seconds were in `reference_events`, primarily grammar decoding.
The GPU microbenchmark excluded this preprocessing. This is why measured full
epoch time, rather than GPU examples/second alone, is used for the run ETA.

## Validation precision failure and correction

Attempt 513939 completed its training pass but failed during validation at
00:36:48 EDT: a local query grid was not strictly increasing. Secondary
multiprocessing semaphore cleanup warnings also appeared. No checkpoint was
written. Logs remain in the `_spawn` run directory.

The validation data contains meter-transition bars as short as 0.000483333
beats: approximately 0.154 ms at 77.7 seconds into one song. Float32 cannot
represent 128 distinct query timestamps across that span. Query clocks now
retain float64 precision; interpolation already casts its weights to feature
dtype, preserving the neural feature/weight representation and checkpoint
compatibility. No existing records were modified.

An audit checked every validation bar under every stored clock hypothesis:
**17,110 grids checked; eight collapse in float32; zero fail in float64**.
A regression test verifies ordering and a finite planner forward for a similarly
short late-song bar. Final attempt 513971 includes both corrections.
Final monitoring uses login-node reads with `stat --cached=never` to refresh
NFS metadata, without opening extra sessions on the compute node.

## First-epoch verification and ETA

The corrected run completed epoch 0 at approximately **00:52:40 EDT**, and was
still RUNNING at **00:53:10 EDT with 13:45 elapsed**. Both `planner/best.pt` and
`planner/latest.pt` exist, approximately 1.2 GB each. No further monitoring is
scheduled. Raw startup/epoch metrics are in the companion first-epoch JSON.

- Training pass: **496.15 seconds**, 240 batches/rank, 120 optimizer updates,
  2,400 balanced training examples across five ranks.
- Full epoch including validation/checkpoint writes: **688.12 seconds**
  (**11 minutes 28 seconds**).
- Train loss **0.59935** = token CE **0.51842** + 0.2 × alignment BCE **0.40461**.
- Validation loss **0.89751** = token CE **0.82637** + 0.2 × alignment BCE **0.35568**.
- Distributed metric `steps` sums rank-local batches (1,200 train / 4,280 val);
  these are not optimizer-update counts.

**ETA: approximately 30 hours total from the 00:39 EDT start, around 07:00 EDT
on Friday, October 16. Allow 26–34 hours: approximately 03:00–11:00 EDT October
16.** This projects up to 60 planner epochs and 100 geometry epochs. Geometry
has only been microbenchmarked, not timed for a full epoch; its GPU compute is
faster but it shares the substantial alignment-label preparation overhead.
Variation in sampled bar lengths, CPU/filesystem contention, and checkpoint
writes are included as uncertainty. Early stopping (patience 10) may finish
either stage substantially sooner. The hard 48-hour deadline is approximately
00:39 EDT October 17.

Successful teacher-forced training and finite losses do not establish improved
free-running musical alignment or playability; those require post-training
held-out generation and audio-intervention evaluation.

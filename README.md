# OsuM — AI Beatmap Generation from Audio

OsuMapper is an end-to-end deep learning system that automatically generates [osu!](https://osu.ppy.sh/) beatmaps from raw audio. Given an MP3 file of any song, the model produces a fully playable `.osu` beatmap file containing timed hit circles, sliders, and spinners — the core gameplay elements of osu!standard mode.

The model uses an **encoder–decoder Transformer architecture** with a pretrained audio encoder (Audio Spectrogram Transformer) and an autoregressive decoder that generates a structured token sequence representing beatmap objects. A novel **hybrid discrete + continuous output** scheme enables both coarse grid-based placement and sub-pixel precision via learned residual offsets.

### Latest training validation snapshot

Representative end-of-run metrics (100 epochs, validation on held-out shards). Timing MAE and hit F1 are computed on greedy next-token predictions with per-sample `ms_per_beat` from BPM.

| Quantity | Value |
|----------|-------|
| Train loss | 1.717 |
| Val loss (CE) | 2.107 |
| Edit distance | 0.436 |
| Timing MAE | 396.3 ms |
| Hit F1 | 0.108 |
| Learning rate | 5.0×10⁻⁵ |
| Onset zero-out logit diff | 0.0049 |

Compared to the previous model (timing MAE ~900 ms, hit F1 ~0.04), the current model achieves **2× better timing** and **3.5× better hit accuracy** thanks to onset gating, per-channel normalization, and metric fixes described below.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Tokenization Scheme](#tokenization-scheme)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Setup and Usage](#setup-and-usage)
- [Current Status](#current-status)
- [Training vs evaluation vs inference](#training-vs-evaluation-vs-inference)
- [Recent Changes](#recent-changes)

## Problem Statement

An osu! beatmap defines *when* and *where* a player must click, drag, or spin during a song. Human mappers spend hours manually placing hit objects to match the song's rhythm, melody, and intensity. This project frames beatmap generation as a **sequence-to-sequence problem**:

- **Input:** **129-channel** audio features (128-bin log-mel + onset strength, separately z-normalized) for a ~6-second window **plus** user-provided conditioning: difficulty (star-rating bin) and BPM
- **Output:** A variable-length sequence of tokens encoding hit objects with precise timing and spatial placement

Difficulty and BPM are **explicit conditioning inputs**, not prediction targets — they globally control map density, spacing, and structure, and are provided by the user at inference time.

The key challenges are:

1. **Temporal alignment** — hit objects must land on musically meaningful beats
2. **Spatial coherence** — object placement should form readable patterns
3. **Conditioning** — difficulty (5 bins: Easy–Expert) and BPM are given as inputs; the model learns to generate maps that match the requested difficulty
4. **Multi-object types** — circles, sliders (with parametric Bezier curves), and spinners each have different encodings

## Dataset

Training data comes from the [project-riz/osu-beatmaps](https://huggingface.co/datasets/project-riz/osu-beatmaps) dataset on HuggingFace, which contains community-created beatmaps paired with their audio files.

### Preprocessing Pipeline (`src/preprocess.py`)

Raw data is streamed from HuggingFace and preprocessed into WebDataset `.tar` shards for efficient training:

1. **Audio loading:** MP3 bytes are decoded via torchaudio, resampled to 22,050 Hz, and converted to mono
2. **Mel + onset features:** 128-bin mel spectrograms (2048-point FFT, ~20 ms hop) plus one **onset-strength** channel per frame (`librosa.onset.onset_strength`, max-normalized per window and aligned to mel length) → **`[1, 129, T]`** tensors before batching
3. **BPM extraction:** BPM is parsed from the `.osu` file's `[TimingPoints]` section (60000 / ms_per_beat for the first uninherited timing point)
4. **Difficulty binning:** Star rating is mapped to a 5-bin scheme (Easy, Normal, Hard, Insane, Expert) for conditioning
5. **Sliding window chunking:** Each song is split into overlapping 6-second windows with a 3-second stride, predicting the middle 3 seconds of each window
6. **Tokenization:** Inter-onset **time deltas are expressed in beats** (`delta_ms / ms_per_beat`) and quantized into 128 beat bins (1/16-beat steps); residuals are in **beats**, not milliseconds (see [Tokenization](#tokenization-scheme))
7. **Shard writing:** Processed (mel, tokens, residuals, difficulty_id, bpm) tuples are written as WebDataset `.tar` files

**Scale:** 500 shards × 200 examples/shard = **100,000 training examples** (~15 GB total)

### Data Splits

Shards are deterministically split 80/10/10 into train/val/test sets (400/50/50 shards). Split manifests are written to JSON for reproducibility.

## Tokenization Scheme

The tokenizer (`src/tokenizer.py`) converts continuous beatmap data into a discrete vocabulary that a Transformer can model, while preserving sub-bin precision through continuous residual outputs.

### Vocabulary (1,166 tokens)

| Range | Count | Purpose |
|-------|-------|---------|
| 0–3 | 4 | Special tokens: `PAD`, `BOS`, `EOS`, `SEP` |
| 4–9 | 6 | Object types: `CIRCLE`, `SLIDER_START`, `SLIDER_CONTROL`, `SLIDER_END`, `SPINNER_START`, `SPINNER_END` |
| 10–13 | 4 | Curve types: `BEZIER`, `LINEAR`, `PERFECT`, `CATMULL` |
| 14–1037 | 1,024 | Position bins (32×32 grid over the 512×384 playfield) |
| 1038–1165 | 128 | **Beat delta** bins (0–~8 beats in 1/16-beat steps); same rhythm → same token at different BPMs |

Difficulty is **not** a token — it is a separate conditioning input (integer 0–4 for Easy through Expert).

### Two-Stage Precision

Each position and time token represents a coarse bin. To recover precise coordinates, the model also predicts **continuous residual offsets**:

- **Position residuals:** `(x_offset, y_offset)` in pixels, range ±8 px horizontally, ±6 px vertically (half the bin width/height)
- **Time residuals:** stored in the same head as `time_offset_ms` but semantically **beat offsets** within ±1/32 beat (half of one 1/16-beat bin); training targets are clamped accordingly

At inference, **detokenization** converts beat deltas back to milliseconds using **`ms_per_beat = 60000 / BPM`**: `delta_ms = (bin + residual) * ms_per_beat`. Final playfield coordinates are `bin_center + predicted_residual` for x/y.

### Token Sequence Format

Each hit object is encoded as a subsequence within the full sequence. Difficulty and BPM are provided as conditioning inputs, not tokens:

```
[BOS] [time₁] [CIRCLE] [pos₁] [time₂] [SLIDER_START] [pos₂] [CURVE_TYPE] [SLIDER_CONTROL] [cp_pos] ... [SLIDER_END] [end_pos] ... [EOS]
```

- **Circles:** `<beat_bin> <CIRCLE> <pos_bin>` (3 tokens)
- **Sliders:** `<beat_bin> <SLIDER_START> <pos_bin> <curve_type> [<SLIDER_CONTROL> <cp_pos>]* <SLIDER_END> <end_pos>` (variable length)
- **Spinners:** `<beat_bin> <SPINNER_START> <pos_bin> <duration_beat_bin> <SPINNER_END>` (5 tokens)

### Difficulty Binning (5 bins)

| Bin | Star Rating | Label |
|-----|-------------|-------|
| 0 | &lt; 2.0 | Easy |
| 1 | &lt; 3.0 | Normal |
| 2 | &lt; 4.0 | Hard |
| 3 | &lt; 5.3 | Insane |
| 4 | ≥ 5.3 | Expert |

## Model Architecture

The model (`src/model.py`) is an encoder–decoder Transformer with conditioning.

### Conditioning Encoder

Difficulty and BPM are encoded into a single conditioning token prepended to the audio memory:

1. **Difficulty embedding:** Learned embedding for 5 difficulty bins (Easy–Expert)
2. **BPM projection:** Log-scaled BPM (log(bpm)/log(300)) passed through an MLP for perceptual tempo modeling
3. **Combined token:** `[B, 1, D]` conditioning token is concatenated with encoder output so the decoder cross-attends to `[cond | audio]`

### Audio Encoder

The encoder converts **129-channel** input (128 mel bins + 1 onset channel) into a sequence of audio embeddings:

1. **Linear projection:** `mel_proj` maps **all 129 channels** to model dimension `D`, capturing both spectral and onset information.
2. **Backbone:** [Audio Spectrogram Transformer (AST)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) receives **only the 128 mel channels** (onset is stripped before AST to match its pretrained input format).
3. **Onset gating:** The `mel_proj` output (which includes onset information) is temporally aligned to the AST output via linear interpolation, then merged through a **learned sigmoid gate** (`mel_gate`). This ensures the onset signal reaches the decoder even though AST only processes mel. The gate learns how much onset information to inject at each temporal position.
4. **Position embedding resizing:** AST was pretrained on 10-second clips (1024 frames). Since our input is ~6 seconds (301 frames), the positional embeddings are bilinearly interpolated to match the actual frame length at runtime.
5. **Temporal upsampling:** A 1D convolutional stack with 2× upsampling increases the temporal resolution of AST's output.

### Beatmap Decoder

The decoder autoregressively generates the token sequence conditioned on `[cond | encoder_out]`:

1. **Token + positional embeddings:** Learned embeddings for the **1,166-token** vocabulary and up to 2,048 sequence positions
2. **Transformer decoder:** 6 layers, 8 attention heads, 2048-dim feedforward, pre-norm architecture with causal masking
3. **Cross-attention:** Each decoder layer attends to the conditioning token and full audio encoder output
4. **Output heads (hybrid):**
   - `discrete_head`: Linear projection to **1,166** logits for next-token prediction (trained with CrossEntropyLoss)
   - `time_res_head`: Linear → scalar for **beat** residual offset (stored in `Residuals.time_offset_ms` for API compatibility)
   - `x_res_head`: Linear → scalar for x-position residual offset
   - `y_res_head`: Linear → scalar for y-position residual offset

### Why Hybrid Discrete + Continuous?

Pure discretization into a 32×32 grid would limit placement precision to ±8 px — too coarse for a game where precision matters. Pure regression would require the model to predict exact coordinates from scratch. The hybrid approach gets the best of both worlds: the discrete head handles the *combinatorial* decision of "which region," while the regression heads handle the *fine-grained* adjustment of "exactly where within that region."

## Training vs evaluation vs inference

| Stage | Audio input | Conditioning | Time semantics |
|-------|-------------|--------------|----------------|
| **Training** | Log-mel + log(onset+ε), z-normalized per channel group on `mel` `[B,1,129,T]`; batches include `difficulty_id`, `bpm` | Same as inference | Targets use **beat** bins + beat residuals; loss is CE + SmoothL1 on residuals |
| **Validation (`train.py`)** | Same as training | Same | Metrics use **greedy argmax** on teacher-forced positions; **timing MAE / hit F1** call `detokenize_to_hitobjects(..., ms_per_beat=60000/bpm)` per sample |
| **`src.eval`** | Cached shards | Same | Same detokenize rule; can run full test split |
| **Inference (`src.inference`)** | Mel + **onset** per window, same log + z-norm as training; user `--difficulty`, `--bpm` | Required | `detokenize` uses `ms_per_beat` from BPM; sliding windows + dedup |

Evaluation metrics in **milliseconds** are always reconstructed from beat tokens using the sample's BPM, so BPM must be correct for timing numbers to be meaningful.

## Training Pipeline

Training (`src/train.py`) is designed for SLURM-based GPU clusters with several optimizations.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-2) |
| Scheduler | CosineAnnealingWarmRestarts (T₀=20, T_mult=2) |
| Precision | Mixed precision (bfloat16 via GradScaler) |
| Batch size | Auto-detected from GPU VRAM (16 for A4000/A5000, 32 for A100) |
| Gradient accumulation | 1–4 steps depending on config |
| Gradient clipping | Max norm 1.0 |
| Epochs | 100 |
| TF32 | Enabled on Ampere+ GPUs |
| torch.compile | Enabled for kernel fusion |

### Loss Function

The total loss is a weighted combination of two components:

```
L_total = 0.8 × L_discrete + 0.2 × L_residual
```

- **L_discrete** (weight 0.8): CrossEntropyLoss over 1,166 classes for next-token prediction. PAD tokens are masked via `ignore_index`.
- **L_residual** (weight 0.2): SmoothL1Loss on the three residual heads (time, x, y), only computed on non-PAD positions. Residual targets are clamped (±1/32 beat for time, ±8 px x, ±6 px y).

### Audio Normalization

After computing `torch.log(mel + 1e-7)` on the full 129-channel tensor, mel (128 channels) and onset (1 channel) are **separately z-normalized** (mean=0, std=1) per batch. This corrects a scale mismatch where log-onset values were compressed to a narrow negative range (~[-16, 0]) while log-mel ranged wider (~[-10, 2]), which previously caused the model to ignore the onset channel.

### Curriculum Learning

Training progresses through three phases to help the model learn simpler patterns before complex ones:

1. **Epochs 1–20:** Circles only — the model only sees maps filtered to contain circles
2. **Epochs 21–40:** Circles + sliders
3. **Epochs 41–100:** Full maps (circles, sliders, spinners)

### Teacher Forcing Decay

Teacher forcing ratio linearly decays from 1.0 to 0.5 over the first 50 epochs, gradually transitioning the model from using ground truth tokens as decoder input to using its own predictions.

### Data Loading Strategy

To maximize I/O throughput on the cluster:

1. **Cold storage:** Preprocessed shards live on shared network storage (`/common/users/`)
2. **Hot copy:** At job start, all shards are copied to node-local RAM disk (`/dev/shm`) for near-zero read latency
3. **WebDataset streaming:** Shards are read as `.tar` files using the WebDataset library with 8 workers, prefetch factor 4, persistent workers, and pinned memory

### Audio Augmentation

After epoch 5, training samples are augmented with random time-stretching (0.9×–1.1×) applied to the audio waveform. Token timing deltas are inversely adjusted to maintain alignment.

### Checkpoint Management

The top 3 checkpoints by validation loss are retained. A `best.pt` symlink always points to the best checkpoint. Checkpoints include model weights, optimizer state, scheduler state, and GradScaler state for seamless resumption.

### NaN Detection and Recovery

The training loop includes comprehensive non-finite detection. At each step, all model outputs and losses are checked for NaN/Inf values. If detected, the batch is logged with diagnostic information (mel stats, token ranges, per-head finiteness) and skipped to prevent gradient corruption.

## Evaluation Metrics

Four metrics (`src/metrics.py`) measure generation quality. **Timing** and **hit F1** require **`ms_per_beat`** (from each sample's BPM) so beat tokens decode to the same millisecond times used in ground truth.

| Metric | Description |
|--------|-------------|
| **Token Edit Distance** | Normalized Levenshtein distance between predicted and ground truth token sequences. Lower is better. |
| **Timing MAE** | Mean absolute error of hit object times in **milliseconds** after `detokenize_to_hitobjects` (beat deltas × `ms_per_beat` + residuals). Greedy matching of predicted to true timestamps. |
| **Hit F1** | F1 score where a true positive requires matching within **50 ms** and 50 px of a ground truth hit object. (Tolerance was increased from 20 ms to account for beat-bin quantization at typical BPMs.) |
| **Slider IoU** | Intersection-over-Union of slider curves (Shapely buffered line geometry, 10 px buffer). |

**Training vs offline eval:** During training, validation prints these metrics on **greedy one-step-ahead** predictions over a **subset** of the val batch (fast, approximate). **`python -m src.eval`** runs the same metric definitions on the full split with your checkpoint for a more stable report.

## Inference

The inference script (`src/inference.py`) generates a complete `.osu` file from an MP3. **Difficulty and BPM are required conditioning inputs** that directly control the model's output:

```bash
python -m src.inference \
    --input song.mp3 \
    --output generated.osu \
    --checkpoint models/best.pt \
    --difficulty 4.5 \
    --bpm 174 \
    --temperature 0.9
```

- **`--difficulty`**: Star rating (mapped to 5-bin scheme: Easy/Normal/Hard/Insane/Expert). Controls map density and structure.
- **`--bpm`**: Song tempo in BPM. Use `0` to auto-estimate via librosa beat tracking (fallback: 120).

### Generation Process

1. Load and resample audio to 22,050 Hz mono
2. Get BPM: use `--bpm` if provided, else estimate via librosa
3. Convert difficulty star rating to conditioning bin (0–4)
4. Slide a 6-second window across the song with 3-second stride
5. For each window: compute **mel spectrogram** and **onset strength** (aligned to mel frames), concatenate to **129 channels**, apply **`log(mel + 1e-7)`** then **per-channel-group z-normalization** (mel and onset normalized separately) to match training; run autoregressive generation with conditioning (diff_id, bpm)
6. Detokenize tokens + residuals to hit objects with **`ms_per_beat = 60000 / bpm`**
7. Deduplicate overlapping windows (remove objects within 5 ms of each other)
8. Write the complete `.osu` file with metadata, timing points, and hit objects

## Project Structure

```
osu_beatmap_generator/
├── src/
│   ├── model.py          # OsuMapper: AST encoder + onset gate + Transformer decoder
│   ├── tokenizer.py       # Grid tokenizer with residual precision
│   ├── dataset.py         # WebDataset/streaming data pipeline + z-norm
│   ├── train.py           # Training loop with curriculum + AMP + diagnostics
│   ├── preprocess.py      # HuggingFace → WebDataset shard pipeline
│   ├── inference.py       # MP3 → .osu generation (with onset z-norm)
│   ├── eval.py            # Evaluation on held-out test split
│   ├── metrics.py         # Edit distance, timing MAE, hit F1, slider IoU
│   └── splits.py          # Deterministic train/val/test shard splitting
├── scripts/
│   └── diagnose_timing.py # Standalone diagnostic script for checkpoint analysis
├── configs/
│   ├── train_a100.sh      # SLURM job script for A100 GPU
│   ├── train_any_gpu.sh   # SLURM job script for any GPU
│   ├── train_multi_gpu.sh # Multi-GPU training config
│   └── preprocess.sh      # SLURM job script for preprocessing
├── requirements.txt       # Python dependencies
├── Singularity.def        # Container definition for cluster
└── README.md
```

## Setup and Usage

### Requirements

- Python 3.10+
- CUDA-capable GPU (16+ GB VRAM recommended)
- ~15 GB disk space for preprocessed shards

### Installation

```bash
git clone https://github.com/Anime-Shhh/osu-beatmap-generator.git
cd osu-beatmap-generator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Preprocessing

```bash
python -m src.preprocess \
    --output_dir ./data/shards \
    --max_shards 500 \
    --examples_per_shard 200
```

### Training

```bash
python -m src.train \
    --data_dir ./data/shards \
    --model_dir ./models \
    --log_dir ./logs \
    --epochs 100 \
    --lr 1e-4 \
    --tf32 \
    --compile
```

Or via SLURM:

```bash
sbatch configs/train_any_gpu.sh
```

### Evaluation

```bash
python -m src.eval \
    --data_dir ./data/shards \
    --split_file ./logs/splits.json \
    --split_name test \
    --checkpoint ./models/best.pt
```

### Inference

```bash
python -m src.inference \
    --input song.mp3 \
    --output generated.osu \
    --checkpoint ./models/best.pt \
    --difficulty 4.5 \
    --bpm 174
```

## Current Status

The model uses **explicit conditioning** (difficulty + BPM), **beat-relative time tokens** (128 bins, 1/16-beat resolution), **129-channel** audio input (log-mel + onset with per-channel z-normalization), **onset gating** in the encoder, and **CosineAnnealingWarmRestarts** (`T₀ = epochs/5`, `T_mult=2`) for the learning rate.

**Latest training results (100 epochs):** val CE loss ~**2.11**, edit distance ~**0.44**, timing MAE ~**396 ms**, hit F1 ~**0.11**. This represents a significant improvement over the previous model (timing MAE ~900 ms, hit F1 ~0.04).

### Key Technical Decisions

- **Explicit conditioning over prediction:** Difficulty and BPM are conditioning inputs, not tokens the decoder predicts.
- **Beat-relative timing:** Inter-hit deltas are expressed in beats so the same rhythm maps to the same tokens at different BPMs; milliseconds are recovered at detokenize time with `ms_per_beat`.
- **Onset gating:** Per-window onset strength is projected alongside mel, then merged into the AST encoder output via a learned sigmoid gate. This ensures the decoder receives explicit rhythm cues while AST still processes only its pretrained 128-channel input.
- **Per-channel z-normalization:** After log transform, mel and onset channels are separately z-normalized to comparable scales, preventing the model from ignoring the lower-variance onset signal.
- **AST over training from scratch:** Transfer learning from AudioSet provides strong audio feature extraction with minimal fine-tuned layers.
- **Grid tokenization over direct regression:** Placement uses 1,024 spatial bins + residuals; time uses 128 beat bins + beat residuals.
- **bfloat16 over FP16:** Mixed precision uses bfloat16 to avoid overflow in cross-entropy loss. Log-mel (and log onset) plus clamped residuals improve stability.
- **WebDataset over standard Dataset:** Streaming `.tar` shards enables efficient I/O when training data exceeds RAM.
- **Sliding window over full-song generation:** Processing 6-second windows keeps sequence lengths manageable (~100–500 tokens) and memory bounded.
- **torch.compile compatibility:** Integer counters were removed from the encoder forward path to avoid triggering repeated recompilation by `torch.compile`.

## Recent Changes

### Onset Gating Mechanism (model.py)

Previously, the onset channel was projected through `mel_proj` but effectively discarded when AST succeeded — the encoder output came entirely from the AST pathway. A **learned sigmoid gate** (`mel_gate`) now merges the `mel_proj` output (containing onset information) with the upsampled AST output:

```
projected = mel_proj(mel)          # [B, T, D] — all 129 channels including onset
ast_out = upsample(AST(mel_128))   # [B, T', D] — AST output from 128 mel channels
proj_aligned = interpolate(projected, size=T')
gate = sigmoid(linear(proj_aligned))
encoder_out = ast_out + gate * proj_aligned
```

The onset zero-out diagnostic confirms the onset channel now influences predictions (logit diff grew from 0.0000 to 0.005 over training).

### Per-Channel Z-Normalization (dataset.py, inference.py)

After `log(mel + 1e-7)`, mel (128 channels) and onset (1 channel) are **separately z-normalized** per batch to mean=0, std=1. This fixes a scale mismatch: log-onset values were compressed to ~[-16, 0] while log-mel ranged ~[-10, 2], causing the model to treat onset as noise.

### Hit F1 Tolerance Fix (metrics.py)

The `time_tolerance_ms` for hit F1 was increased from 20 ms to **50 ms**. At common BPMs (e.g., 174 BPM → 344 ms/beat), a single 1/16-beat bin spans ~21.5 ms, so 20 ms tolerance was smaller than the quantization step and made F1 universally 0.

### Validation Metric Averaging Fix (train.py)

`total_hit_f1` accumulation and `num_metrics` increment were moved inside the per-sample loop. Previously they were outside the loop, causing hit F1 to only count the last sample per batch and metrics to be averaged incorrectly.

### torch.compile Recompilation Fix (model.py)

Removed `self.ast_success` and `self.ast_fallback` integer counters from `AudioEncoder.forward()`. `torch.compile` treated these as static graph guards, triggering a full recompilation every time they changed (causing the first ~8 training steps to take 2–32 seconds each instead of ~0.15s). Also removed the associated `get_ast_stats()` method and its call in the training loop.

### Validation Diagnostics (train.py, dataset.py)

- **Onset zero-out test:** At validation step 0, the onset channel is zeroed and logit differences are measured. Values < 0.01 indicate the model may be ignoring onset.
- **Collate diagnostics:** One-time print of mel/onset statistics after z-normalization to verify both feature groups are properly scaled.
- **Token analysis:** First validation sample logs token breakdown, bin distributions, timing accuracy, and category confusion for debugging.

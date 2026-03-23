# OsuM — AI Beatmap Generation from Audio

OsuMapper is an end-to-end deep learning system that automatically generates [osu!](https://osu.ppy.sh/) beatmaps from raw audio. Given an MP3 file of any song, the model produces a fully playable `.osu` beatmap file containing timed hit circles, sliders, and spinners — the core gameplay elements of osu!standard mode.

The model uses an **encoder–decoder Transformer architecture** with a pretrained audio encoder (Audio Spectrogram Transformer) and an autoregressive decoder that generates a structured token sequence representing beatmap objects. A novel **hybrid discrete + continuous output** scheme enables both coarse grid-based placement and sub-pixel precision via learned residual offsets.

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

## Problem Statement

An osu! beatmap defines *when* and *where* a player must click, drag, or spin during a song. Human mappers spend hours manually placing hit objects to match the song's rhythm, melody, and intensity. This project frames beatmap generation as a **sequence-to-sequence problem**:

- **Input:** Mel spectrogram of a ~6-second audio window **plus** user-provided conditioning: difficulty (star-rating bin) and BPM
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
2. **Mel spectrogram extraction:** 128-bin mel spectrograms with 2048-point FFT and ~20 ms hop length
3. **BPM extraction:** BPM is parsed from the `.osu` file's `[TimingPoints]` section (60000 / ms_per_beat for the first uninherited timing point)
4. **Difficulty binning:** Star rating is mapped to a 5-bin scheme (Easy, Normal, Hard, Insane, Expert) for conditioning
5. **Sliding window chunking:** Each song is split into overlapping 6-second windows with a 3-second stride, predicting the middle 3 seconds of each window
6. **Tokenization:** Each window's hit objects are converted to a discrete token sequence with continuous residuals (see [Tokenization](#tokenization-scheme))
7. **Shard writing:** Processed (mel, tokens, difficulty_id, bpm) tuples are written as WebDataset `.tar` files

**Scale:** 500 shards × 200 examples/shard = **100,000 training examples** (~15 GB total)

**Note:** The shard format includes `difficulty_id` and `bpm` per window. Re-preprocessing is required if upgrading from older versions.

### Data Splits

Shards are deterministically split 80/10/10 into train/val/test sets (400/50/50 shards). Split manifests are written to JSON for reproducibility.

## Tokenization Scheme

The tokenizer (`src/tokenizer.py`) converts continuous beatmap data into a discrete vocabulary that a Transformer can model, while preserving sub-bin precision through continuous residual outputs.

### Vocabulary (1,238 tokens)

| Range | Count | Purpose |
|-------|-------|---------|
| 0–3 | 4 | Special tokens: `PAD`, `BOS`, `EOS`, `SEP` |
| 4–9 | 6 | Object types: `CIRCLE`, `SLIDER_START`, `SLIDER_CONTROL`, `SLIDER_END`, `SPINNER_START`, `SPINNER_END` |
| 10–13 | 4 | Curve types: `BEZIER`, `LINEAR`, `PERFECT`, `CATMULL` |
| 14–1037 | 1,024 | Position bins (32×32 grid over the 512×384 playfield) |
| 1038–1237 | 200 | Time delta bins (0–1990 ms in 10 ms steps) |

Difficulty is **not** a token — it is a separate conditioning input (integer 0–4 for Easy through Expert).

### Two-Stage Precision

Each position and time token represents a coarse bin. To recover precise coordinates, the model also predicts **continuous residual offsets**:

- **Position residuals:** `(x_offset, y_offset)` in pixels, range ±8 px horizontally, ±6 px vertically (half the bin width/height)
- **Time residuals:** `time_offset` in milliseconds, range ±5 ms

At inference, the final coordinate is: `bin_center + predicted_residual`.

### Token Sequence Format

Each hit object is encoded as a subsequence within the full sequence. Difficulty and BPM are provided as conditioning inputs, not tokens:

```
[BOS] [time₁] [CIRCLE] [pos₁] [time₂] [SLIDER_START] [pos₂] [CURVE_TYPE] [SLIDER_CONTROL] [cp_pos] ... [SLIDER_END] [end_pos] ... [EOS]
```

- **Circles:** `<time_bin> <CIRCLE> <pos_bin>` (3 tokens)
- **Sliders:** `<time_bin> <SLIDER_START> <pos_bin> <curve_type> [<SLIDER_CONTROL> <cp_pos>]* <SLIDER_END> <end_pos>` (variable length)
- **Spinners:** `<time_bin> <SPINNER_START> <pos_bin> <duration_bin> <SPINNER_END>` (5 tokens)

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

The encoder converts a mel spectrogram into a sequence of audio embeddings:

1. **Backbone:** [Audio Spectrogram Transformer (AST)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) — a Vision Transformer pretrained on AudioSet for audio classification. Most layers are frozen; only the last 2 Transformer layers are fine-tuned.
2. **Position embedding resizing:** AST was pretrained on 10-second clips (1024 frames). Since our input is ~6 seconds (301 frames), the positional embeddings are bilinearly interpolated to match the actual mel length at runtime.
3. **Temporal upsampling:** A 1D convolutional stack with 2× upsampling increases the temporal resolution of AST's output.

BPM is provided as a conditioning input, not predicted.

### Beatmap Decoder

The decoder autoregressively generates the token sequence conditioned on `[cond | encoder_out]`:

1. **Token + positional embeddings:** Learned embeddings for the 1,238-token vocabulary and up to 2,048 sequence positions
2. **Transformer decoder:** 6 layers, 8 attention heads, 2048-dim feedforward, pre-norm architecture with causal masking
3. **Cross-attention:** Each decoder layer attends to the conditioning token and full audio encoder output
4. **Output heads (hybrid):**
   - `discrete_head`: Linear projection to 1,238 logits for next-token prediction (trained with CrossEntropyLoss)
   - `time_res_head`: Linear → scalar for time residual offset
   - `x_res_head`: Linear → scalar for x-position residual offset
   - `y_res_head`: Linear → scalar for y-position residual offset

### Why Hybrid Discrete + Continuous?

Pure discretization into a 32×32 grid would limit placement precision to ±8 px — too coarse for a game where precision matters. Pure regression would require the model to predict exact coordinates from scratch. The hybrid approach gets the best of both worlds: the discrete head handles the *combinatorial* decision of "which region," while the regression heads handle the *fine-grained* adjustment of "exactly where within that region."

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

- **L_discrete** (weight 0.8): CrossEntropyLoss over 1,238 classes for next-token prediction. PAD tokens are masked via `ignore_index`.
- **L_residual** (weight 0.2): SmoothL1Loss on the three residual heads (time, x, y), only computed on non-PAD positions. Residual targets are clamped to documented ranges (±5 ms time, ±8 px x, ±6 px y).

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

Four metrics (`src/metrics.py`) measure generation quality:

| Metric | Description |
|--------|-------------|
| **Token Edit Distance** | Normalized Levenshtein distance between predicted and ground truth token sequences. Lower is better. |
| **Timing MAE** | Mean absolute error of hit object timing in milliseconds, computed by greedy matching predicted to true timestamps. |
| **Hit F1** | F1 score where a true positive requires matching within 20 ms and 50 px of a ground truth hit object. Measures both precision (no spurious objects) and recall (no missed objects). |
| **Slider IoU** | Intersection-over-Union of slider curves, computed using Shapely buffered line geometry with a 10 px buffer. Measures how well predicted slider shapes match ground truth. |

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
5. For each window: encode mel + conditioning (diff_id, bpm) → autoregressively sample tokens with temperature
6. Detokenize tokens + residuals back to hit objects with absolute timestamps
7. Deduplicate overlapping windows (remove objects within 5 ms of each other)
8. Write the complete `.osu` file with metadata, timing points, and hit objects

## Project Structure

```
osu_beatmap_generator/
├── src/
│   ├── model.py          # OsuMapper: AST encoder + Transformer decoder
│   ├── tokenizer.py       # Grid tokenizer with residual precision
│   ├── dataset.py         # WebDataset/streaming data pipeline
│   ├── train.py           # Training loop with curriculum + AMP
│   ├── preprocess.py      # HuggingFace → WebDataset shard pipeline
│   ├── inference.py       # MP3 → .osu generation
│   ├── eval.py            # Evaluation on held-out test split
│   ├── metrics.py         # Edit distance, timing MAE, hit F1, slider IoU
│   └── splits.py          # Deterministic train/val/test shard splitting
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

The model has been refactored to use **explicit conditioning** for difficulty and BPM. These are now user-provided inputs rather than prediction targets, improving controllability and training stability.

**Breaking change:** Old checkpoints and preprocessed shards are incompatible. Re-run preprocessing (`sbatch configs/preprocess.sh`) before training.

### Key Technical Decisions

- **Explicit conditioning over prediction:** Difficulty and BPM are conditioning inputs, not tokens the decoder predicts. This gives the user direct control over map style and reduces model instability.
- **AST over training from scratch:** Transfer learning from AudioSet provides strong audio feature extraction with minimal fine-tuning.
- **Grid tokenization over direct regression:** Framing placement as classification over 1,024 bins (with residual refinement) gives the model a structured output space. Residual targets are clamped and SmoothL1Loss is used for numerical stability.
- **bfloat16 over FP16:** Mixed precision uses bfloat16 to avoid overflow in cross-entropy loss. Log-mel spectrograms and clamped residuals further improve training stability.
- **WebDataset over standard Dataset:** Streaming `.tar` shards enables efficient I/O when training data exceeds RAM.
- **Sliding window over full-song generation:** Processing 6-second windows keeps sequence lengths manageable (~100–500 tokens) and memory bounded.

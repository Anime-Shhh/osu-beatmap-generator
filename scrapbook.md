# What a good model would look like after full training


| Metric        | Likely achievable |
| ------------- | ----------------- |
| Val loss      | ~2.05–2.15        |
| Edit distance | ~0.75–0.85        |
| Timing MAE    | ~500–650 ms       |
| Hit F1        | ~0.08–0.14        |


---

# What an excellent version would look like (harder to reach)


| Metric                 | Strong result |
| ---------------------- | ------------- |
| Val loss               | ~1.9–2.0      |
| Edit distance          | ~0.6–0.7      |
| Timing MAE             | ~250–400 ms   |
| Hit F1                 | ~0.15–0.30    |
| Main issues to address |               |


### old model metrics before beat-relative tokenizer and onset strength featuring(123159):
  train_loss=1.5598 | val_loss=2.2213
  edit_dist=0.8944 | timing_mae=836.4ms | hit_f1=0.0388
  lr=0.000050 | time=93.3s
  encoder_ast: success=67600 fallback=0

Training complete. Best val_loss: 2.2224

---

## What is currently "untraditional" (the real weaknesses)

These are the biggest things holding it back.

### 1. You're predicting time in milliseconds instead of rhythm space (biggest issue)

Right now you effectively model:

audio → predict time delta (ms)

But good music models model:

audio → rhythm structure → time

Music is structured as:

Beats
Measures
Subdivisions

Not milliseconds.

Milliseconds are noisy from the model's perspective.

Example:

Same rhythm:

Song A (120 BPM):

500 ms between beats

Song B (180 BPM):

333 ms between beats

Model sees:

500 vs 333

But musically they are identical.

This forces your model to relearn rhythm per BPM.

That slows learning dramatically.

This is probably your #1 timing bottleneck.

### 2. No explicit rhythm representation

Your model has:

Audio features
Tokens
Residuals

But nothing that explicitly represents:

Beat phase
Bar structure
Downbeats
Onsets

Good music models almost always add at least one rhythm feature.

Because rhythm is the main signal.

Right now you're hoping the transformer extracts it implicitly.

That works, but slowly.

### 3. Decoder must learn too many things at once

Your decoder currently learns:

Difficulty
Density
Timing
Position
Object type
Flow patterns

All simultaneously.

Strong osu models reduce burden by giving:

Difficulty
Tempo
Sometimes density hints

So decoder focuses on:

Placement + timing quality.

You already fixed difficulty and BPM.

That was a big improvement.

### 4. No density conditioning (subtle but important)

Difficulty isn't just star rating.

It's also:

Notes per second.
Rhythm complexity.
Spacing aggression.

Your model only gets:

difficulty category

But not:

target density.

So it must infer:

How many notes to place.

That increases variance.

Good models often condition on:

Notes per second (NPS)
Object count per window

Even if implicitly.

### 5. Loss treats all tokens equally (small issue)

Right now:

Cross entropy treats:

Timing tokens
Position tokens
Type tokens

equally.

But timing mistakes hurt quality most.

Some osu models weight:

Timing tokens higher.

This improves rhythm learning.

## What would make your architecture "good"

These are the highest ROI additions.

Ordered by impact.

### 1. Represent time in beat space instead of ms (highest impact)

Instead of:

time_bin(delta_ms)

Use:

beat_delta = delta_ms / ms_per_beat
beat_bin(beat_delta)

Now model learns:

0.25 beat
0.5 beat
1 beat

Which generalizes across BPM.

This is probably the single biggest upgrade possible.

This alone can reduce timing MAE a lot.

### 2. Add beat phase feature (very powerful)

Add:

phase = (time % ms_per_beat) / ms_per_beat

Model now knows:

Where inside beat event occurs.

Music models love this.

Because:

Most hits occur on subdivisions.

### 3. Add density conditioning (very effective)

Add:

notes_per_second

as conditioning.

Because difficulty alone is vague.

Example:

Two 5★ maps:

One stream heavy.
One jump heavy.

Density differs.

Conditioning example:

cond = difficulty_embed
     + bpm_embed
     + density_embed

This stabilizes generation.

Predict density from audio automatically

You can estimate rhythm activity from:

Onset strength.
Spectral flux.
Tempo changes.

Example approach:
```
onset_env = librosa.onset.onset_strength(y=audio)
density_estimate = mean(onset_env)
```
This gives a rhythm activity measure.

More rhythmic songs → higher density.

This requires no user input.

Music models often do this implicitly.


### 4. Add audio onset features (very high value)

Right now mel spectrogram is general.

But osu mapping depends heavily on:

Onsets (note starts).

Add onset strength feature:

Librosa has:

onset_strength()

Many music models use this.

It gives model:

"notes likely belong here".

Huge signal.

Often improves timing more than architecture changes.

### 5. Predict beat position instead of raw time tokens

Instead of:

[TIME_DELTA]

Use:

[BEAT_DELTA]
[BEAT_PHASE]

This separates:

Rhythm structure.
Micro timing.

Cleaner learning.

### 6. Add token type embeddings (small but standard)
Add embedding indicating:

TIME token
POS token
TYPE token

Transformers like knowing token roles.

Very standard in sequence modeling.

### 7. Weight timing loss slightly higher (small change)
Example:
loss =
0.7 token CE
0.2 residual loss
0.1 timing emphasis

Or:

Multiply time-token CE by 1.3.
Small change. Can improve timing.
What strong osu ML architectures usually contain

Typical mature osu model has:

Audio encoder ✔
Difficulty conditioning ✔
Tempo conditioning ✔
Beat-space timing ✔
Residual precision ✔
Onset features ✔
Density conditioning ✔

You're missing:

Beat timing
Onset info
Density signal

Everything else is solid.

What I would fix first (highest ROI order)

If this was my project:

I would do exactly this:

#### Tier 1 (biggest improvements)

1 Add beat-space timing representation
2 Add onset strength feature

#### Tier 2 (medium improvements)

3 Add density conditioning
4 Add beat phase feature

#### Tier 3 (small polish)

5 Token type embeddings
6 Slight timing loss weighting

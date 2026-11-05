# November 4: recover training and compare Kaguya inference

The October training jobs 814837/814838 never started: their acquisition
dependency 814768 failed and Slurm retained them as `DependencyNeverSatisfied`.
There was no new trained checkpoint to evaluate at the beginning of this check.
The project allocation was over its 100GiB quota despite free space on the
underlying filesystem. A first recovery attempt there also failed; new writes
and logs now use the separate home allocation. Historical runs are preserved.

## Recovered data and completed-work provenance

`recover_flow_corpus.py` verifies the last complete journal, every feature/chart
hash, all original source identities/labels/profiles/splits and absence of Kaguya's
mapset ID before constructing a fresh trainable manifest. Unjournaled files are
not silently included. Cross-filesystem feature links point to existing verified
immutable data; native charts and small manifests are retained in the new run.

* Recovered charts: 3,436, including 2,060 additional charts.
* Independent source groups: 637 train, 84 validation, 84 test (805 total).
* Chart splits: 2,765 train, 346 validation, 325 test.
* Recovery covers the first 1,222 scanned records, not the entire planned 1,500.
* Manifest SHA256: `104c8d54338cc075baabba20809d39841a07a1729db14b5a67d7000ba343b5e4`.
* Previous training split contained 260 independent sources.

Run root: `/common/home/asj102/osu_flow_runs/run_20261104`.
Immutable training/inference source: `/common/home/asj102/osu_flow_runs/source_20261104`.
Original data/checkpoints remain in `/common/users/asj102/osu_project`.

## Controlled comparison

Both new models initialize from the same original best onset checkpoint and use
20 epochs, 1e-4 learning rate, batch size 16 and identical source splits. The
control keeps the original onset architecture; the phrase model additionally
uses the two pooled attention layers and local-density supervision implemented
in October. Placement weights are unchanged in all Kaguya runs.

The generator now permits pairing independently trained stages when the rhythm
checkpoint explicitly binds the geometry training manifest as its direct parent.
Unrelated training manifests still fail, feature extraction must match, and the
exact stage manifest hashes are recorded in the generation audit. A regression
test covers both the permitted parent and rejected unrelated cases. This avoids
silently bypassing provenance to use the new rhythm stage with old geometry.

Each model is compared on the exact original Kaguya MP3, a 4.5-star request, the
same training-only settings guide, 48 integration steps and 16 seeds (1731–1746).
Thresholds come from generated validation-window calibration, not Kaguya. Full-song
validation also compares all models at the fixed .5 threshold on the identical
12 frozen validation songs and two seeds. Kaguya is a development example and
does not select model weights or the recommended model.

For inference diagnosis, the star gate is widened to ±2 stars so an overly hard
map can still be inspected; all other native checks remain active. The report
separately counts candidates satisfying the normal ±.5-star tolerance. The chosen
candidate minimizes star error, not distance from the reference map. Any playable
artifact is labeled with its measured stars, rather than claiming the 4.5 target.

The original nearest requested difficulty is Confession, 4.4874 stars, with 324
objects. Exact event matching describes similarity to that human interpretation,
not the uniqueness of a correct mapping. Sampling ranges over one song do not
establish generalization; the full-song validation source bootstrap is separate.

## Why “doubles” need a placement diagnostic too

The original has 36 short consecutive circle pairs and 13 whose raw positions
overlap within 8 pixels. The previous selected generated map has 53 short pairs
but zero overlapping pairs, and only six reference pairs are recovered at the
right times as circles. Median head spacing within its short pairs is about 200px
versus 49px in the original. Thus generating fast onsets is insufficient: the
placement stage also lacks the repeated-position behavior of stacked doubles.

These runs hold placement fixed. Any rhythm gain must not be described as fixing
stacked doubles or professional geometry. The existing geometry flow has no
explicit learned copy/repeat-position decision; that is a concrete next hypothesis
to evaluate rather than a claimed completed improvement.

## Run ledger

* Recovery 1024564 succeeded. Earlier project-allocation recovery 1024354 failed.
* Original-model fresh 16-seed inference 1024579 completed and reproduces the
  previous selected map. Mean onset F1 at 50ms: 70.60%; mean object count 249.81;
  short native circle pairs recovered: mean 5.875, range 4–8 of 36.
* Control training/evaluation: 1024575. Phrase training/evaluation: 1024576.
* Dependent Kaguya inference: 1024647 (control), 1024652 (phrase).
* Old impossible dependency jobs and replacement jobs blocked on failed recovery
  were canceled; no unrelated user jobs were canceled.

## Completed results (November 5)

Both training runs completed 20 epochs / 12,740 updates; both selected epoch 6
by validation loss. All inference jobs completed. Summary/package job 1025202
and fixed-threshold diagnostic 1045459 also completed.

Kaguya averages over the same 16 seeds (calibrated thresholds):

| Model | Onset F1 ±50ms | Onset F1 ±10ms | Objects | Reference short pairs recovered /36 |
| --- | ---: | ---: | ---: | ---: |
| Original | 70.60% | 47.22% | 249.81 | 5.88 |
| More-data control | 71.47% | 58.35% | 257.38 | 6.19 |
| More-data + phrase context | 70.73% | 58.54% | 254.06 | 6.19 |

F1 balances precision and recall of matched object start times; it is not a
percentage of overall human mapping quality. The original has 324 objects.
Timing precision improved substantially at the tight tolerance, but overall
note selection barely improved. The attention variant did not outperform the
more-data control on this song. Sound-source selection is not directly measured
by these event metrics.

Calibration chose .5/.55/.7 for original/control/phrase. Re-running the latter
two at .5 produced identical object counts and ±50ms scores; phrase ±10ms F1
was 58.52%. Threshold choice does not explain the coverage deficit.

On 12 frozen full-song validation sources, F1 at ±50ms was 59.51% original,
59.92% control and 60.09% phrase. Paired source-bootstrap 95% intervals include
zero: phrase minus original +0.582 percentage points [-0.531, +1.567]; phrase
minus control +0.169 [-0.071, +0.447]. This is insufficient evidence of a
reliable general improvement from the added context layers.

The phrase model was chosen by full-song validation, not Kaguya similarity.
Its closest-star candidate has 223 objects (104 circles,119 sliders), versus
324 (85 circles,239 sliders) in the original. Only 185/324 reference starts
match within 50ms; six of 36 short circle pairs are recovered. It has zero
short pairs within 8px versus 13 in the original. Median short-pair spacing
is 206px versus 49px. Median slider duration is 268ms versus 109ms. Matched
onset error averages 7.50ms versus 8.63ms for the old selected output, but the
selected new output has fewer matched notes (185 versus195).

None of the 16 candidates from any model meets the default 4.5 ± .5 star gate.
The packaged phrase candidate measures 5.1658 stars and is explicitly labeled
5.17 stars. It is an experimental playtest, not a successful 4.5-star or
ranked-quality result. Its archive passes ZIP CRC checks, original MP3 hash
matches, gameplay sections match the generated map, and metadata changes leave
measured difficulty unchanged.

## Interpretation and next architectural targets

The data expansion explains most of the observed tight-timing gain; extra
attention/density supervision has not demonstrated a clear additional gain.
Placement stayed fixed, so these runs cannot establish a geometry improvement.
The model still needs better event coverage/type selection (especially short
sliders), duration prediction and a learned repeat-position/stack decision.
Difficulty control also fails: selecting by closest measured stars can favor
sparser candidates while leaving placement too hard. A follow-up should test
joint rhythm/geometry difficulty conditioning and native short-pattern metrics,
with a frozen validation protocol, rather than selecting against Kaguya events.
These are hypotheses for further work, not implemented fixes or guarantees.

Evidence: [machine-readable summary](evidence/audio-native-flow-2026-11-05/comparison-summary.json).
Local user artifacts are in
`/Users/anime_shh/personalCS/projects/codex-mds/osu/2026-11-05-flow-comparison`:
`playtest/Kaguya-phrase-5.17-stars.osz` and `kaguya-comparison.png`.
OSZ SHA256: `1fe1c72f2af87b1a7e597b0d5ea4343231fa3b1b04203225e8695b2f8823a0e6`.

Validation performed: model/generation regression tests in the remote environment,
Python compilation, shell syntax, Git whitespace checks, complete Slurm training
and inference, same-audio/seed comparisons, and archive/audio integrity checks.


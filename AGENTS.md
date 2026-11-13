# Project operations

V2, coverage repair and the experimental rhythm-flow work are now merged into
`main`. The preserved `codex/v2-coverage-rhythm-flow` branch was based on the
`architecture-v2-hierarchical-beatgrid` baseline `d75f6a8`.

Codex accesses Rutgers iLab directly from the Mac through the password-authenticated
OpenSSH ControlMaster. Do not add SSH keys or use the Hermes container for access.

- Execute remote commands with `/Users/anime_shh/personalCS/projects/ilab-access/ilab 'command'`.
- If the sandbox cannot access the socket, retry with approved elevated execution
  before assuming authentication expired.
- If the master really expired, ask the user to run
  `/Users/anime_shh/personalCS/projects/ilab-access/ilab-connect` in their Mac terminal.
  The password must stay in the terminal and never enter chat or tool arguments.
- Existing deployment: `/common/users/asj102/osu_project`; Python environment:
  `/common/users/asj102/osu_project/env`.
- Leave existing job 450738 queued, per the user’s explicit instruction.
- Do not overwrite historical runs. Use a new source directory and run directory
  for changed code/data. Submit GPU work through Slurm.

The `run_20261010_v2` records were generated from synthetic sine waves and fixed
objects, despite claiming ranked provenance. They and their planner checkpoints
are unsuitable for production training/generation. See `docs/v2-audit-2026-10-13.md`.

Audio-alignment follow-up: use `configs/v2_audio_alignment_planner.json` for the
new within-bar audio path, with `--initialize-planner` for legacy weight transfer.
Keep legacy checkpoints unchanged. Do not use `--resume` to silently change
`local_audio_tokens`. See `docs/v2-audio-alignment-2026-10-14.md` for evidence,
commands, and the distinction between implementation validation and map quality.

October 17 follow-up: production V2 generation now retries empty audible bars,
audits full-song coverage, and evaluates rendered slider paths including native
length extension. Keep these audits enabled when selecting playable outputs.
The separate `src/rhythm_flow/` experiment predicts timing signals only; its
64-step smoke checkpoint is not a usable beatmap model. See
`docs/v2-coverage-and-diffusion-2026-10-17.md` for evidence and run locations.

October 18 rhythm-flow evaluation: job 541156 completed 30 epochs; the retained
checkpoint is `v2_runs/run_20261017_rhythm_training/train/best.pt`. This is still
rhythm-only, not a native-geometry or OSZ checkpoint. Matched test onset F1 at
50 ms is 42.3%, with poor circle recall and hold durations. See
`docs/rhythm-flow-results-2026-10-18.md` before claiming full-map quality or
comparing its loss numerically with V2's unrelated objectives. Evaluation jobs
549284 and 549331 completed; no continuing training or monitoring was scheduled.

October 25 audio/native flow work is isolated on `codex/audio-native-flow` in
`osu-diffusion-lab`. The user explicitly authorized iLab access, edits and Slurm
training toward MP3 + difficulty -> human-quality OSZ. Read
`docs/audio-native-flow-worklog-2026-10-25.md` for current paths, source snapshots,
jobs and remaining evaluation work before continuing. Both new 60-epoch training
runs finished; this alone does not establish professional output quality. The new
native-label cache preserves all prior source splits. Do not reuse a saved test
comparison as an untouched test after tuning on its results.

Latest user steering: prioritize missing rhythmic objects, incorrect timing/sound
selection and short two-note patterns. Exact-original-audio Kaguya comparison and
12-source full-song validation are completed; read `docs/rhythm-followup-2026-10-25.md`.
November 5 update: expansion 814768 failed and dependent 814837/814838 never
trained. Recovery and replacement 20-epoch control/phrase runs now completed,
along with all Kaguya comparisons. Read `docs/inference-check-2026-11-04.md` and
its evidence before further work. Timing improved, but coverage, stacked doubles
and difficulty control remain poor; added phrase context has no conclusive gain.
The project allocation exceeded quota; new outputs use
`/common/home/asj102/osu_flow_runs/run_20261104`. Preserve existing artifacts.
No evaluation jobs remain pending from this check.
The user-provided original Kaguya MP3 is shifted relative to the older inference
MP3: never compare those files' raw timestamps without audio alignment.


November 5 upstream baseline: the user explicitly requested osu-dreamer.
`docs/osu-dreamer-review-2026-11-05.md` records source review and completed job
1045509: 16 Kaguya inferences using published v8 weights and unchanged v8.0 code.
Run/source/dependencies/weights live under
`/common/home/asj102/osu_flow_runs/dreamer_v8_20261105` (home allocation).
It improves Kaguya ±50ms onset coverage but not all timing/type/double metrics;
training overlap is unknown and upstream export lacks musical timing recovery.
Do not claim this was trained locally, is a held-out score or is ranked-ready.
The upstream baseline is a promising direction for joint chart/cursor modeling;
retain it unchanged when testing adaptations. No jobs remain from this check.


November6 pilot: isolated `src/dreamer_adapter` implements the approved joint
latent AE/VAE experiment; see `docs/dreamer-pilot-2026-11-05.md`. Preparation
1046631 and verified compression1047507 completed; no full training is launched.
Home quota is below the required8GiB. Preserve historical data and do not relax
the threshold. The uv cache is inventoried, not authorized for removal.
Deployed source snapshots are immutable; upload a new snapshot for changes.
Thirteen focused tests and initial real-data GPU smoke passed; further changes
require verification before submitting the coordinator. All pilot GPU work,
including failures, shares the48-hour budget ledger.

The user subsequently authorized cleaning obsolete project `codex_*` folders.
All34 were removed after an off-host hash-verified recovery archive; relevant
records and unique artifacts are in `osu_project/legacy_snapshot_records`.
Project quota reclaimed59.39MiB. Home quota is separate and remains below8GiB;
uv-cache removal still lacks authorization, and no new GPU jobs were submitted.

Latest update: user explicitly authorized uv-cache cleaning and eleven numbered
V2/legacy checkpoint deletions; both completed. Six best/latest checkpoints were
hash-verified unchanged; synthetic-run checkpoints remain. Home now has over16GiB
free.13 tests passed after cleanup. Remote ledgers: pilot_20261105/approved-cleanup/.
source_d846f29 is deployed; expanded GPU smoke submission hit SSH expiration
before execution. Reconnect, then submit smoke and the gated pilot.

Pilot is now submitted as job1062088 from immutable source_afdcdea. Cleanup
reclaimed8.84GiB home/10.12GiB project; storage gate clears.14 tests passed.
Smoke1062086 failed on an overly strict measurable-star assertion under bad audio;
fixed smoke1062087 passed on A10040GB and retained invalid-output diagnostics.
Full pipeline begins with frozen baseline evaluation, then gated training.
Read pilot-status.json and pilot-1062088.log before any new submissions; do not
change the running source or reset the shared GPU budget. No quality gain yet.

November 9 upstream comparison completed: see docs/upstream-comparison-results-2026-11-09.md.
Both osu-dreamer v8 and OliBomby/osu-diffusion completed 2,000-update pretrained
fine-tuning, jobs1087334/1087335, followed by successful Kaguya comparison1087336.
All related jobs ended successfully. Six hash/audio-verified archives and report
are on the Mac in osu-beatmap-generator/codex files/upstream-kaguya-20261107/.
Dreamer selected4.24 stars for4.6 request; geometry with human rhythm4.58, generated
chain4.34. Human-rhythm timing scores are inherited, not audio generation results.
All geometry exports retain strict native validation findings. No model promoted.

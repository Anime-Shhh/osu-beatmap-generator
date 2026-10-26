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
Expansion job 814768 is followed by dependent control/phrase training jobs 814837
and 814838. Check their current states; documentation records status at writing.
The user-provided original Kaguya MP3 is shifted relative to the older inference
MP3: never compare those files' raw timestamps without audio alignment.

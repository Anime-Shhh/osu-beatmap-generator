# Project operations

The active V2 branch is `architecture-v2-hierarchical-beatgrid`.

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

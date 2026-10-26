"""Audio-only Beat This! beat/downbeat diagnostic, with reproducible weights."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from beat_this.inference import Audio2Beats
from src.v2.inference import load_audio, extract_audio_features
from src.beatmap_flow.clock import fit_global_grid


def fit_grid(beats):
    return fit_global_grid(beats)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='final0')
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    torch.set_num_threads(1)
    wave, sr = load_audio(a.audio)
    bundle, hypotheses = extract_audio_features(wave, sr)
    tracker = Audio2Beats(checkpoint_path=a.model, device='cuda', dbn=False)
    beats, downbeats = tracker(wave[0].numpy(), sr)
    beats = np.asarray(beats)*1000
    downbeats = np.asarray(downbeats)*1000
    checkpoint = Path(torch.hub.get_dir())/'checkpoints'/f'beat_this-{a.model}.ckpt'
    result = {'package': 'beat-this', 'version': importlib.metadata.version('beat-this'),
        'model': a.model, 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'audio_sha256': hashlib.sha256(a.audio.read_bytes()).hexdigest(),
        'beats_ms': beats.tolist(), 'downbeats_ms': downbeats.tolist(),
        'global_grid': fit_grid(beats), 'legacy_clock': hypotheses[0].to_dict(),
        'duration_ms': bundle.duration_ms}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('beats_ms', 'downbeats_ms')}), flush=True)


if __name__ == '__main__':
    main()

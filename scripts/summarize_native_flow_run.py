"""Collect compact run evidence and optional training-only settings guide."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--export-profile-guide', action='store_true')
    a = p.parse_args()
    result = {}
    for stage in ('onset', 'geometry'):
        history = json.loads((a.run/stage/'history.json').read_text())
        checkpoint = torch.load(a.run/stage/'best.pt', weights_only=True, map_location='cpu')
        result[stage] = {'parameters': history['parameters'], 'epochs': len(history['history']),
            'best_epoch': checkpoint['epoch'], 'best_step': checkpoint['step'],
            'best_validation': checkpoint['history'][-1]['validation'],
            'final': history['history'][-1], 'first': history['history'][0],
            'max_gpu_memory_bytes': history['max_gpu_memory_bytes'],
            'checkpoint_sha256': hashlib.sha256((a.run/stage/'best.pt').read_bytes()).hexdigest()}
    manifest_path = a.run/'cache/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if a.export_profile_guide:
        guide = {'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                 'training_profiles': [r['profile'] for r in manifest['charts'] if r['split'] == 'train'],
                 'method': 'median settings among 64 closest training chart star labels; OD/HP use their encoded mean'}
        path = a.run/'profile-guide.json'
        with path.open('x') as f: json.dump(guide, f, indent=2)
    result['cache_audit'] = json.loads((a.run/'cache/audit.json').read_text())
    path = a.run/'training-summary.json'
    with path.open('x') as f: json.dump(result, f, indent=2)
    print(json.dumps({k: {x: v for x, v in stage.items() if x not in ('first', 'final', 'best_validation')}
                      for k, stage in result.items() if k != 'cache_audit'}, indent=2))


if __name__ == '__main__': main()

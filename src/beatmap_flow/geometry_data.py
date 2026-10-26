"""Source-uniform native geometry windows, including cross-bar transitions."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset


class GeometryWindows(Dataset):
    def __init__(self, root, split, window=64, samples_per_source=16, seed=1731):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'geometry-manifest.json').read_text())
        if self.manifest['schema'] != 'osu-native-geometry-flow-cache/1':
            raise ValueError('incompatible geometry cache')
        if hashlib.sha256((self.root/'manifest.json').read_bytes()).hexdigest() != self.manifest['rhythm_manifest_sha256']:
            raise ValueError('geometry and rhythm manifests differ')
        groups, assignments = {}, {}
        for row in self.manifest['charts']:
            source = row['source_component']
            if source in assignments and assignments[source] != row['split']:
                raise ValueError('geometry source leakage')
            assignments[source] = row['split']
            if row['split'] == split: groups.setdefault(source, []).append(row)
        self.groups = list(groups.values())
        if not self.groups: raise ValueError('empty geometry split')
        self.window, self.samples, self.seed, self.epoch = window, samples_per_source, seed, 0
        self.training = split == 'train'
        self.cache = OrderedDict()

    def __len__(self): return len(self.groups)*self.samples

    def __getitem__(self, index):
        rng = random.Random(self.seed+self.epoch*1000003+index)
        group = self.groups[index//self.samples]
        row = group[rng.randrange(len(group))]
        name = row['file']
        if name not in self.cache:
            path = (self.root/name).resolve()
            if not path.is_relative_to(self.root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('geometry path/checksum mismatch')
            self.cache[name] = torch.load(path, weights_only=True, map_location='cpu')
            if len(self.cache) > 64: self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        record = self.cache[name]
        total = len(record['kinds'])
        start = rng.randrange(max(1, total-self.window+1))
        n = min(self.window, total-start)
        b = {}
        for key, value in record.items():
            selected = value[start:start+n]
            b[key] = torch.cat((selected, torch.zeros((self.window-n, *value.shape[1:]), dtype=value.dtype)), 0)
        b['mask'] = torch.arange(self.window) < n
        b['positions'] = torch.arange(start, start+self.window)
        # All dimensions use real milliseconds and physical length, no oracle BPM.
        b['features'] = torch.cat((b['condition'][..., -3:], torch.log1p(b['lengths'][..., None]/100)/4), -1)
        if self.training:
            for axis in (0, 1):
                if rng.random() < .5: b['geometry'][..., axis] = 1-b['geometry'][..., axis]
            # Continuous conditioning perturbations help generated-plan tolerance;
            # target geometry and discrete topology remain the original native chart.
            b['condition'][..., -3:] *= 1+rng.uniform(-.05, .05)
        b['source'] = row['source_component']
        return b

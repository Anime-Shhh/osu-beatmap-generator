"""Frozen source splits and uniformly sampled source groups, shared audio cache."""
import math
import random
import hashlib
import bisect

import torch
from src.rhythm_flow.data import RhythmWindows


class OnsetWindows(RhythmWindows):
    def __getitem__(self, index):
        rng = random.Random(self.seed+self.epoch*1000003+index)
        group = self.groups[index//self.samples]
        row = group[rng.randrange(len(group))]
        name = row['audio_file']
        if name not in self.cache:
            path = self.root/name
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['audio_file_sha256']:
                raise ValueError('audio cache checksum mismatch')
            self.cache[name] = torch.load(path, weights_only=True, map_location='cpu')['audio'].float()
            if len(self.cache) > 32:
                self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        audio = self.cache[name]
        start = rng.randrange(max(1, audio.shape[0]-self.window+1))
        times = (start+torch.arange(self.window))*10.
        clip = audio[start:start+self.window].T
        mask = torch.arange(self.window) < clip.shape[-1]
        clip = torch.nn.functional.pad(clip, (0, self.window-clip.shape[-1]))
        onset = torch.zeros(1, self.window)
        onset_count = torch.zeros_like(onset)
        kind = torch.full((self.window,), -1, dtype=torch.long)
        duration = torch.zeros(self.window)
        nearest = torch.full((self.window,), float('inf'))
        starts = row.setdefault('_event_starts', [e[0] for e in row['events']])
        lo = bisect.bisect_left(starts, float(times[0])-30)
        hi = bisect.bisect_right(starts, float(times[-1])+30)
        for s, e, k in row['events'][lo:hi]:
            center = math.floor(s/10+.5)-start
            if 0 <= center < self.window and mask[center]:
                onset_count[0, center] += 1
            left = max(0, math.floor(s/10-start)-3)
            right = min(self.window, math.ceil(s/10-start)+4)
            ids = torch.arange(left, right)
            distance = (times[ids]-s).abs()
            onset[0, ids] = torch.maximum(onset[0, ids], (1-distance/25).clamp_min(0))
            selected = ids[(distance <= 20) & (distance < nearest[ids])]
            nearest[selected] = (times[selected]-s).abs()
            kind[selected] = int(k)
            duration[selected] = math.log1p(max(0, e-s)/100)
        return {'audio': clip, 'profile': torch.tensor(row['profile']), 'mask': mask,
                'onset': onset*mask, 'kind': kind, 'duration': duration,
                'onset_count': onset_count,
                'source': row['source_component']}

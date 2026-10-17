"""Compact shared-audio cache with source-group split validation."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset

SCHEMA='osu-rhythm-flow-cache/1'


def validate_manifest(manifest):
    if manifest.get('schema')!=SCHEMA or manifest.get('frame_ms')!=10:
        raise ValueError('incompatible rhythm cache')
    assignments={}
    for row in manifest['charts']:
        if row['split'] not in ('train','val','test'): raise ValueError('unknown split')
        for identity in (row['audio_key'],row['source_component']):
            if identity in assignments and assignments[identity]!=row['split']:
                raise ValueError('source leakage across dataset splits')
            assignments[identity]=row['split']
        if Path(row['audio_file']).name != row['audio_file']:
            raise ValueError('audio cache paths must be filenames')
        if len(row['profile'])!=6: raise ValueError('profile mismatch')


def targets_for_times(times,events):
    target=torch.zeros(5,len(times))
    for start,end,kind in events:
        target[kind]=torch.maximum(target[kind],(1-(times-start).abs()/25).clamp_min(0))
        if kind in (1,2):
            target[2+kind]=torch.maximum(target[2+kind],((times>=start)&(times<end)).float())
    return target


class RhythmWindows(Dataset):
    def __init__(self,root,split,window=1024,samples_per_source=16,seed=1731):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        validate_manifest(self.manifest)
        self.rows=[r for r in self.manifest['charts'] if r['split']==split]
        self.groups={}
        for row in self.rows: self.groups.setdefault(row['source_component'],[]).append(row)
        self.groups=list(self.groups.values())
        if not self.groups: raise ValueError('empty requested split')
        self.window=window;self.samples=samples_per_source;self.seed=seed;self.epoch=0;self.cache=OrderedDict()
        self.training=split=='train'
    def __len__(self): return len(self.groups)*self.samples
    def __getitem__(self,index):
        rng=random.Random(self.seed+self.epoch*1000003+index)
        group=self.groups[index//self.samples];row=group[rng.randrange(len(group))]
        name=row['audio_file']
        if name not in self.cache:
            path=self.root/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['audio_file_sha256']: raise ValueError('audio cache hash mismatch')
            self.cache[name]=torch.load(path,weights_only=True,map_location='cpu')['audio'].float()
            if len(self.cache)>8: self.cache.popitem(last=False)
        self.cache.move_to_end(name);audio=self.cache[name]
        start=rng.randrange(max(1,audio.shape[0]-self.window+1))
        clip=audio[start:start+self.window].T
        mask=torch.arange(self.window)<clip.shape[-1]
        clip=torch.nn.functional.pad(clip,(0,self.window-clip.shape[-1]))
        times=(start+torch.arange(self.window))*10.
        target=targets_for_times(times,row['events'])*mask
        return {'audio':clip,'target':target,'profile':torch.tensor(row['profile']),
                'mask':mask,'source':row['source_component']}

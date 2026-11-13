from .contracts import artifact_path
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .contracts import digest, validate_sources


def read_manifest(path,verify=False):
    path=Path(path);m=json.loads(path.read_text());validate_sources(m['rows'])
    from . import FEATURE_KEY,UPSTREAM_COMMIT
    if m['features']!=FEATURE_KEY or m['upstream']!=UPSTREAM_COMMIT:raise ValueError('feature/upstream mismatch')
    if digest(path.parent/'frozen-splits.json')!=m['frozen_splits_sha256']:raise ValueError('split manifest changed')
    if 'reference_chart_hashes' in m:
        for split in ('val','test'):
            chosen=m['reference_chart_hashes'][split]
            if set(chosen)!=set(m['groups'][split]):raise ValueError('reference sources changed')
            for group,sha in chosen.items():
                if not any(r['group']==group and r['sha256']==sha and r['split']==split for r in m['rows']):
                    raise ValueError('frozen reference chart missing')
    if verify:
        for a in m['audio'].values():
            if digest(path.parent/a['path'])!=a['sha256']:raise ValueError('audio feature changed')
        for r in m['rows']:
            if digest(path.parent/r['map_path'])!=r['map_sha256'] or digest(artifact_path(r['native_path']))!=r['sha256']:
                raise ValueError('chart changed')
    return m


def load_pair(root,m,row):
    from osu_dreamer.data.beatmap.encode import read_beatmap
    path=Path(root)/m['audio'][row['audio_key']]['path']
    if path.suffix=='.npz':
        with np.load(path) as f:a=f['spec'].astype('float32')/255
    else:a=np.load(path).astype('float32')/255
    with (Path(root)/row['map_path']).open('rb') as f:x,labels=read_beatmap(f)
    return torch.from_numpy(a),torch.tensor(x,dtype=torch.float32),torch.tensor(labels,dtype=torch.float32)


def selected_rows(m,split):
    if split not in ('val','test'):raise ValueError(split)
    if 'reference_chart_hashes' not in m:raise ValueError('freeze reference chart identities before evaluation')
    result=[]
    for group in m['groups'][split]:
        candidates=[r for r in m['rows'] if r['group']==group and r['split']==split]
        result.append(next(r for r in candidates if r['sha256']==m['reference_chart_hashes'][split][group]))
    return result


def freeze_references(frozen, available):
    """Select from all originally eligible native charts, never surviving conversions."""
    from osu_dreamer.osu.beatmap import Beatmap
    native_root=artifact_path(available[0]['native_path']).parent
    choices={s:{} for s in ('val','test')}
    for split in choices:
        for group in frozen['groups'][split]:
            candidates=[]
            for r in frozen['rows']:
                if r['group']!=group or r['split']!=split:continue
                path=native_root/Path(r['path']).name
                if digest(path)!=r['sha256']:raise ValueError('reference native chart changed')
                bm=Beatmap(path.read_text(encoding='utf-8-sig'))
                if int(bm.mode)!=0:raise ValueError('nonstandard reference chart')
                candidates.append((abs(float(bm.sr)-4.5),r['sha256']))
            sha=min(candidates)[1]
            if not any(r['sha256']==sha for r in available):raise ValueError('nearest frozen reference failed conversion')
            choices[split][group]=sha
    return choices


class Crops(Dataset):
    def __init__(self,manifest,length=2052,seed=1731):
        self.path=Path(manifest);self.m=read_manifest(manifest)
        self.groups={}
        for r in self.m['rows']:
            if r['split']=='train' and self.m['audio'][r['audio_key']]['frames']>=length:
                self.groups.setdefault(r['group'],[]).append(r)
        self.keys=sorted(self.groups);self.length=length;self.seed=seed
        if not self.keys:raise ValueError('no train crops')
    def __len__(self):return 10000000
    def __getitem__(self,index):
        rng=random.Random(self.seed+index)
        group=self.keys[rng.randrange(len(self.keys))]
        row=rng.choice(self.groups[group]);a,x,l=load_pair(self.path.parent,self.m,row)
        start=rng.randrange(x.shape[-1]-self.length+1)
        a,x=a[:,start:start+self.length],x[:,start:start+self.length].clone()
        if rng.random()<.5:x[7]=1-x[7]
        if rng.random()<.5:x[8]=1-x[8]
        return a,x,l

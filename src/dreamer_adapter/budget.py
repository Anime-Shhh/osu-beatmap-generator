"""A serialized allocation ledger; every pilot GPU command requires one slot."""
import fcntl
import json
import os
from pathlib import Path
import time
from .contracts import write_json

HOURS={'baseline':4,'codec-ae':4,'codec-vae001':4,'codec-vae01':4,
       'generator-ae':6,'generator-vae':6,'confirmation':8,'diagnostics':4,'final':8}


class Budget:
    def __init__(self,path,slot):
        self.path=Path(path);self.slot=slot
        if slot not in HOURS:raise ValueError('unknown budget slot')
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.lock=self.path.with_suffix('.lock').open('a')
        fcntl.flock(self.lock,fcntl.LOCK_EX)
        self.ledger=json.loads(self.path.read_text()) if self.path.exists() else {'runs':[]}
        if any(r['status']=='running' for r in self.ledger['runs']):
            raise RuntimeError('another GPU command active or needs accounting reconciliation')
        used=sum(r['seconds'] for r in self.ledger['runs'] if r['slot']==self.slot)
        self.seconds=HOURS[self.slot]*3600-used
        if self.seconds<60:raise RuntimeError('stage budget exhausted')
        self.record={'slot':self.slot,'start':time.time(),'job':os.environ.get('SLURM_JOB_ID'),
                     'seconds':0,'status':'running'}
        self.ledger['runs'].append(self.record);write_json(self.path,self.ledger)
        fcntl.flock(self.lock,fcntl.LOCK_UN)
        return self
    def __exit__(self,kind,value,tb):
        fcntl.flock(self.lock,fcntl.LOCK_EX)
        self.record.update(seconds=time.time()-self.record['start'],status='failed' if kind else 'complete')
        write_json(self.path,self.ledger);fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close()

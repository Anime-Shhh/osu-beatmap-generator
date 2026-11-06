import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description='Pinned osu-dreamer pilot')
    s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('cleanup');q.add_argument('--output',required=True);q.add_argument('--pip-cache');q.add_argument('--candidate-ledger')
    q=s.add_parser('prepare');q.add_argument('--native-manifest',required=True);q.add_argument('--output',required=True);q.add_argument('--scan-limit',type=int,default=1500)
    q=s.add_parser('compact');q.add_argument('--source',required=True);q.add_argument('--output',required=True)
    for name in ('smoke','evaluate','train-codec','train-generator','generate','examples','diagnostics','decide'):
        q=s.add_parser(name);q.add_argument('--config',type=Path,required=True)
    a=p.parse_args()
    if a.command=='cleanup':
        from .cleanup import cleanup
        entries=json.loads(Path(a.candidate_ledger).read_text()) if a.candidate_ledger else {}
        print(json.dumps(cleanup(a.output,a.pip_cache,**entries)));return
    if a.command=='prepare':
        from .contracts import require_space
        from .prepare import prepare
        require_space()
        prepare(a.native_manifest,a.output,a.scan_limit);return
    if a.command=='compact':
        from .compact import compact
        compact(a.source,a.output);return
    config=json.loads(a.config.read_text())
    if a.command=='decide':
        dispatch(a.command,config);return
    import os
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU commands must run inside Slurm')
    from .budget import Budget
    import signal
    def expired(*_): raise TimeoutError('GPU allocation budget exhausted')
    with Budget(config.pop('budget_ledger'),config.pop('budget_slot')) as budget:
        signal.signal(signal.SIGALRM,expired);signal.alarm(int(budget.seconds))
        try:
            if a.command in ('train-codec','train-generator'):
                from .contracts import require_space
                preflight=Path(config.get('output')).parent/'training-space-preflight.json'
                from .contracts import write_json,digest,storage_inventory
                try:free=require_space(8)
                except RuntimeError:
                    write_json(preflight.parent/'storage-preflight-failed.json',storage_inventory())
                    raise
                checks=json.loads(preflight.read_text()) if preflight.exists() else {'manifest_sha256':digest(config['manifest']),'checks':[]}
                if checks['manifest_sha256']!=digest(config['manifest']):
                    raise ValueError('training preflight belongs to a different corpus')
                checks['checks'].append({'free_bytes':free,'command':a.command,'output':config['output']})
                write_json(preflight,checks)
            if a.command!='generate':config['seconds']=min(config.get('seconds',budget.seconds),budget.seconds-30)
            dispatch(a.command,config)
        finally:signal.alarm(0)


def dispatch(command,config):
    if command=='smoke':
        from .smoke import smoke
        smoke(**config)
    elif command=='evaluate':
        from .evaluate import evaluate
        evaluate(**config)
    elif command=='train-codec':
        from .train import train_codec
        train_codec(**config)
    elif command=='train-generator':
        from .train import train_generator
        train_generator(**config)
    elif command=='generate':
        from .generate import generate
        generate(**config)
    elif command=='examples':
        from .examples import examples
        examples(**config)
    elif command=='diagnostics':
        from .diagnostics import diagnostics
        diagnostics(**config)
    elif command=='decide':
        from .decisions import decide
        decide(**config)


if __name__=='__main__': main()

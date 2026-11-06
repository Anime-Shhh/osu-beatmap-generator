"""Sequential, fail-closed Slurm coordinator for the predeclared 48 GPU-hour pilot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from .contracts import write_json,digest
from .decisions import codec_gate,promotion_gate


def run(root,upstream,manifest,beat_checkpoint=None,beat_checkpoint_sha256=None,parallel_inputs=False):
    root=Path(root);configs=root/'configs';configs.mkdir(exist_ok=True)
    state={'phase':'starting','steps':[],'promoted':False};write_json(root/'pilot-status.json',state)
    common={'manifest':str(manifest),'upstream':str(upstream),'budget_ledger':str(root/'gpu-budget.json')}
    def launch(name,command,slot,**config):
        path=configs/(name+'.json')
        if path.exists():raise FileExistsError('Use a new coordinator run or explicitly reconcile completed stages')
        configuration={**common,**config,'budget_slot':slot}
        if command=='diagnostics':configuration.pop('upstream')
        write_json(path,configuration)
        state['phase']=name;write_json(root/'pilot-status.json',state)
        subprocess.run([sys.executable,'-u','-m','src.dreamer_adapter',command,'--config',str(path)],check=True)
        state['steps'].append(name);write_json(root/'pilot-status.json',state)
    def read(path):return json.loads(Path(path).read_text())
    def codec_report(name):
        checkpoint=root/name/'best.pt';st=torch.load(checkpoint,map_location='cpu',weights_only=True)
        return checkpoint,read(root/name/f"validation-{st['step']}"/'metrics.json')
    try:
        if not parallel_inputs:launch('baseline','evaluate','baseline',output=str(root/'baseline'))
        baseline=read(root/('baseline-retry' if parallel_inputs else 'baseline')/'metrics.json')
        if not baseline.get('complete'):raise ValueError('incomplete parallel baseline')
        for name,kind,beta in [('codec-ae','ae',0.),('codec-vae001','vae',.001),('codec-vae01','vae',.01)]:
            if parallel_inputs:
                if not (root/name/'complete.json').exists():raise ValueError('incomplete parallel codec')
            else:launch(name,'train-codec',name,output=str(root/name),kind=kind,beta=beta)
        ae_path,ae_report=codec_report('codec-ae');qualified=[];gates={}
        for name in ('codec-vae001','codec-vae01'):
            path,report=codec_report(name);gate=codec_gate(ae_report,report);gates[name]=gate
            if gate['passed']:qualified.append((gate['summary']['f1_10'],name,path))
        write_json(root/'codec-decision.json',{'gates':gates,'selected_vae':max(qualified)[1] if qualified else None})
        arms=[('generator-ae',ae_path)]
        if qualified:arms.append(('generator-vae',max(qualified)[2]))
        results={}
        for name,path in arms:
            launch(name,'train-generator',name,output=str(root/name),codec_path=str(path),cache=str(root/(name+'-latents')))
            launch(name+'-audit','evaluate','final',output=str(root/(name+'-audit')),codec_path=str(path),
                   generator_path=str(root/name/'best.pt'),modes=['generation'])
            report=read(root/(name+'-audit')/'metrics.json')
            results[name]={'codec':path,'checkpoint':root/name/'best.pt','report':report,'gate':promotion_gate(baseline,report)}
        histories={name:read(root/name/'history.json')['history'] for name,_ in arms}
        common_steps=set.intersection(*(set(r['step'] for r in h) for h in histories.values()))
        write_json(root/'common-update-comparison.json',{
            'common_completed_update':max(common_steps) if common_steps else None,
            'arms':{name:next((r for r in h if r['step']==max(common_steps)),None)
                    if common_steps else None for name,h in histories.items()},
            'note':'Aligned full-song validation at equal updates; selection still uses each arm’s best validated checkpoint.'})
        if 'generator-vae' in results:
            results['generator-vae']['vae_gate']=promotion_gate(results['generator-ae']['report'],results['generator-vae']['report'],vae=True)
            if not results['generator-vae']['vae_gate']['passed']:results['generator-vae']['gate']['passed']=False
        acceptable=[(v['gate']['gain']['gain'],name) for name,v in results.items() if v['gate']['passed']]
        selected=None
        if acceptable:
            name=max(acceptable)[1];candidate=results[name]
            launch('confirmation','train-generator','confirmation',output=str(root/'confirmation'),
                   codec_path=str(candidate['codec']),cache=str(root/(name+'-latents')),seed=2831,seconds=28800)
            launch('confirmation-audit','evaluate','final',output=str(root/'confirmation-audit'),
                   codec_path=str(candidate['codec']),generator_path=str(root/'confirmation/best.pt'),modes=['generation'])
            confirm=promotion_gate(baseline,read(root/'confirmation-audit/metrics.json'))
            state['confirmation_gate']=confirm
            # Confirmation establishes repeatability; keep the primary run as the preselected artifact.
            if confirm['passed']:selected=candidate;state['promoted']=True
        launch('diagnostics','diagnostics','diagnostics',output=str(root/'diagnostics'),
               beat_checkpoint=beat_checkpoint,beat_checkpoint_sha256=beat_checkpoint_sha256)
        selected_hash=digest(selected['checkpoint']) if selected else 'upstream'
        decision={'selected_generator':selected_hash,'selected_codec':digest(selected['codec']) if selected else None,
                  'promoted':bool(selected),'candidate_gates':{k:{x:y for x,y in v.items() if x in ('gate','vae_gate')} for k,v in results.items()}}
        write_json(root/'selection.json',decision)
        launch('locked-test','evaluate','final',output=str(root/'locked-test'),split='test',modes=['generation'],
               codec_path=str(selected['codec']) if selected else None,generator_path=str(selected['checkpoint']) if selected else None,
               test_authorization=str(root/'selection.json'))
        # Kaguya is a development example; no matching against its chart is used for selection.
        launch('playtest','generate','final',output=str(root/'playtest'),
               audio='/common/users/asj102/osu_project/flow_runs/run_20261025/kaguya-original-input/audio.mp3',stars=4.5,
               codec_path=str(selected['codec']) if selected else None,generator_path=str(selected['checkpoint']) if selected else None)
        launch('examples','examples','final',output=str(root/'examples'),
               codec_path=str(selected['codec']) if selected else None,generator_path=str(selected['checkpoint']) if selected else None)
        state.update(phase='complete',selection=decision);write_json(root/'pilot-status.json',state)
    except Exception as exc:
        state.update(phase='stopped',error=repr(exc));write_json(root/'pilot-status.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--upstream',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--beat-checkpoint');p.add_argument('--beat-checkpoint-sha256')
    p.add_argument('--parallel-inputs',action='store_true')
    run(**vars(p.parse_args()))

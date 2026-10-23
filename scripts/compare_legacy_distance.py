"""Run a matched evaluation after training is separately authorized.

No training/submission. Both checkpoints must record equal splits, update budgets,
initial weights and training settings except the auxiliary-distance weight.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import zipfile
import torch
from src.legacy_spacing_eval import audit_generated_map, distribution, GAP_BINS_MS
from src.legacy_distance import corpus_fingerprint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_matched_checkpoints(baseline, auxiliary, split_hash):
    if baseline.get('distance_config', {}).get('loss_weight') != 0:
        raise ValueError('baseline must use distance weight 0')
    if auxiliary.get('distance_config', {}).get('loss_weight', 0) <= 0:
        raise ValueError('auxiliary checkpoint must use a positive distance weight')
    for key in ('legacy_model_config', 'initial_checkpoint_sha256', 'split_manifest_sha256', 'optimizer_updates', 'data_fingerprint'):
        if key not in baseline or key not in auxiliary or baseline[key] != auxiliary[key]:
            raise ValueError(f'unmatched checkpoint provenance: {key}')
    if baseline['split_manifest_sha256'] != split_hash or baseline['optimizer_updates'] <= 0:
        raise ValueError('split hash mismatch or no recorded optimizer updates')
    if not baseline['data_fingerprint']:
        raise ValueError('controlled comparison requires fingerprinted cached data')
    if not baseline['initial_checkpoint_sha256']:
        raise ValueError('use the same --initialize_from checkpoint for both controlled runs')
    # Paths, logging, initialization mode and the treatment itself may differ.
    ignored = {'model_dir','log_dir','resume','initialize_from','distance_loss_weight',
               'use_wandb','wandb_project','log_every','data_dir','split_file'}
    left = {k:v for k,v in baseline['training_args'].items() if k not in ignored}
    right = {k:v for k,v in auxiliary['training_args'].items() if k not in ignored}
    if left != right:
        raise ValueError('training settings differ; use matched seeds/batches/updates/schedules')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--auxiliary',type=Path,required=True)
    p.add_argument('--split-file',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--songs',type=Path,required=True,help='JSON list: id, audio, bpm, difficulties; use held-out songs')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[1731,1732,1733])
    p.add_argument('--temperature',type=float,default=.9);p.add_argument('--max-tokens',type=int,default=512)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--max-batches',type=int,default=0)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if not math.isfinite(a.temperature) or a.temperature <= 0 or min(a.max_tokens,a.batch_size) <= 0 or a.max_batches < 0:
        p.error('invalid decoding or evaluation budgets')
    if len(set(a.seeds)) != len(a.seeds): p.error('generation seeds must be unique')
    checkpoints = {name:torch.load(path,map_location='cpu',weights_only=False)
                   for name,path in [('baseline',a.baseline),('auxiliary',a.auxiliary)]}
    verify_matched_checkpoints(checkpoints['baseline'],checkpoints['auxiliary'],sha(a.split_file))
    from src.dataset import OsuCachedDataset, build_dataloader, WINDOW_SAMPLE, TARGET_SR
    from src.splits import load_split_manifest, resolve_split_shards
    from src.eval import run_eval_legacy
    from src.inference import (_load_legacy_model,generate_legacy_objects,_load_audio,OSU_TEMPLATE,
                               star_to_difficulty_name,star_to_od_ar)
    from src.tokenizer import hitobjects_to_osu_lines
    splits = resolve_split_shards(str(a.data_dir),load_split_manifest(str(a.split_file)))
    if corpus_fingerprint(splits) != checkpoints['baseline']['data_fingerprint']:
        raise ValueError('current shard bytes differ from training data')
    shards = splits['test']
    shard_hashes = {str(path):sha(path) for path in shards}
    if not shards: raise ValueError('empty test split')
    songs = json.loads(a.songs.read_text())
    if not songs or len({s['id'] for s in songs}) != len(songs): raise ValueError('unique song IDs required')
    for s in songs:
        if Path(s['id']).name != s['id'] or s['id'] in ('.','..'): raise ValueError('song id must be a filename')
        if s.get('split') != 'test': raise ValueError('explicit held-out test song declarations required')
        if not math.isfinite(s['bpm']) or s['bpm'] <= 0 or not s['difficulties']:
            raise ValueError('finite positive BPM and difficulties required')
        if len(set(s['difficulties'])) != len(s['difficulties']) or any(not math.isfinite(d) or d <= 0 for d in s['difficulties']):
            raise ValueError('finite positive unique difficulties required')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    report = {'schema':'legacy-distance-comparison/1','settings':vars(a).copy(),
              'split_sha256':sha(a.split_file),'test_shard_sha256':shard_hashes,
              'songs_manifest_sha256':sha(a.songs), 'models':{},
              'playtest_observations':'Not performed; fill playtests.md after human review.',
              'limitations':['Song test membership is supplied by the manifest, not independently inferred.',
                             'Legacy shard splits can leak songs if preprocessing mixed song windows across shards.',
                             'No positional match to one human layout is used as a map-quality score.']}
    for label, checkpoint in checkpoints.items():
        model = _load_legacy_model(SimpleNamespace(initialize_distance_head=False),checkpoint,torch.device(a.device))
        if a.max_tokens >= model.decoder.pos_embed.num_embeddings:
            raise ValueError('max-tokens exceeds decoder positional capacity')
        dataset = OsuCachedDataset(shard_pattern=str(a.data_dir/'shard-*.tar'),shards=shards,
                                  shuffle=False,sample_type=WINDOW_SAMPLE,quality_filter=checkpoint['training_args']['quality_filter'])
        loader = build_dataloader(dataset,a.batch_size,num_workers=0,pin_memory=False)
        values = run_eval_legacy(model,loader,torch.device(a.device),a.max_batches)
        result = {'teacher_forced':{k:(sum(v)/len(v) if v else None) for k,v in values.items()},
                  'distance_head_supervised':label=='auxiliary','examples':[],
                  'checkpoint_sha256':sha(a.baseline if label=='baseline' else a.auxiliary)}
        # A zero-weight distance head is untrained, not a valid original-model score.
        if label == 'baseline':
            result['teacher_forced']['distance_mae_px'] = None
            result['teacher_forced']['distance_loss'] = None
        for song in songs:
            audio = Path(song['audio']).expanduser().resolve()
            waveform = _load_audio(str(audio));duration_ms=waveform.shape[-1]*1000/TARGET_SR
            for difficulty in song['difficulties']:
                for seed in a.seeds:
                    args = SimpleNamespace(difficulty=difficulty,seed=seed,temperature=a.temperature,
                                           max_tokens_per_window=a.max_tokens)
                    diagnostics=[]
                    objects = generate_legacy_objects(args,checkpoint,waveform,song['bpm'],torch.device(a.device),model=model,diagnostics=diagnostics)
                    od,ar = star_to_od_ar(difficulty)
                    text = OSU_TEMPLATE.format(audio_filename=audio.name,title=song['id'],
                        difficulty_name=star_to_difficulty_name(difficulty),od=od,ar=ar,
                        beat_length=f"{60000/song['bpm']:.12f}",hitobjects='\n'.join(hitobjects_to_osu_lines(objects)))
                    stem=f'{label}-{song["id"]}-{difficulty}-{seed}'
                    (a.output_dir/(stem+'.osu')).write_text(text)
                    with zipfile.ZipFile(a.output_dir/(stem+'.osz'),'w',compression=zipfile.ZIP_DEFLATED) as z:
                        z.writestr(stem+'.osu',text);z.write(audio,audio.name)
                    audit=audit_generated_map(text,duration_ms=duration_ms,target_star=difficulty)
                    audit['raw_window_diagnostics']={k:sum(d[k] for d in diagnostics) for k in
                        ('raw_head_tokens','raw_head_bounds_violations','invalid_head_position_tokens')}
                    audit['raw_window_diagnostics']['windows']=len(diagnostics)
                    result['examples'].append(dict(song=song['id'],seed=seed,audio_sha256=sha(audio),
                                                   archive=stem+'.osz',audit=audit))
        report['models'][label]=result
        del model
    # Keep per-song/seed pairs for uncertainty analysis; summarize by target difficulty.
    for result in report['models'].values():
        groups={}
        for ex in result['examples']:
            audit=ex['audit'];key=str(audit['target_star'])
            groups.setdefault(key,[]).extend(audit['pairs'])
        result['spacing_by_difficulty']={}
        for key,pairs in groups.items():
            bins={};lower=0
            for upper in GAP_BINS_MS:
                bins[f'({lower},{upper}]']=distribution([p['distance_px'] for p in pairs if lower<p['gap_ms']<=upper])
                lower=upper
            result['spacing_by_difficulty'][key]={'all':distribution([p['distance_px'] for p in pairs]),'by_gap_ms':bins}
    report['paired_changes']=[]
    for left,right in zip(report['models']['baseline']['examples'],report['models']['auxiliary']['examples']):
        old,new=left['audit'],right['audit']
        changes={k:new[k]-old[k] for k in ('objects','events_per_minute','head_playfield_violations',
                                          'invalid_sliders','slider_body_violations','nonpositive_head_gaps')}
        changes['raw_head_bounds_violations']=new['raw_window_diagnostics']['raw_head_bounds_violations']-old['raw_window_diagnostics']['raw_head_bounds_violations']
        for key,field in [('spacing','p95'),('head_speed','max'),('realized_difficulty','stars')]:
            x,y=old[key].get(field),new[key].get(field)
            changes[key+'_'+field]=y-x if x is not None and y is not None else None
        report['paired_changes'].append(dict(song=left['song'],seed=left['seed'],target_star=old['target_star'],
                                             auxiliary_minus_baseline=changes))
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2,default=str,allow_nan=False))
    (a.output_dir/'playtests.md').write_text('# Matched playtests — not yet performed\n\nRecord song, seed, target difficulty, model (prefer blind A/B), timing/spacing/slider observations, fatigue, and preference.\nDo not replace missing observations with inferred quality from auxiliary loss.\n')
    print(a.output_dir/'report.json')

if __name__ == '__main__': main()

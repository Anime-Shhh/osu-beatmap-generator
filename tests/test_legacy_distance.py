import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from src.tokenizer import (BOS,EOS,PAD,TYPE_CIRCLE,TYPE_SLIDER_START,TYPE_SLIDER_CONTROL,
    TYPE_SLIDER_END,TYPE_SPINNER_START,TYPE_SPINNER_END,TIME_OFFSET,Residuals,
    xy_to_bin,object_head_distance_targets,tokenize_beatmap,TOTAL_VOCAB)
from src.dataset import collate_fn
from src.model import BeatmapDecoder,OsuMapper
from src.legacy_distance import distance_loss,shifted_distance_loss,distance_config,TeacherForcedMetrics
from src.checkpoints import load_legacy_state_dict


def sequence(items):
    tokens=[BOS];res=[Residuals()]
    for kind,x,y in items:
        token,r=xy_to_bin(x,y)
        tokens.extend([TIME_OFFSET,kind,token]);res.extend([Residuals(),Residuals(),r])
    return tokens+[EOS],res+[Residuals()]


def sample(items):
    tokens,res=sequence(items)
    return dict(mel=torch.rand(1,129,8),tokens=tokens,residuals=res,difficulty_id=2,
                bpm=120.,star_rating=4.,signal=None)


def test_exact_reference_coordinates_and_diagonal():
    tokens,res=sequence([(TYPE_CIRCLE,2,3),(TYPE_CIRCLE,5,7)])
    d,m=object_head_distance_targets(tokens,res)
    assert [i for i,v in enumerate(m) if v]==[6]
    assert d[6]==pytest.approx(5/640)
    tokens,res=sequence([(TYPE_CIRCLE,0,0),(TYPE_CIRCLE,512,384)])
    assert object_head_distance_targets(tokens,res)[0][6]==1


def test_compact_retention_preserves_distinct_best_and_last(tmp_path):
    from src.train import CheckpointManager
    manager=CheckpointManager(str(tmp_path),top_k=1,compact=True)
    for epoch,loss in [(0,1.),(1,2.),(2,.5)]:
        manager.save({'epoch':epoch,'value':torch.tensor(epoch)},loss,save_last=True)
        assert torch.load(tmp_path/'last.pt',weights_only=True)['epoch']==epoch
        expected_best=0 if epoch<2 else 2
        assert torch.load(tmp_path/'best.pt',weights_only=True)['epoch']==expected_best
        assert len(list(tmp_path.glob('checkpoint_*.pt')))==1
        assert len({p.stat().st_ino for p in tmp_path.glob('*.pt')})==(2 if epoch==1 else 1)
        assert not list(tmp_path.glob('*.tmp'))


def test_controls_tail_spinner_and_boundaries_are_masked():
    tokens,res=sequence([(TYPE_CIRCLE,0,0),(TYPE_SLIDER_START,3,4),
        (TYPE_SLIDER_CONTROL,400,300),(TYPE_SLIDER_END,500,350),
        (TYPE_CIRCLE,6,8),(TYPE_SPINNER_START,256,192),
        (TYPE_SPINNER_END,256,192),(TYPE_CIRCLE,100,100),(TYPE_CIRCLE,103,104)])
    d,m=object_head_distance_targets(tokens+[PAD],res+[Residuals()])
    assert [i for i,v in enumerate(m) if v]==[6,15,27]
    assert all(d[i]==pytest.approx(5/640) for i in (6,15,27))
    # Separate sequence/window never borrows an unavailable preceding head.
    assert not object_head_distance_targets(*sequence([(TYPE_CIRCLE,103,104)]))[1][3]


def test_real_tokenizer_slider_does_not_change_distance_context():
    text='[HitObjects]\n0,0,0,1,0\n3,4,100,2,0,B|400:300|500:350,1,100\n6,8,200,1,0\n'
    encoded=tokenize_beatmap(text)
    d,m=object_head_distance_targets(encoded.tokens,encoded.residuals)
    assert [v for v,keep in zip(d,m) if keep]==pytest.approx([5/640,5/640])
    cropped=tokenize_beatmap(text,window_start_ms=100,window_end_ms=300)
    assert sum(object_head_distance_targets(cropped.tokens,cropped.residuals)[1])==1


def test_collation_before_clamp_padding_and_teacher_forcing_alignment():
    batch=collate_fn([sample([(TYPE_CIRCLE,10,10),(TYPE_CIRCLE,530,10)]),sample([(TYPE_CIRCLE,1,1)])])
    assert batch['x_residuals'][0,6]==8 # existing coordinate training behavior retained
    assert batch['distance_targets'][0,6]==pytest.approx(520/640)
    assert not batch['distance_mask'][1].any()
    prediction=batch['distance_targets'][:,1:,None].clone().requires_grad_()
    loss,mae,n=shifted_distance_loss({'distance_predictions':prediction},batch)
    assert loss==0 and mae==0 and n==1
    # Decoder index 5 consumes TYPE_CIRCLE and predicts token/target at index 6.
    assert batch['tokens'][0,:-1][5]==TYPE_CIRCLE and batch['distance_mask'][0,1:][5]
    wrong=batch['distance_targets'][:,:-1,None]
    assert shifted_distance_loss({'distance_predictions':wrong},batch)[0]>0


def test_masked_loss_normalization_and_no_valid_targets():
    pred=torch.tensor([[[.5],[99.],[float('nan')]]],requires_grad=True)
    target=torch.tensor([[0.,float('nan'),float('nan')]])
    mask=torch.tensor([[True,False,False]])
    loss,mae,n=distance_loss(pred,target,mask)
    assert loss.item()==pytest.approx(.125) and mae==320 and n==1
    loss.backward();assert pred.grad[0,0,0]!=0 and (pred.grad[0,1:]==0).all()
    empty=pred.detach().requires_grad_()
    zero,mae,n=distance_loss(empty,target,torch.zeros_like(mask))
    assert zero==0 and mae==0 and n==0
    zero.backward();assert (empty.grad==0).all()
    with pytest.raises(ValueError,match='align'):distance_loss(pred,target[:,:1],mask)


def test_causal_distance_head_has_no_target_or_future_access():
    torch.manual_seed(9)
    decoder=BeatmapDecoder(d_model=16,nhead=2,num_layers=1,dim_feedforward=32,dropout=0).eval()
    tokens,res=sequence([(TYPE_CIRCLE,0,0),(TYPE_CIRCLE,3,4)])
    inputs=torch.tensor([tokens[:-1]]);memory=torch.randn(1,4,16)
    original=decoder(memory,inputs,return_distance=True)
    modified=inputs.clone();modified[:,6:]=123
    changed=decoder(memory,modified,return_distance=True)
    # Position token at full index 6 is still absent from this prediction prefix.
    for a,b in zip(original,changed):torch.testing.assert_close(a[:,:6],b[:,:6])
    assert (original[-1]>=0).all() and len(decoder(memory,inputs))==4
    original[-1][:,5].sum().backward()
    assert decoder.distance_head[0].weight.grad.abs().sum()>0
    assert decoder.token_embed.weight.grad.abs().sum()>0


class TinyAudio(nn.Module):
    def __init__(self,pretrained='',d_model=16,unfreeze_last_n=2):
        super().__init__();self.proj=nn.Linear(129,d_model)
    def forward(self,mel):return self.proj(mel.squeeze(1).transpose(1,2))


@pytest.fixture
def mapper(monkeypatch):
    monkeypatch.setattr('src.model.AudioEncoder',TinyAudio)
    torch.manual_seed(41)
    return OsuMapper(d_model=16,decoder_layers=1,decoder_heads=2,decoder_ff=32,dropout=0,max_seq_len=64)


def test_checkpoint_migration_only_initializes_new_head(mapper):
    full=copy.deepcopy(mapper.state_dict())
    old={k:v for k,v in full.items() if not k.startswith('decoder.distance_head.')}
    with pytest.raises(RuntimeError,match='explicitly'):load_legacy_state_dict(mapper,old)
    compiled={'_orig_mod.'+k:v for k,v in old.items()}
    report=load_legacy_state_dict(mapper,compiled,initialize_distance_head=True)
    assert report['initialized_distance_head'] and report['normalized_compiled_keys']
    for k,v in old.items():torch.testing.assert_close(mapper.state_dict()[k],v)
    for bad in ({k:v for k,v in old.items() if k!='decoder.discrete_head.bias'},
                {**old,'surprise':torch.zeros(1)},
                {k:v for k,v in full.items() if k!='decoder.distance_head.0.bias'}):
        with pytest.raises(RuntimeError):load_legacy_state_dict(mapper,bad,initialize_distance_head=True)
    with pytest.raises(RuntimeError):
        load_legacy_state_dict(mapper,{**full,'decoder.x_res_head.weight':torch.zeros(2,2)})
    assert not load_legacy_state_dict(mapper,full)['initialized_distance_head']


def test_generation_ignores_distance_predictions_and_checkpoint_roundtrip(mapper,tmp_path):
    mapper.eval();mel=torch.randn(1,1,129,8);diff=torch.tensor([2]);bpm=torch.tensor([[120.]])
    torch.manual_seed(7);before=mapper.generate(mel,diff,bpm,max_len=7)
    with torch.no_grad():
        for parameter in mapper.decoder.distance_head.parameters():parameter.fill_(100.)
    torch.manual_seed(7);after=mapper.generate(mel,diff,bpm,max_len=7)
    assert before==after
    checkpoint={'model_state_dict':mapper.state_dict(),'legacy_model_config':mapper.model_config,
                'distance_config':distance_config(.1)}
    path=tmp_path/'new.pt';torch.save(checkpoint,path)
    restored=OsuMapper(**checkpoint['legacy_model_config'])
    load_legacy_state_dict(restored,torch.load(path,weights_only=True)['model_state_dict'])
    torch.manual_seed(7);assert restored.generate(mel,diff,bpm,max_len=7)==before


def test_training_adds_auxiliary_loss_and_logs_pixel_mae(mapper):
    from src.train import train_one_epoch_legacy
    from torch.cuda.amp import GradScaler
    batch=collate_fn([sample([(TYPE_CIRCLE,0,0),(TYPE_CIRCLE,3,4)])])
    args=SimpleNamespace(teacher_forcing_start=1.,teacher_forcing_end=1.,teacher_forcing_decay_epochs=1,
        difficulty=-1.,w_discrete=.8,w_residual=.2,distance_loss_weight=0.,accum_steps=1,
        max_grad_norm=1.,log_every=1,dry_run=False,max_updates=1)
    initial=copy.deepcopy(mapper.state_dict());reports=[]
    for weight in (0.,.1):
        mapper.load_state_dict(initial);args.distance_loss_weight=weight;args.optimizer_updates=0
        opt=torch.optim.SGD(mapper.parameters(),lr=0.)
        reports.append(train_one_epoch_legacy(mapper,[batch,batch],opt,GradScaler(enabled=False),0,args,torch.device('cpu')))
    assert reports[1]['train/loss']-reports[0]['train/loss']==pytest.approx(.1*reports[1]['train/distance_loss'],abs=2e-6)
    assert reports[1]['train/distance_target_count']==1
    assert reports[1]['train/optimizer_updates']==1
    assert reports[1]['train/distance_mae_px']>=0


def test_distance_metrics_are_weighted_by_valid_targets():
    metric=TeacherForcedMetrics()
    metric.add('distance_mae_px',10,1);metric.add('distance_mae_px',90,9)
    assert metric.result()['distance_mae_px']==10
    with pytest.raises(ValueError):distance_config(-1)
    with pytest.raises(ValueError):distance_config(float('nan'))


def test_existing_split_is_read_only(tmp_path):
    from src.train import prepare_cached_splits
    files=[]
    for split in ('train','val','test'):
        p=tmp_path/(split+'.tar');p.write_bytes(b'fixture');files.append(p.name)
    manifest=tmp_path/'splits.json';manifest.write_text(json.dumps(dict(zip(('train','val','test'),[[f] for f in files]))))
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    args=SimpleNamespace(split_file=str(manifest),data_dir=str(tmp_path),stream=False)
    paths,_=prepare_cached_splits(args)
    assert paths['train']==[str(tmp_path/'train.tar')]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==digest


def test_free_running_metrics_condition_on_gap_and_reset_spinner():
    from src.legacy_spacing_eval import audit_generated_map
    text='''osu file format v14
[General]
Mode:0
[Difficulty]
CircleSize:4
OverallDifficulty:5
ApproachRate:5
HPDrainRate:5
SliderMultiplier:1.4
SliderTickRate:1
[TimingPoints]
0,500,4,2,0,70,1,0
[HitObjects]
100,100,0,1,0
103,104,100,1,0
256,192,200,8,0,250
400,300,300,1,0
403,304,400,2,0,L|413:304,1,10
'''
    audit=audit_generated_map(text,duration_ms=1000,target_star=4.)
    assert len(audit['pairs'])==2 and audit['spacing']['mean']==5
    assert audit['distance_by_gap_ms']['(50,100]']['count']==2
    assert audit['events_per_minute']==300 and audit['invalid_sliders']==0
    assert audit['realized_difficulty']['available']


def test_comparison_rejects_unmatched_update_budgets():
    from scripts.compare_legacy_distance import verify_matched_checkpoints
    base=dict(distance_config=distance_config(0),legacy_model_config={},initial_checkpoint_sha256='initial',
        split_manifest_sha256='split',data_fingerprint='data',optimizer_updates=10,training_args=dict(seed=4,distance_loss_weight=0))
    aux={**base,'distance_config':distance_config(.1),'training_args':dict(seed=4,distance_loss_weight=.1)}
    verify_matched_checkpoints(base,aux,'split')
    with pytest.raises(ValueError,match='optimizer_updates'):
        verify_matched_checkpoints(base,{**aux,'optimizer_updates':11},'split')


def test_seeded_cached_order_and_corpus_fingerprint(tmp_path):
    import io
    import tarfile
    from src.dataset import OsuCachedDataset
    from src.legacy_distance import corpus_fingerprint
    path=tmp_path/'shard.tar'
    with tarfile.open(path,'w') as archive:
        for i in range(12):
            row=sample([(TYPE_CIRCLE,i,i)])
            for suffix,value in [('mel.pt',row['mel']),('tokens.pt',{'tokens':row['tokens'],'residuals':row['residuals']})]:
                buf=io.BytesIO();torch.save(value,buf);data=buf.getvalue()
                info=tarfile.TarInfo(f'{i:03}.{suffix}');info.size=len(data);archive.addfile(info,io.BytesIO(data))
    def keys(seed):
        return [r['key'] for r in OsuCachedDataset(str(path),shards=[str(path)],shuffle=True,shuffle_seed=seed)]
    assert keys(12)==keys(12) and keys(12)!=keys(13)
    digest=corpus_fingerprint({'train':[str(path)]})
    path.write_bytes(path.read_bytes()+b'changed')
    assert digest!=corpus_fingerprint({'train':[str(path)]})


def test_training_entrypoint_preserves_inputs_and_saves_configuration(mapper,tmp_path,monkeypatch):
    import sys
    import src.train as train
    initial=tmp_path/'old.pt'
    old={k:v for k,v in mapper.state_dict().items() if not k.startswith('decoder.distance_head.')}
    torch.save({'model_state_dict':old},initial)
    old_digest=hashlib.sha256(initial.read_bytes()).hexdigest()
    for name in ('train','val','test'):(tmp_path/(name+'.tar')).write_bytes(b'fixture')
    splits=tmp_path/'splits.json';splits.write_text(json.dumps({n:[n+'.tar'] for n in ('train','val','test')}))
    split_bytes=splits.read_bytes()
    batch=collate_fn([sample([(TYPE_CIRCLE,0,0),(TYPE_CIRCLE,30,40)])])
    monkeypatch.setattr(train,'build_legacy_model',lambda *a:mapper)
    monkeypatch.setattr(train,'build_dataset',lambda *a:None)
    monkeypatch.setattr(train,'build_dataloader',lambda *a,**kw:[batch])
    logged=[];monkeypatch.setattr(train,'build_logger',lambda args:logged.append)
    monkeypatch.setattr(sys,'argv',['train','--data_dir',str(tmp_path),'--split_file',str(splits),
        '--initialize_from',str(initial),'--model_dir',str(tmp_path/'run'),'--log_dir',str(tmp_path/'logs'),
        '--distance_loss_weight','0.1','--max_updates','1','--batch_size','1','--num_workers','0'])
    train.main()
    saved=torch.load(tmp_path/'run/last.pt',weights_only=True)
    assert saved['distance_config']==distance_config(.1) and saved['optimizer_updates']==1
    assert saved['legacy_model_config']==mapper.model_config
    assert saved['initial_checkpoint_sha256']==old_digest
    assert saved['data_fingerprint']
    assert hashlib.sha256(initial.read_bytes()).hexdigest()==old_digest and splits.read_bytes()==split_bytes
    assert any('val/distance_mae_px' in row for row in logged)
    with pytest.raises(FileExistsError):train.main()


def test_comparison_runner_writes_paired_full_song_artifacts(mapper,tmp_path,monkeypatch):
    import sys
    from scripts import compare_legacy_distance as compare
    from src.legacy_distance import corpus_fingerprint
    import src.dataset as dataset
    import src.inference as inference
    for split in ('train','val','test'):(tmp_path/(split+'.tar')).write_bytes(b'test fixture')
    splits={k:[str(tmp_path/(k+'.tar'))] for k in ('train','val','test')}
    split_file=tmp_path/'splits.json';split_file.write_text(json.dumps({k:[Path(p).name for p in v] for k,v in splits.items()}))
    settings=dict(seed=4,quality_filter='all',distance_loss_weight=0)
    common=dict(legacy_model_config=mapper.model_config,model_state_dict=mapper.state_dict(),
        initial_checkpoint_sha256='unit-test-fixture',split_manifest_sha256=compare.sha(split_file),
        data_fingerprint=corpus_fingerprint(splits),optimizer_updates=1)
    for label,weight in [('baseline',0.),('auxiliary',.1)]:
        torch.save({**common,'training_args':{**settings,'distance_loss_weight':weight},
                    'distance_config':distance_config(weight)},tmp_path/(label+'.pt'))
    audio=tmp_path/'audio.wav';audio.write_bytes(b'fixture; decoder mocked')
    songs=tmp_path/'songs.json';songs.write_text(json.dumps([dict(id='test',audio=str(audio),bpm=120.,difficulties=[4.],split='test')]))
    batch=collate_fn([sample([(TYPE_CIRCLE,100,100),(TYPE_CIRCLE,103,104)])])
    monkeypatch.setattr(dataset,'OsuCachedDataset',lambda **kw:None)
    monkeypatch.setattr(dataset,'build_dataloader',lambda *a,**kw:[batch])
    monkeypatch.setattr(inference,'_load_audio',lambda *a:torch.zeros(1,22050*9))
    monkeypatch.setattr(inference,'generate_legacy_objects',lambda *a,**kw:[
        dict(type='circle',x=100,y=100,time=100),dict(type='circle',x=103,y=104,time=200)])
    out=tmp_path/'comparison'
    monkeypatch.setattr(sys,'argv',['compare','--baseline',str(tmp_path/'baseline.pt'),'--auxiliary',str(tmp_path/'auxiliary.pt'),
        '--data-dir',str(tmp_path),'--split-file',str(split_file),'--songs',str(songs),'--output-dir',str(out),
        '--seeds','7','--max-tokens','12','--device','cpu'])
    compare.main()
    report=json.loads((out/'report.json').read_text())
    assert len(list(out.glob('*.osz')))==2
    assert report['models']['baseline']['teacher_forced']['distance_mae_px'] is None
    assert report['paired_changes'][0]['auxiliary_minus_baseline']['spacing_p95']==0
    assert 'Not performed' in report['playtest_observations']


def test_raw_head_audit_exposes_preclamp_violations():
    from src.legacy_spacing_eval import audit_raw_head_tokens
    tokens,res=sequence([(TYPE_CIRCLE,530,100),(TYPE_SLIDER_CONTROL,900,900),(TYPE_SLIDER_START,20,20)])
    result=audit_raw_head_tokens(tokens,res)
    assert result['raw_head_tokens']==2 and result['raw_head_bounds_violations']==1


def test_update_budget_does_not_count_amp_skipped_steps(mapper):
    from src.train import train_one_epoch_legacy
    class SkipFirstUpdate:
        def __init__(self):self.value=8.;self.calls=0
        def scale(self,loss):return loss
        def unscale_(self,opt):pass
        def get_scale(self):return self.value
        def step(self,opt):
            self.calls+=1
            if self.calls>1:opt.step()
        def update(self):
            if self.calls==1:self.value/=2
    batch=collate_fn([sample([(TYPE_CIRCLE,0,0),(TYPE_CIRCLE,3,4)])])
    args=SimpleNamespace(teacher_forcing_start=1.,teacher_forcing_end=1.,teacher_forcing_decay_epochs=1,
        difficulty=-1.,w_discrete=.8,w_residual=.2,distance_loss_weight=.1,accum_steps=1,
        max_grad_norm=1.,log_every=1,dry_run=False,max_updates=1)
    scaler=SkipFirstUpdate();opt=torch.optim.SGD(mapper.parameters(),lr=0.)
    result=train_one_epoch_legacy(mapper,[batch]*3,opt,scaler,0,args,torch.device('cpu'))
    assert scaler.calls==2 and result['train/optimizer_updates']==1
    assert result['train/distance_target_count']==2

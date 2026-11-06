import copy
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from src.dreamer_adapter.contracts import validate_sources,validate_binding,bind,digest,evaluation_groups
from src.dreamer_adapter.model import masked_kl,mask_latents,intervene,Codec
from src.dreamer_adapter.cleanup import cleanup
from src.dreamer_adapter.timing import replace_timing,clock_candidate
from src.dreamer_adapter.decisions import codec_gate


def row(i=0,split='train'):
    return {'sha256':str(i),'group':f'g{i}','split':split,'source':{'beatmap_status':'ranked',
           'audio_sha256':f'a{i}','canonical_audio_sha256':f'p{i}','beatmap_set_id':str(i)}}


def test_leakage_and_kaguya():
    a=row();b=row(1,'val');b['source']['audio_sha256']='a0'
    with pytest.raises(ValueError):validate_sources([a,b])
    b=row(1);b['source']['beatmap_set_id']='1147361'
    with pytest.raises(ValueError):validate_sources([b])
    rows=[row(i,'val') for i in range(24)]+[row(i+24,'test') for i in range(24)]
    assert evaluation_groups(rows)==evaluation_groups(list(reversed(rows)))


def test_contract_rejects_wrong_codec():
    expected=bind({'data':1},'a')
    with pytest.raises(ValueError):validate_binding(bind({'data':1},'b'),expected)
    for key,value in [('upstream','different'),('weights','different'),('features','old-mel'),
                      ('manifest','different'),('normalization',{'mean':[1.]})]:
        changed={**expected,key:value}
        with pytest.raises(ValueError):validate_binding(changed,expected)


def test_kl_and_mask():
    mu=torch.zeros(2,6,20);lv=torch.zeros_like(mu)
    assert masked_kl(mu,lv)==0
    mu[:,:,10:]=4
    mask=torch.zeros(2,20);mask[:,:10]=1
    assert masked_kl(mu,lv,mask)==0
    assert masked_kl(mu,lv)>0
    torch.manual_seed(4);z,masked=mask_latents(torch.ones(32,6,40))
    assert masked.any() and masked.sum(1).max()<10
    assert (z.masked_select(masked[:,None].expand_as(z))==0).all()


def test_audio_interventions_do_not_wrap():
    x=torch.arange(200.).repeat(72,1)
    y=intervene(x,'shift+500')
    assert (y[:,:84]==0).all() and torch.equal(y[:,84:],x[:,:116])
    y=intervene(x,'shift-500');assert (y[:,-84:]==0).all()
    assert intervene(x,'zero').count_nonzero()==0
    assert intervene(x,'wrong',torch.ones(72,50)).shape==x.shape


OSU='''osu file format v14
[General]
AudioFilename: audio.mp3
Mode: 0
[Metadata]
Title: test
Artist: test
Creator: test
Version: test
[Difficulty]
HPDrainRate:5
CircleSize:4
OverallDifficulty:5
ApproachRate:8
SliderMultiplier:1
SliderTickRate:1
[TimingPoints]
0,500,4,0,0,50,1,0
[HitObjects]
100,100,1000,2,0,L|200:100,2,100
100,100,2500,1,0,0:0:0:0:
'''


def test_timing_preserves_slider_duration():
    out=replace_timing(OSU,{'accepted':True,'beat_length_ms':430.,'offset_ms':23.})
    assert out.split('[HitObjects]')[1]==OSU.split('[HitObjects]')[1]
    from scripts.compare_native_rhythm import events_of
    from src.v2.mapspec import mapspec_from_osu
    before,after=map(lambda s:events_of(mapspec_from_osu(s)),[OSU,out])
    assert abs(before[0]['end_ms']-after[0]['end_ms'])<.001
    with pytest.raises(ValueError):replace_timing(OSU,{'accepted':False})
    assert clock_candidate(list(range(0,20000,500)),list(range(0,20000,2000)))['accepted']
    assert not clock_candidate(list(range(0,20000,500)),[])['accepted']


def test_cleanup_requires_metrics_and_preserves_protected(tmp_path):
    p=tmp_path/'candidate-1.osu';p.write_text('example')
    metrics=tmp_path/'metrics.json';metrics.write_text('{}')
    entry={'path':str(p),'sha256':digest(p),'evaluation_complete':True,'seed':1731,
           'source_revision':'abc','metrics':str(metrics),'metrics_sha256':digest(metrics)}
    with patch('src.dreamer_adapter.cleanup.quota_free_bytes',return_value=10**10):
        with pytest.raises(ValueError):cleanup(tmp_path/'blocked',candidates=[entry],protected=[p])
        assert p.exists()
        cleanup(tmp_path/'allowed',candidates=[entry])
    assert not p.exists() and metrics.exists()


def report(f1=.99,pairs=.98,shift=.7,cursor=1.,shuffled_cursor=2.):
    metrics={'onsets':{t:{'matches':99} for t in ('10','25','50')},'generated':{'objects':100},'reference':{'objects':100},
             'type_f1':{t:.99 for t in ('10','25','50')},'short_circle_pairs':{'reference':100,'both_as_circles_50ms':98},
             'star_error':.1,'invalid':False,'slider_duration_median_ms':5.}
    sh=copy.deepcopy(metrics)
    for t in sh['onsets']:sh['onsets'][t]['matches']=int(shift*100)
    r={'group':'a','mode':'reconstruction','condition':'aligned','seed':1,'metrics':metrics,
       'sampled_metrics':metrics,'shuffled_metrics':sh,'cursor_mse':cursor,'shuffled_cursor_mse':shuffled_cursor}
    summary={'f1_10':f1,'type_f1_10':.99,'pair_recall':pairs,'pair_reference_count':100,'slider_duration_median_ms':5}
    return {'complete':True,'rows':[r],'summaries':{'reconstruction/aligned':summary}}


def test_vae_gate_rejects_collapse_and_missing_pairs():
    ae=report();vae=report()
    assert codec_gate(ae,vae)['passed']
    vae=report(shift=.99,shuffled_cursor=1.)
    assert not codec_gate(ae,vae)['passed']
    vae=report(pairs=.5)
    assert not codec_gate(ae,vae)['passed']


def test_budget_counts_failures_and_refuses_active(tmp_path):
    from src.dreamer_adapter.budget import Budget
    with pytest.raises(ValueError):
        with Budget(tmp_path/'budget.json','codec-ae'):
            with pytest.raises(RuntimeError):
                with Budget(tmp_path/'budget.json','codec-ae'):pass
            raise ValueError('test failure')
    ledger=json.loads((tmp_path/'budget.json').read_text())
    assert ledger['runs'][0]['status']=='failed' and ledger['runs'][0]['seconds']>=0


def test_bad_slider_is_counted_invalid():
    from src.dreamer_adapter.metrics import native_metrics
    from src.v2.mapspec import mapspec_from_osu
    reference=mapspec_from_osu(OSU)
    bad=OSU.replace('L|200:100','L|900:100').replace(',2,100\n',',2,800\n')
    assert native_metrics(bad,reference)['invalid']


def test_valid_native_map_is_measurable():
    from src.dreamer_adapter.metrics import native_metrics
    from src.v2.mapspec import mapspec_from_osu
    measured=native_metrics(OSU,mapspec_from_osu(OSU))
    assert not measured['invalid'] and measured['stars'] is not None
    assert measured['onsets']['10']['f1']==1.
    assert measured['slider_duration_median_ms']==0.


def test_frozen_reference_is_not_substituted():
    from src.dreamer_adapter.data import selected_rows
    m={'groups':{'val':['a']},'reference_chart_hashes':{'val':{'a':'chosen'}},
       'rows':[{'group':'a','split':'val','sha256':'chosen','stars':4.6},
               {'group':'a','split':'val','sha256':'other','stars':4.5}]}
    assert selected_rows(m,'val')[0]['sha256']=='chosen'
    del m['reference_chart_hashes']
    with pytest.raises(ValueError):selected_rows(m,'val')


def test_audio_intervention_reaches_decoder_connections():
    from types import SimpleNamespace
    from src.dreamer_adapter.evaluate import sample
    observed={}
    class Latent:
        def audio_encoder(self,a):
            observed['encoder']=a.clone();return [a],a[:,:6,::27]
        def decode_logits(self,z,s,skips):
            observed['decoder']=skips[0].clone()
            return skips[0].new_zeros(1,9,skips[0].shape[-1])
        def label_predictor(self,s):return s.new_ones(1,5)
    model=SimpleNamespace(style=SimpleNamespace(sample=lambda labels:labels),
                          diffusion=SimpleNamespace(sample=lambda h,s,steps:h))
    codec=SimpleNamespace(latent=Latent())
    a=torch.ones(72,54)
    for condition in ('zero','shift+500','wrong'):
        changed=intervene(a,condition,torch.full((72,20),.25))
        sample(model,changed,torch.ones(1,5),codec=codec,
               generator_state={'normalization':{'mean':[0]*6,'std':[1]*6}})
        assert torch.equal(observed['encoder'][0],changed)
        assert torch.equal(observed['decoder'],observed['encoder'])


def test_lossless_compact_preserves_arrays_and_native(tmp_path):
    import numpy as np
    from src.dreamer_adapter.compact import compact
    from src.dreamer_adapter.contracts import write_json
    from src.dreamer_adapter import FEATURE_KEY,UPSTREAM_COMMIT
    source=tmp_path/'cache';source.mkdir();output=tmp_path/'compact'
    native=tmp_path/'native.osu';native.write_text(OSU)
    r=row();r.update(sha256=digest(native),native_path=str(native),map_path='chart.npz')
    rng=np.random.default_rng(1731);spec=rng.integers(0,255,(72,100),dtype=np.uint8)
    np.save(source/'audio.npy',spec)
    np.savez(source/'chart.npz',hit=spec[:7],xy=spec[:2].astype('uint16'),labels=np.array([4.5,8,6,4,6]))
    r['map_sha256']=digest(source/'chart.npz')
    write_json(source/'frozen-splits.json',{'rows':[r]})
    m={'features':FEATURE_KEY,'upstream':UPSTREAM_COMMIT,'rows':[r],
       'groups':{'val':[],'test':[]},'frozen_splits_sha256':digest(source/'frozen-splits.json'),
       'audio':{'a':{'path':'audio.npy','sha256':digest(source/'audio.npy')}}}
    write_json(source/'manifest.json',m)
    with patch('src.dreamer_adapter.compact.freeze_references',return_value={'val':{},'test':{}}):
        result=compact(source,output)
    with np.load(output/result['audio']['a']['path']) as f:assert np.array_equal(spec,f['spec'])
    assert digest(native)==r['sha256'] and (source/'chart.npz').exists()
    assert json.loads((output/'compression-audit.json').read_text())['verified_equal']


def test_validation_preserves_training_rng_even_on_failure():
    import random
    import numpy as np
    from src.dreamer_adapter.evaluate import preserve_rng
    torch.manual_seed(12);random.seed(12);np.random.seed(12)
    expected=torch.get_rng_state().clone();py=random.getstate();n=np.random.get_state()
    @preserve_rng
    def failing_evaluation():
        torch.manual_seed(999);random.seed(999);np.random.seed(999)
        raise ValueError('partial evaluation')
    with pytest.raises(ValueError):failing_evaluation()
    assert torch.equal(torch.get_rng_state(),expected) and random.getstate()==py
    assert np.array_equal(np.random.get_state()[1],n[1])

import torch
import pytest
from src.rhythm_flow.model import RhythmFlow,RhythmFlowConfig,flow_loss,sample_full_song,window_starts,decode_rhythm
from src.rhythm_flow.data import targets_for_times,validate_manifest,SCHEMA


def test_flow_gradients_and_audio_conditioning():
    torch.manual_seed(3)
    model=RhythmFlow(RhythmFlowConfig(width=16,layers=2,window_frames=32,overlap_frames=8))
    audio=torch.randn(2,101,32);target=torch.rand(2,5,32);profile=torch.rand(2,6)
    mask=torch.ones(2,32,dtype=torch.bool);mask[1,20:]=False
    opt=torch.optim.Adam(model.parameters(),lr=.01)
    for _ in range(2):
        opt.zero_grad();loss=flow_loss(model,audio,target,profile,mask);loss.backward();opt.step()
    assert torch.isfinite(loss)
    assert model.audio.weight.grad.abs().sum()>0
    x=torch.randn(2,5,32);t=torch.ones(2)*.5
    assert not torch.allclose(model(x,audio,profile,t),model(x,torch.zeros_like(audio),profile,t))
    assert (model(x,audio,profile,t,mask)[1,:,20:]==0).all()


def test_full_song_flow_covers_tail_and_is_seeded():
    model=RhythmFlow(RhythmFlowConfig(width=8,layers=1,window_frames=32,overlap_frames=8)).eval()
    audio=torch.randn(1,101,91);profile=torch.rand(1,6)
    a=sample_full_song(model,audio,profile,steps=2,generator=torch.Generator().manual_seed(3))
    b=sample_full_song(model,audio,profile,steps=2,generator=torch.Generator().manual_seed(3))
    assert a.shape==(1,5,91) and torch.isfinite(a).all();torch.testing.assert_close(a,b)
    for length in (1,31,32,33,90,9015):
        covered=set()
        for start in window_starts(length,32,8): covered.update(range(start,min(length,start+32)))
        assert len(covered)==length


def test_rhythm_targets_cross_window_holds_and_tail_peak():
    target=targets_for_times(torch.arange(100)*10.,[(100,400,1),(990,990,0)])
    assert target[3,20]==1 and target[1,20]==0
    events=decode_rhythm(target)
    assert events[-1]['start_ms']==990 and events[0]['kind']==1
    assert events[0]['end_ms']==390


def test_combining_corpora_cannot_leak_same_audio():
    row={'audio_key':'pcm','source_component':'source','audio_file':'pcm.pt','profile':[0]*6,'split':'train'}
    with pytest.raises(ValueError,match='leakage'):
        validate_manifest({'schema':SCHEMA,'frame_ms':10,'charts':[row,{**row,'split':'test'}]})


def test_expansion_preserves_old_split_and_links_new_encode():
    from scripts.extend_rhythm_flow_cache import preserve_splits
    from src.v2.data import SourceIdentity
    a=SourceIdentity(audio_sha256='a'*64,canonical_audio_sha256='b'*64)
    b=SourceIdentity(audio_sha256='c'*64,canonical_audio_sha256='b'*64)
    rows=[{'source':a.to_dict(),'split':'val'}, {'source':b.to_dict()}]
    preserve_splits(rows,1)
    assert rows[0]['split']==rows[1]['split']=='val'
    assert rows[0]['source_component']==rows[1]['source_component']
    # A new alternate encode cannot bridge old held-out and training sources.
    c=SourceIdentity(audio_sha256='d'*64,canonical_audio_sha256='e'*64)
    bridge=SourceIdentity(audio_sha256='d'*64,canonical_audio_sha256='b'*64)
    rows=[{'source':a.to_dict(),'split':'val'},{'source':c.to_dict(),'split':'train'},{'source':bridge.to_dict()}]
    with pytest.raises(ValueError,match='bridges'):preserve_splits(rows,2)


def test_corrupt_audio_is_audited_without_losing_base_cache(tmp_path,monkeypatch):
    import hashlib
    import json
    import sys
    from scripts import extend_rhythm_flow_cache as expansion
    from src.v2.data import SourceIdentity
    base=tmp_path/'base';base.mkdir();output=tmp_path/'expanded'
    source=SourceIdentity(audio_sha256='a'*64,canonical_audio_sha256='b'*64)
    audio_file=base/'audio.pt';torch.save({'audio':torch.zeros(10,101)},audio_file)
    row={'audio_key':'b'*64,'source_component':'old','source':source.to_dict(),
         'audio_file':'audio.pt','audio_file_sha256':hashlib.sha256(audio_file.read_bytes()).hexdigest(),
         'chart_sha256':'c'*64,'split':'train','profile':[0]*6,'events':[]}
    (base/'manifest.json').write_text(json.dumps({'schema':SCHEMA,'frame_ms':10,'charts':[row],
        'feature_pipeline_key':'test','audio_bytes':audio_file.stat().st_size,'audio_files':1}))
    monkeypatch.setattr(expansion,'hf_sources',lambda *args:iter([
        (b'corrupt',[{}],{'source_record_id':'bad-1'}),(b'corrupt',[{}],{'source_record_id':'bad-2'})]))
    def bad_decode(path):raise RuntimeError('invalid audio')
    monkeypatch.setattr(expansion,'load_audio',bad_decode)
    monkeypatch.setattr(sys,'argv',['expand','--base-cache',str(base),'--output-dir',str(output),'--hf-revision','d'*40])
    expansion.main()
    result=json.loads((output/'manifest.json').read_text())
    assert len(result['charts'])==1 and result['charts'][0]['split']=='train'
    assert len(result['expansion']['rejected'])==2
    assert result['expansion']['new_charts']==0
    assert (output/'audio.pt').read_bytes()==audio_file.read_bytes()

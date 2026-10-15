"""Audio alignment must affect training and cached generation without label leakage."""
from dataclasses import replace
import torch
import pytest
from src.v2.alignment import query_grid, chart_targets, alignment_loss
from src.v2.dataset import SparseSongDataset, collate_sparse_examples
from src.v2.preprocess import write_song_record
from src.v2.sparse_model import SparseEventPlanner
from src.v2.training import (TrainingConfig, TrainingStage, loss_for_batch, forward_planner,
                            initialize_planner, checkpoint_payload, CheckpointProvenance, save_checkpoint)
import test_v2_records as records_fixture
import test_v2_training as model_fixture


def batch_and_model(tmp_path):
    record = records_fixture.SongRecordTests()._make_record_with_bars(2)
    write_song_record(record, tmp_path/'record.pt')
    ds = SparseSongDataset(tmp_path, sampling_mode='bar', alignment_frames=32)
    batch = collate_sparse_examples([ds[0], ds[1]])
    batch['rhythm_features'] = torch.randn_like(batch['rhythm_features'][...,:4])
    batch['semantic_features'] = torch.randn_like(batch['semantic_features'][...,:3])
    old, geometry = model_fixture.TrainingCheckpointTests()._models()
    planner = SparseEventPlanner(replace(old.config, local_audio_tokens=32)).eval()
    return batch, planner, old, geometry


def test_alignment_has_absolute_clock_and_cross_bar_slider_occupancy():
    from src.v2.timing import timing_map_from_points, TimingPoint
    timing = timing_map_from_points([TimingPoint(0,500), TimingPoint(1000,250)])
    queries = query_grid(timing,0,4,8)
    torch.testing.assert_close(queries, torch.tensor([0.,250,500,750,1000,1125,1250,1375]))
    targets = chart_targets(torch.tensor([1000.,1025,1050,1075]), [(900,1060,1),(1025,1025,0)])
    assert targets[:,3].tolist()==[1,1,1,0]
    assert targets[1,0]==1 and targets[3,0]==0
    assert targets[:,1].sum()==0  # A carried slider is active, not a new onset.


def test_audio_path_gets_gradients_and_targets_are_not_inputs(tmp_path):
    batch, planner, _, _ = batch_and_model(tmp_path)
    out = forward_planner(planner,batch)
    changed=dict(batch);changed['alignment_targets']=1-batch['alignment_targets']
    torch.testing.assert_close(out['logits'],forward_planner(planner,changed)['logits'])
    shifted=dict(batch);shifted['rhythm_features']=batch['rhythm_features'].roll(11,1)
    assert not torch.allclose(out['logits'],forward_planner(planner,shifted)['logits'])
    losses=loss_for_batch(planner,None,batch,TrainingConfig(stage=TrainingStage.PLANNER,
        profile_dropout=0,boundary_context_dropout=0,mixed_precision='none'))
    losses['loss'].backward()
    for module in (planner.local_audio_projection,planner.alignment_head,planner.audio_encoder.rhythm_in):
        assert module.weight.grad.abs().sum()>0
        assert torch.isfinite(module.weight.grad).all()
    assert torch.isfinite(losses['alignment_loss'])
    with pytest.raises(ValueError,match='timestamped'):
        broken=dict(batch);del broken['local_query_times_ms'];forward_planner(planner,broken)


def test_cached_decode_uses_same_local_audio_as_training(tmp_path):
    batch,planner,_,_=batch_and_model(tmp_path)
    row={k:v[:1] if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    output=forward_planner(planner,row)
    memory=output['bar_memory']; mask=torch.cat((row['bar_mask'],output['alignment_mask']),1)
    cache=None; hidden=[]
    for i in range(row['decoder_tokens'].shape[1]):
        item,cache=planner.decode_step(row['decoder_tokens'][:,i:i+1],memory,position=i,
            target_bar_index=row['target_bar_index'], memory_mask=mask,
            previous_bar_tokens=row['previous_bar_tokens'],previous_bar_mask=row['previous_bar_mask'],cache=cache)
        hidden.append(item)
    torch.testing.assert_close(torch.cat(hidden,1),output['decoder_hidden'],atol=2e-6,rtol=2e-5)



def test_masked_out_of_audio_frames_and_empty_target_loss(tmp_path):
    batch,planner,_,_=batch_and_model(tmp_path)
    batch['local_query_times_ms']+=100000
    out=forward_planner(planner,batch)
    assert not out['alignment_mask'].any()
    loss=alignment_loss(out['alignment_logits'],batch['alignment_targets'],out['alignment_mask'])
    assert loss==0 and torch.isfinite(out['logits']).all()


def test_legacy_transfer_is_explicit_and_preserves_existing_weights(tmp_path):
    _,new,old,geometry=batch_and_model(tmp_path)
    provenance=CheckpointProvenance(feature_pipeline_key='a'*64,split_manifest_sha256='b'*64)
    payload=checkpoint_payload(planner=old,geometry=geometry,optimizer=None,scaler=None,epoch=2,global_step=3,
        training_config=TrainingConfig(stage=TrainingStage.PLANNER),provenance=provenance)
    path=save_checkpoint(payload,tmp_path/'old.pt')
    result=initialize_planner(path,new,expected_provenance=provenance)
    assert result['new_parameters'] and all(torch.equal(new.state_dict()[k],v) for k,v in old.state_dict().items())
    with pytest.raises(ValueError,match='only legacy'):
        initialize_planner(path,old,expected_provenance=provenance)
    with pytest.raises(ValueError,match='feature_pipeline_key'):
        initialize_planner(path,new,expected_provenance=replace(provenance,feature_pipeline_key='c'*64))


def test_existing_record_quantization_at_bar_end_is_not_dropped():
    from src.v2.alignment import reference_events
    from src.v2.grammar import BarEventCodec
    from src.v2.events import BeatEvent, EventKind
    record=records_fixture.SongRecordTests()._make_record_with_bars(2)
    tokens=BarEventCodec().encode([BeatEvent(EventKind.CIRCLE,2.-1e-8,100,100)],bar_start_beat=0,meter=2)
    record['bar_targets']=[{'tokens':torch.tensor(tokens),'bar_start_beat':0.,'bar_length_beats':2.}]
    assert reference_events(record)==[(1000.,1000.,0)]


def test_new_audio_path_is_used_in_full_song_sampling(tmp_path, monkeypatch):
    from src.v2.inference import extract_audio_features, _sample_candidate_map
    from src.v2.generation import GenerationRequest
    from src.v2.grammar import BarEventCodec
    from src.v2.events import BeatEvent, EventKind
    _,planner,_,geometry=batch_and_model(tmp_path)
    features,clocks=extract_audio_features(torch.zeros(1,48000),rhythm_dim=4,semantic_dim=3,bpm=120,offset_ms=0)
    codec=BarEventCodec(planner.vocabulary)
    calls=[]
    def generate(memory,**kwargs):
        calls.append(memory.shape[1])
        assert memory.shape[1]==1+planner.config.local_audio_tokens
        return codec.encode([BeatEvent(EventKind.CIRCLE,.5,100,100)],bar_start_beat=0,meter=4)
    monkeypatch.setattr(planner,'generate_bar',generate)
    request=GenerationRequest(audio_filename='fixture.wav',target_star=4.5,candidate_count=1,
                              feature_revision=features.feature_cache.pipeline_revision,model_revision='test')
    result=_sample_candidate_map(planner=planner,geometry=geometry.eval(),features=features,
        hypothesis=clocks[0],request=request,seed=1,temperature=1.,geometry_steps=3)
    assert calls and len(result.events)==1


def test_audio_intervention_respects_padding_and_history(tmp_path):
    from scripts.audit_v2_audio_dependence import intervene
    batch,_,_,_=batch_and_model(tmp_path)
    batch['rhythm_mask'][1,-3:]=False
    batch['rhythm_features'][1,-3:]=123
    shifted=intervene(batch,'shift_137')
    assert torch.equal(shifted['rhythm_features'][1,-3:],batch['rhythm_features'][1,-3:])
    assert torch.equal(shifted['labels'],batch['labels'])
    assert torch.equal(shifted['rhythm_times_ms'],batch['rhythm_times_ms'])
    assert intervene(batch,'no_history')['previous_bar_tokens'].shape[1]==0


def test_resume_cannot_silently_change_local_grid(tmp_path):
    from src.v2.training import load_checkpoint
    _,planner,_,geometry=batch_and_model(tmp_path)
    provenance=CheckpointProvenance(feature_pipeline_key='a'*64,split_manifest_sha256='b'*64)
    payload=checkpoint_payload(planner=planner,geometry=geometry,optimizer=None,scaler=None,epoch=1,global_step=1,
        training_config=TrainingConfig(stage=TrainingStage.PLANNER),provenance=provenance)
    path=save_checkpoint(payload,tmp_path/'aligned.pt')
    incompatible=SparseEventPlanner(replace(planner.config,local_audio_tokens=64))
    with pytest.raises(ValueError,match='local_audio_tokens'):
        load_checkpoint(path,planner=incompatible,geometry=geometry)


def test_analysis_detects_attack_timing_without_instrument_claims():
    import numpy as np
    from scripts.analyze_v2_audio_alignment import acoustic_lanes,peak_times,summarize_alignment
    waveform=torch.zeros(1,48000);waveform[0,12000:12020]=1
    times,spectrum,lanes=acoustic_lanes(waveform,24000)
    peaks=peak_times(times,lanes['mixture attacks'])
    result=summarize_alignment(np.array([.5]),peaks)
    assert result['within_50ms']==1
    assert 'drums' not in lanes and 'vocals' not in lanes


def test_stem_inspection_rejects_duration_mismatch(tmp_path):
    import wave
    from scripts.analyze_v2_audio_alignment import main
    for name,length in (('mix.wav',24000),('short.wav',12000)):
        with wave.open(str(tmp_path/name),'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(24000);f.writeframes(b'\0\0'*length)
    with pytest.raises(ValueError,match='duration/sample rate'):
        main(['--audio',str(tmp_path/'mix.wav'),'--stems','drums='+str(tmp_path/'short.wav'),
              '--output-dir',str(tmp_path/'report')])

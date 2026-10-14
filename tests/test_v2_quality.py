"""Regression tests for the failure modes observed in the October V2 run."""
from dataclasses import replace
import math
from unittest.mock import patch

import pytest
import torch

from src.v2.events import EventKind
from src.v2.geometry import (ActiveEventGeometryDiffuser, GeometryDiffusionConfig,
                             geometry_pattern_losses, geometry_diffusion_loss,
                             sample_geometry, sample_geometry_windows)
from src.v2.evaluation import (CallbackDifficultyCalculator, DifficultyCalculation, Candidate,
                              StrictAuditPolicy, audit_candidate, rerank_candidates)
from src.v2.inference import load_v2_models
from src.v2.training import (TrainingConfig, TrainingStage, CheckpointProvenance,
                             checkpoint_payload, save_checkpoint)
from test_v2_training import TrainingCheckpointTests
from test_v2_generation import _map_spec
from src.v2.events import BeatEvent
from src.v2.grammar import MAX_ANCHORS


def test_planner_weights_are_never_treated_as_trained_geometry(tmp_path):
    planner, geometry = TrainingCheckpointTests()._models()
    payload = checkpoint_payload(planner=planner, geometry=geometry, optimizer=None, scaler=None,
        epoch=9, global_step=10, training_config=TrainingConfig(stage=TrainingStage.PLANNER),
        provenance=CheckpointProvenance(feature_pipeline_key='a'*64, split_manifest_sha256='b'*64))
    path = save_checkpoint(payload, tmp_path / 'planner.pt')
    with pytest.raises(ValueError, match='trained geometry-stage'):
        load_v2_models(path)
    payload['training_config']['stage'] = 'geometry'
    payload['data_audit'] = {'production_ready': False}
    save_checkpoint(payload, path, overwrite=True)
    with pytest.raises(ValueError, match='diagnostic/unverified'):
        load_v2_models(path)


def pattern():
    points = torch.zeros(1, 3, MAX_ANCHORS + 1, 2)
    points[0, :, 0] = torch.tensor([[.2,.3],[.4,.3],[.4,.6]])
    points[0, 1, 1:3] = torch.tensor([[.5,.5],[.6,.3]])
    kinds = torch.tensor([[int(EventKind.CIRCLE),int(EventKind.SLIDER),int(EventKind.CIRCLE)]])
    counts = torch.tensor([[0,2,0]])
    mask = torch.ones(1,3,dtype=torch.bool)
    return points, kinds, counts, mask


def test_relative_losses_preserve_mirrored_translated_patterns_and_detect_flat_sliders():
    target, kinds, counts, mask = pattern()
    reflected = target.clone()
    reflected[...,0] = .9 - reflected[...,0]
    reflected[...,1] += .1
    losses = geometry_pattern_losses(reflected, target, kinds, counts, mask)
    assert all(value < 1e-10 for value in losses.values())
    wrong = target.clone()
    wrong[0,2,0] = torch.tensor([.9,.9])
    wrong[0,1,1] = torch.tensor([.5,.3])
    losses = geometry_pattern_losses(wrong, target, kinds, counts, mask)
    assert losses['spacing_loss'] > 0
    assert losses['turn_loss'] > 0
    assert losses['slider_shape_loss'] > 0


def test_pattern_losses_ignore_padding_and_spinner_and_have_finite_gradients():
    target, kinds, counts, mask = pattern()
    mask[:,2] = False
    pred = target.clone(); pred[:,2] = 100
    pred.requires_grad_()
    losses = geometry_pattern_losses(pred, target, kinds, counts, mask)
    sum(losses.values()).backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[:,2].abs().sum() == 0
    kinds[:] = int(EventKind.SPINNER)
    assert all(v == 0 for v in geometry_pattern_losses(pred, target, kinds, counts, mask).values())


def test_geometry_diffusion_backward_and_wrapped_model():
    target, kinds, counts, mask = pattern()
    model = ActiveEventGeometryDiffuser(GeometryDiffusionConfig(context_dim=16,hidden_dim=16,heads=2,layers=1,dropout=0))
    # DDP-like wrapper deliberately has no config attribute.
    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.module=model
        def forward(self,*args,**kwargs): return self.module(*args,**kwargs)
    losses = geometry_diffusion_loss(Wrapped(), torch.zeros(1,3,16), kinds, counts, target, mask,
                                    noise_level=torch.tensor([.3]), return_components=True)
    losses['geometry_loss'].backward()
    assert all(torch.isfinite(v) for v in losses.values())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_overlap_is_preserved_and_conditions_next_window():
    model = ActiveEventGeometryDiffuser(GeometryDiffusionConfig(context_dim=16,hidden_dim=16,heads=2,layers=1,dropout=0)).eval()
    context = torch.zeros(1,7,16); kinds = torch.zeros(1,7,dtype=torch.long); counts = kinds.clone()
    samples=[]
    def spy(*args,**kwargs):
        result=sample_geometry(*args,**kwargs)
        samples.append((kwargs.get('known_geometry'), result.clone()))
        return result
    with patch('src.v2.geometry.sample_geometry', side_effect=spy):
        result=sample_geometry_windows(model,context,kinds,counts,window_events=4,overlap_events=2,steps=3)
    assert len(samples)==3
    torch.testing.assert_close(result[:,:4],samples[0][1])
    torch.testing.assert_close(samples[1][1][:,:2],samples[0][1][:,2:])
    assert torch.isfinite(result).all()


def test_target_star_gate_and_near_simultaneous_onsets():
    events=(BeatEvent(kind=EventKind.CIRCLE,beat_position=0,x=100,y=100),
            BeatEvent(kind=EventKind.CIRCLE,beat_position=1,x=200,y=100))
    candidate=Candidate('c',_map_spec(events))
    calculator=CallbackDifficultyCalculator(lambda _: DifficultyCalculation(True,stars=7.75,calculator_id='test'))
    audit=audit_candidate(candidate,target_star=4.5,calculator=calculator,policy=StrictAuditPolicy(max_star_error=.5),compiler=lambda _: 'fixture')
    assert not audit.valid and any('realized difficulty' in e for e in audit.errors)
    unavailable=audit_candidate(candidate,target_star=4.5,policy=StrictAuditPolicy(max_star_error=.5),compiler=lambda _: 'fixture')
    assert not unavailable.valid
    close=(events[0],replace(events[1],beat_position=1/96))
    audit=audit_candidate(Candidate('fast',_map_spec(close)),policy=StrictAuditPolicy(minimum_start_gap_ms=10))
    assert not audit.valid


def test_distant_candidate_ranking_does_not_saturate():
    events=(BeatEvent(kind=EventKind.CIRCLE,beat_position=0,x=100,y=100),)
    candidates=[Candidate('a',_map_spec(events)),Candidate('b',_map_spec((replace(events[0],x=200),)))]
    calc=CallbackDifficultyCalculator(lambda text: DifficultyCalculation(True,stars=15 if text == '100' else 8,calculator_id='test'))
    ranked=rerank_candidates(candidates,target_star=4.5,calculator=calc,compiler=lambda spec: str(spec.events[0].x))
    assert ranked[0].candidate.candidate_id=='b'


def test_relative_anchor_encoding_preserves_controls_outside_canvas():
    from src.v2.geometry import encode_geometry, decode_geometry
    target, kinds, counts, mask = pattern()
    target[0,1,1] = torch.tensor([1.03,-.02])
    encoded = encode_geometry(target, 'head-logit-relative-anchors-v1')
    decoded = decode_geometry(encoded, 'head-logit-relative-anchors-v1')
    torch.testing.assert_close(decoded[:,:,0],target[:,:,0])
    torch.testing.assert_close(decoded[:,:,1:],target[:,:,1:],atol=1e-6,rtol=1e-5)


def test_real_materializer_preserves_curves_and_does_not_call_local_maps_ranked(tmp_path):
    import wave
    import zipfile
    import json
    import shutil
    from scripts.materialize_v2_data import osz_sources, materialize
    from src.v2.corpus import audit_training_corpus
    from src.v2.grammar import BarEventCodec
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg required for real audio materialization')
    wav=tmp_path/'audio.wav'
    with wave.open(str(wav),'wb') as audio:
        audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(24000)
        audio.writeframes(b'\x00\x00'*48000)
    text='''osu file format v14
[General]
AudioFilename: audio.wav
Mode: 0
[Metadata]
Title:unit test fixture
Artist:unit test fixture
Creator:unit test fixture
Version:unit test fixture
BeatmapID:1
BeatmapSetID:1
[Difficulty]
HPDrainRate:5
CircleSize:4
OverallDifficulty:5
ApproachRate:7
SliderMultiplier:1.4
SliderTickRate:1
[TimingPoints]
0,500,4,2,0,70,1,0
[HitObjects]
400,100,500,2,0,P|520:120|420:140,1,140
200,200,1500,1,0,0:0:0:0:
'''
    with zipfile.ZipFile(tmp_path/'fixture.osz','w') as archive:
        archive.writestr('map.osu',text);archive.write(wav,'audio.wav')
    report=materialize(osz_sources(tmp_path),tmp_path/'records')
    assert report['slider_curves']=={'P':1}
    assert not report['rejected']
    path=next((tmp_path/'records').glob('*.pt'))
    record=torch.load(path,weights_only=True)
    assert record['metadata']['source']['beatmap_status']=='unverified-local'
    assert any((bar['geometry'][:,:,0]>1).any() for bar in record['bar_targets'])
    manifest=json.loads((tmp_path/'records/splits.json').read_text())
    with pytest.raises(ValueError,match='smoke-sized'):
        audit_training_corpus(tmp_path/'records',manifest)
    assert not audit_training_corpus(tmp_path/'records',manifest,allow_diagnostic_data=True)['production_ready']
    path.write_bytes(path.read_bytes()+b'tampered')
    with pytest.raises(ValueError,match='contents differ'):
        audit_training_corpus(tmp_path/'records',manifest,allow_diagnostic_data=True)


def test_song_balancing_can_cover_multiple_bars_without_changing_split(tmp_path):
    from test_v2_records import SongRecordTests
    from src.v2.preprocess import write_song_record
    from src.v2.dataset import SparseSongDataset
    record=SongRecordTests()._make_record_with_bars(20)
    write_song_record(record,tmp_path/'record.pt')
    dataset=SparseSongDataset(tmp_path,split='train',bars_per_song=16)
    assert len(dataset)==16
    assert len({int(dataset[i]['target']['bar_index']) for i in range(16)})==16
    assert all(dataset[i]['metadata']['split']=='train' for i in range(16))


def test_empty_geometry_batch_runs_forward_and_backpropagates_zero():
    from src.v2.training import loss_for_batch
    model = ActiveEventGeometryDiffuser(GeometryDiffusionConfig(
        context_dim=16, hidden_dim=16, heads=2, layers=1, dropout=0))
    batch = {
        'labels': torch.zeros(1, 1, dtype=torch.long),
        'event_mask': torch.zeros(1, 0, dtype=torch.bool),
        'event_token_indices': torch.zeros(1, 0, dtype=torch.long),
        'event_kinds': torch.zeros(1, 0, dtype=torch.long),
        'anchor_counts': torch.zeros(1, 0, dtype=torch.long),
        'event_positions': torch.zeros(1, 0, dtype=torch.long),
        'geometry': torch.zeros(1, 0, MAX_ANCHORS + 1, 2),
        'event_features': torch.zeros(1, 0, 4),
    }
    output = {'logits': torch.zeros(1, 1, 2), 'decoder_hidden': torch.zeros(1, 1, 16)}
    with patch('src.v2.training.forward_planner', return_value=output):
        losses = loss_for_batch(torch.nn.Identity(), model, batch,
                               TrainingConfig(stage=TrainingStage.GEOMETRY))
    assert losses['loss'].item() == 0
    losses['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() == 0
               for p in model.parameters())


def test_perfect_slider_topology_is_constrained_by_grammar():
    from src.v2.grammar import EventVocabulary, EventGrammar, GrammarState, DecodePhase
    vocabulary=EventVocabulary()
    grammar=EventGrammar(vocabulary)
    state=GrammarState(phase=DecodePhase.SLIDER_ANCHORS,current_curve='P')
    assert grammar.allowed_ids(state)==(vocabulary.id_for('anchors:2'),)
    with pytest.raises(ValueError):
        grammar.transition(state,vocabulary.id_for('anchors:1'))


def test_geometry_epoch_freezes_planner_and_partial_accumulation_is_normalized(tmp_path):
    from test_v2_records import SongRecordTests
    from src.v2.dataset import SparseSongDataset,collate_sparse_examples
    from src.v2.preprocess import write_song_record
    from src.v2.training import run_epoch
    record=SongRecordTests()._make_record_with_bars(2)
    write_song_record(record,tmp_path/'record.pt')
    dataset=SparseSongDataset(tmp_path,sampling_mode='bar')
    batch=collate_sparse_examples([dataset[0],dataset[1]])
    batch['rhythm_features']=batch['rhythm_features'][...,:4]
    batch['semantic_features']=batch['semantic_features'][...,:3]
    planner,geometry=TrainingCheckpointTests()._models()
    other_planner,other_geometry=TrainingCheckpointTests()._models()
    other_planner.load_state_dict(planner.state_dict());other_geometry.load_state_dict(geometry.state_dict())
    before={k:v.clone() for k,v in planner.state_dict().items()}
    before_geometry={k:v.clone() for k,v in geometry.state_dict().items()}
    config=TrainingConfig(stage=TrainingStage.GEOMETRY,mixed_precision='none',profile_dropout=0,
        boundary_context_dropout=0,symbolic_context_dropout=0,geometry_flip_augmentation=False,max_gradient_norm=10000)
    for model_p,model_g,accumulation in ((planner,geometry,1),(other_planner,other_geometry,4)):
        torch.manual_seed(45)
        metrics=run_epoch(planner=model_p,geometry=model_g,loader=[batch],
            optimizer=torch.optim.SGD(model_g.parameters(),lr=.001),device='cpu',
            config=replace(config,gradient_accumulation=accumulation))
        assert math.isfinite(metrics['geometry_loss'])
    assert all(torch.equal(before[k],v) for k,v in planner.state_dict().items())
    assert any(not torch.equal(before_geometry[k],v) for k,v in geometry.state_dict().items())
    for a,b in zip(geometry.parameters(),other_geometry.parameters()):
        torch.testing.assert_close(a,b)
    values=[]
    for drop in (0.0,0.9):
        torch.manual_seed(46)
        values.append(run_epoch(planner=planner,geometry=geometry,loader=[batch],optimizer=None,device='cpu',
            config=replace(config,profile_dropout=drop,boundary_context_dropout=drop,symbolic_context_dropout=drop)))
    assert values[0]==values[1]

from dataclasses import replace
import numpy as np
import pytest
import torch
from src.v2.coverage import audit_coverage
from src.v2.evaluation import StrictAuditPolicy
from src.v2.slider_path import slider_path, fit_event_to_playfield, event_path, circle_radius
from src.v2.events import BeatEvent, EventKind


def test_short_chart_cannot_pass_full_song_gate():
    policy=StrictAuditPolicy(audio_duration_ms=90000)
    short=[(t,t,None) for t in range(5400,52300,500)]
    errors,metrics=audit_coverage(short,policy)
    assert metrics['active_audio_coverage']<.65
    assert any('outro' in x for x in errors)
    assert not audit_coverage([(t,t,None) for t in range(1000,89000,500)],policy)[0]


def test_silent_intro_outro_and_break_do_not_require_notes():
    policy=StrictAuditPolicy(audio_duration_ms=90000,active_audio_intervals_ms=((10000,30000),(50000,80000)))
    events=[(t,t,None) for t in [*range(11000,30000,500),*range(51000,80000,500)]]
    assert not audit_coverage(events,policy)[0]
    # Merely hitting the first and last second must not disguise an empty middle.
    assert audit_coverage([(11000,11000,None),(79000,79000,None)],policy)[0]


def test_native_extension_can_leave_screen_despite_in_bounds_controls():
    event=BeatEvent(EventKind.SLIDER,1,490,370,end_beat=2,curve_type='L',control_points=((500,380),),slider_length=140)
    before=event_path(event)
    assert before[-1,0]>512 and before[-1,1]>384
    fixed=fit_event_to_playfield(event,4)
    after=event_path(fixed);radius=circle_radius(4)
    assert (after.min(0)>=radius).all() and (after.max(0)<=(512-radius,384-radius)).all()
    np.testing.assert_allclose(np.diff(before,axis=0),np.diff(after,axis=0))
    assert fixed.slider_length==event.slider_length and fixed.end_beat==event.end_beat


@pytest.mark.parametrize('curve,controls', [('P',((512,280),(480,320))),('B',((550,300),(530,340))),('C',((500,350),(480,360)))])
def test_curves_fit_without_clamping_or_replacing_type(curve,controls):
    e=BeatEvent(EventKind.SLIDER,1,480,300,end_beat=2,curve_type=curve,control_points=controls,slider_length=100)
    fixed=fit_event_to_playfield(e,4)
    assert fixed.curve_type==curve
    assert (event_path(fixed).min(0)>circle_radius(4)).all()
    np.testing.assert_allclose(np.diff(event_path(e),axis=0),np.diff(event_path(fixed),axis=0),atol=1e-7)


def test_trim_and_multisegment_and_oversize():
    np.testing.assert_allclose(slider_path((0,0),((100,0),(100,100)),'L',150)[-1],[100,50])
    p=slider_path((0,0),((100,0),(100,0),(100,100)),'B',150)
    np.testing.assert_allclose(p[-1],[100,50])
    e=BeatEvent(EventKind.SLIDER,1,0,0,end_beat=2,curve_type='L',control_points=((100,0),),slider_length=1000)
    with pytest.raises(ValueError,match='cannot fit'): fit_event_to_playfield(e,4)


def test_empty_audible_bar_retries_without_reusing_empty_history(tmp_path,monkeypatch):
    from test_v2_alignment import batch_and_model
    from src.v2.inference import extract_audio_features,_sample_candidate_map
    from src.v2.generation import GenerationRequest
    from src.v2.grammar import BarEventCodec
    _,planner,_,geometry=batch_and_model(tmp_path)
    features,clocks=extract_audio_features(torch.zeros(1,48000),rhythm_dim=4,semantic_dim=3,bpm=120,offset_ms=0)
    codec=BarEventCodec(planner.vocabulary);calls=[]
    def generate(memory,**kwargs):
        calls.append(kwargs['previous_bar_tokens'])
        events=[] if len(calls)==1 else [BeatEvent(EventKind.CIRCLE,.5,100,100)]
        return codec.encode(events,bar_start_beat=0,meter=4)
    monkeypatch.setattr(planner,'generate_bar',generate)
    trace={}
    result=_sample_candidate_map(planner=planner,geometry=geometry.eval(),features=features,hypothesis=clocks[0],
        request=GenerationRequest(audio_filename='test.wav',target_star=4.5),seed=1,temperature=.9,geometry_steps=3,diagnostics=trace)
    assert len(result.events)==1 and len(calls)==2 and calls[-1] is None
    assert trace['bars'][0]['initial_events']==0 and trace['bars'][0]['retries']==1

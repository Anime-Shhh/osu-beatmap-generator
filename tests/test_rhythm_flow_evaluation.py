import random

import torch

from scripts.evaluate_rhythm_flow import match_onsets, score_windows


def events(*times):
    return [{'start_ms':t} for t in times]


def test_onset_matching_is_one_to_one_and_includes_tolerance_boundary():
    assert len(match_onsets(events(100),events(90,110),25)) == 1
    assert len(match_onsets(events(10,40),events(35,65),25)) == 2
    assert match_onsets(events(0),events(100),25) == []


def test_ordered_matcher_has_maximum_cardinality_on_small_cases():
    def exhaustive(p,r,tolerance):
        if not p or not r:
            return 0
        best=exhaustive(p[1:],r,tolerance)
        for i,value in enumerate(r):
            if abs(p[0]-value)<=tolerance:
                best=max(best,1+exhaustive(p[1:],r[:i]+r[i+1:],tolerance))
        return best
    rng=random.Random(42)
    for _ in range(100):
        p=[rng.randrange(100) for _ in range(rng.randrange(6))]
        r=[rng.randrange(100) for _ in range(rng.randrange(6))]
        assert len(match_onsets(events(*p),events(*r),20))==exhaustive(p,r,20)


def test_exact_onsets_and_hold_occupancy_score_perfectly():
    signal=torch.zeros(5,80)
    signal[0,10]=1; signal[1,30]=1; signal[3,30:50]=1
    reference={'source':'song','frames':80,'target':signal.clone(),'events':[
        {'start_ms':100.,'end_ms':100.,'kind':0},
        {'start_ms':300.,'end_ms':500.,'kind':1}]}
    result=score_windows([signal],[reference],.5)
    for tolerance in ('25','50','100'):
        assert result['onsets'][tolerance]['f1']==1
    assert result['kind_accuracy_among_50ms_matches']==1
    assert result['occupancy_iou']['slider']==1
    # The existing decoder places the end on the final occupied 10 ms frame.
    assert result['matched_hold_duration_error_ms']['mean']==10


def test_missing_events_and_empty_predictions_do_not_become_true_matches():
    signal=torch.zeros(5,80)
    reference={'source':'song','frames':80,'target':signal.clone(),'events':[
        {'start_ms':100.,'end_ms':100.,'kind':0}]}
    result=score_windows([signal],[reference],.5)
    assert result['onsets']['50']['recall']==0
    assert result['onsets']['50']['f1']==0
    assert result['empty_windows']==1
    assert result['matched_onset_error_ms'] is None

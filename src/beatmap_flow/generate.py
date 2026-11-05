"""Audio + continuous star request -> sampled native osu!standard draft.

Both learned stages require their own trained checkpoint. Native slider velocity
is solved from predicted duration, repeats and length, then independently parsed
and measured. Exports are conditional on the configured hard output gates.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import torch
from src.beatmap_flow.geometry import CURVES, event_condition, sample_geometry_flow
from src.beatmap_flow.rhythm import sample_rhythm, decode_events
from scripts.train_onset_flow import load_model as load_rhythm
from scripts.train_native_geometry_flow import load_model as load_geometry
from src.v2.inference import load_audio, extract_audio_features, timing_points_from_hypothesis
from src.v2.preprocess import compact_profile
from src.v2.mapspec import (MapSpec, GeneralSettings, MapMetadata, DifficultySettings,
                           compile_osu, mapspec_from_osu, validate_map_spec)
from src.v2.events import BeatEvent, EventKind
from src.v2.timing import TimingPoint, timing_map_from_points
from src.v2.slider_path import fit_event_to_playfield
from src.v2.coverage import policy_for_audio
from src.v2.evaluation import Candidate, StrictAuditPolicy, audit_candidate, RosuDifficultyCalculator
from src.v2.packaging import GeneratedPackage, write_osz_archive


def profile_for_request(stars, guide=None):
    """Use training-only settings near the requested stars when a guide exists."""
    if not math.isfinite(stars) or not 0 < stars <= 12:
        raise ValueError('target stars must be finite and in (0,12]')
    settings = DifficultySettings()
    if guide is not None:
        profiles = torch.tensor(guide['training_profiles'])
        distance = (profiles[:, 0]*10-stars).abs()
        neighbors = profiles[distance.argsort()[:min(64, len(profiles))]]
        median = neighbors.median(0).values
        sm = .05*200**float(median[4]); tick = .05*200**float(median[5])
        settings = DifficultySettings(hp_drain_rate=float(median[3]*10),
            circle_size=float(median[2]*10), overall_difficulty=float(median[3]*10),
            approach_rate=float(median[1]*10), slider_multiplier=sm, slider_tick_rate=tick)
    return settings, compact_profile(target_star=stars, approach_rate=settings.approach_rate,
        circle_size=settings.circle_size, overall_difficulty=settings.overall_difficulty,
        hp_drain_rate=settings.hp_drain_rate, slider_multiplier=settings.slider_multiplier,
        slider_tick_rate=settings.slider_tick_rate)


def compatible_training_sources(rhythm, geometry):
    """Independent stages may share a corpus or its verified split-preserving parent."""
    return geometry['rhythm_manifest_sha256'] in (
        rhythm['manifest_sha256'], rhythm.get('parent_manifest_sha256'))


@torch.inference_mode()
def predict_topology(model, condition, starts):
    n = len(starts)
    width, overlap = model.config.window_events, model.config.overlap_events
    result = condition.new_zeros(n, 46)
    count = condition.new_zeros(n, 1)
    for start in range(0, n, width-overlap):
        end = min(n, start+width)
        mask = torch.ones(1, end-start, device=condition.device, dtype=torch.bool)
        _, attributes = model.encode(condition[None, start:end], starts[None, start:end], mask)
        result[start:end] += attributes[0]; count[start:end] += 1
        if end == n: break
    return result/count


def native_plan(events, attributes, timing_points, settings):
    """Compile native topology and solve SV without altering predicted times."""
    timing = timing_map_from_points(timing_points)
    starts = torch.tensor([e['start_ms'] for e in events], device=attributes.device)
    ends = torch.tensor([e['end_ms'] for e in events], device=attributes.device)
    kinds = torch.tensor([e['kind'] for e in events], device=attributes.device)
    curves = attributes[:, :4].argmax(-1)
    curves = torch.where(kinds.eq(1), curves, 0)
    counts = attributes[:, 4:36].argmax(-1)+1
    counts = torch.where(curves.eq(2), 2, counts)
    counts = torch.where(kinds.eq(1), counts, 0)
    repeats = attributes[:, 36:44].argmax(-1)+1
    repeats = torch.where(kinds.eq(1), repeats, 1)
    lengths = attributes[:, 44].clamp(0, math.log1p(50)).expm1()*100
    lengths = torch.where(kinds.eq(1), lengths.clamp_min(5), 0.)
    points = list(timing_points)
    constrained = 0
    for i, e in enumerate(events):
        if e['kind'] != 1: continue
        duration = e['end_ms']-e['start_ms']
        if duration <= 0: raise ValueError('nonpositive predicted slider duration')
        beat_length = timing.beat_length_at_time(e['start_ms'])
        unit = 100*settings.slider_multiplier*duration/(beat_length*int(repeats[i]))
        length = min(max(float(lengths[i]), .1*unit), 10*unit)
        constrained += abs(length-float(lengths[i])) > 1e-5
        lengths[i] = length
        sv = length/unit
        points.append(TimingPoint(e['start_ms'], -100/sv, uninherited=False))
    # Red before green at equal timestamps; later slider points override old SV.
    points.sort(key=lambda p: (p.offset_ms, not p.uninherited))
    return {'starts': starts, 'ends': ends, 'kinds': kinds, 'curves': curves, 'counts': counts,
            'repeats': repeats, 'lengths': lengths, 'combo': attributes[:, 45].sigmoid()}, tuple(points), constrained


@torch.inference_mode()
def geometry_for_plan(model, condition, plan, *, steps, seed):
    n, device = len(plan['starts']), condition.device
    output = condition.new_zeros(n, 33, 2)
    width, overlap = model.config.window_events, model.config.overlap_events
    generator = torch.Generator(device=device).manual_seed(seed)
    for start in range(0, n, width-overlap):
        end = min(n, start+width)
        b = {k: v[None, start:end] for k, v in plan.items() if isinstance(v, torch.Tensor)}
        b['condition'] = condition[None, start:end]
        b['mask'] = torch.ones(1, end-start, device=device, dtype=torch.bool)
        b['positions'] = torch.arange(start, end, device=device)[None]
        b['features'] = torch.cat((b['condition'][..., -3:], torch.log1p(b['lengths'][..., None]/100)/4), -1)
        known_mask = torch.zeros_like(b['mask'])
        if start: known_mask[:, :min(overlap, end-start)] = True
        output[start:end] = sample_geometry_flow(model, b, steps=steps, generator=generator,
            known=output[None, start:end] if start else None,
            known_mask=known_mask if start else None)[0]
        if end == n: break
    return output


def compile_plan(plan, geometry, timing_points, settings, *, audio_filename, title, stars):
    timing = timing_map_from_points(timing_points)
    events = []
    for i in range(len(plan['starts'])):
        kind = EventKind(int(plan['kinds'][i]))
        count = int(plan['counts'][i])
        head = geometry[i, 0]*geometry.new_tensor([512., 384.])
        controls = geometry[i, 1:count+1]*geometry.new_tensor([512., 384.])
        start = float(plan['starts'][i]); end = float(plan['ends'][i])
        event = BeatEvent(kind=kind, beat_position=timing.time_to_beat(start),
            end_beat=timing.time_to_beat(end) if kind else None,
            x=256. if kind == EventKind.SPINNER else float(head[0]),
            y=192. if kind == EventKind.SPINNER else float(head[1]),
            curve_type=CURVES[int(plan['curves'][i])],
            control_points=tuple(tuple(map(float, xy)) for xy in controls),
            slider_repeats=int(plan['repeats'][i]), slider_length=float(plan['lengths'][i]),
            new_combo=i == 0 or float(plan['combo'][i]) >= .5, source_object_index=i)
        events.append(fit_event_to_playfield(event, settings.circle_size))
    return MapSpec(14, GeneralSettings(audio_filename),
        MapMetadata(title, 'Unknown artist', 'Audio Native Flow', f'Generated {stars:g} stars', tags='generated draft'),
        settings, timing_points, tuple(events))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audio', type=Path)
    p.add_argument('--stars', type=float, required=True)
    p.add_argument('--rhythm-checkpoint', type=Path, required=True)
    p.add_argument('--geometry-checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--calibration', type=Path, help='validation metrics.json; binds threshold to checkpoint hash')
    p.add_argument('--profile-guide', type=Path, help='training-only profile settings')
    p.add_argument('--threshold', type=float, default=None)
    p.add_argument('--candidates', type=int, default=8)
    p.add_argument('--steps', type=int, default=48)
    p.add_argument('--seed', type=int, default=1731)
    p.add_argument('--star-tolerance', type=float, default=.5)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.candidates < 1 or a.steps < 1 or a.star_tolerance < 0: p.error('invalid candidate budget')
    if not a.audio.is_file(): p.error('input audio missing')
    device = torch.device('cuda:0' if a.device == 'cuda' else a.device)
    torch.set_num_threads(1)
    rhythm, rc = load_rhythm(a.rhythm_checkpoint, device)
    geometry, gc = load_geometry(a.geometry_checkpoint, device)
    if not compatible_training_sources(rc, gc):
        raise ValueError('rhythm and geometry training sources do not match')
    hashes = {k: hashlib.sha256(path.read_bytes()).hexdigest() for k, path in
              [('rhythm', a.rhythm_checkpoint), ('geometry', a.geometry_checkpoint), ('audio', a.audio)]}
    threshold = a.threshold
    if a.calibration:
        calibration = json.loads(a.calibration.read_text())['models']['onset_flow']
        if calibration['checkpoint_sha256'] != hashes['rhythm']:
            raise ValueError('calibration belongs to a different checkpoint')
        if threshold is not None: raise ValueError('choose calibrated threshold or explicit diagnostic override')
        threshold = calibration['threshold']
    if threshold is None: raise ValueError('provide validation calibration or an explicit diagnostic threshold')
    guide = json.loads(a.profile_guide.read_text()) if a.profile_guide else None
    if guide is not None and guide['manifest_sha256'] not in (rc['manifest_sha256'], rc.get('parent_manifest_sha256')):
        raise ValueError('profile guide and model sources differ')
    settings, profile = profile_for_request(a.stars, guide)
    waveform, sample_rate = load_audio(a.audio)
    bundle, hypotheses = extract_audio_features(waveform, sample_rate, audio_sha256=hashes['audio'])
    if bundle.feature_cache.pipeline_key != rc['feature_pipeline_key'] or rc['feature_pipeline_key'] != gc['feature_pipeline_key']:
        raise ValueError('audio feature pipeline differs from training')
    audio = bundle.rhythm_features.to(device)
    profile = profile.to(device)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    policy = policy_for_audio(StrictAuditPolicy(max_star_error=a.star_tolerance, minimum_start_gap_ms=10.), bundle)
    calculator = RosuDifficultyCalculator()
    report = {'schema': 'osu-native-flow-generation/1', 'target_stars': a.stars, 'threshold': threshold,
        'training_sources': {'rhythm': rc['manifest_sha256'], 'geometry': gc['rhythm_manifest_sha256'],
                             'rhythm_parent': rc.get('parent_manifest_sha256')},
        'calibrated_threshold': bool(a.calibration), 'hashes': hashes, 'settings': asdict(settings),
        'audio_duration_ms': bundle.duration_ms, 'steps': a.steps, 'candidates': [],
        'human_playtest': 'not performed', 'selected': None,
        'clock_limitation': 'audio autocorrelation hypotheses; learned events retain millisecond timing'}
    accepted = []
    for i in range(a.candidates):
        seed = a.seed+i
        entry = {'candidate': i, 'seed': seed}
        try:
            onset, marks = sample_rhythm(rhythm, audio.T[None], profile[None], steps=a.steps,
                                         generator=torch.Generator(device=device).manual_seed(seed))
            events = decode_events(onset[0, 0].cpu(), marks[0].cpu(), threshold=threshold, duration_ms=bundle.duration_ms)
            if not events: raise ValueError('no sampled events')
            entry['rhythm'] = {'events': len(events), 'kind_counts': dict(Counter(e['kind'] for e in events)),
                'first_ms': events[0]['start_ms'], 'last_end_ms': max(e['end_ms'] for e in events)}
            (a.output_dir/f'candidate-{i}-events.json').write_text(json.dumps(events))
            starts = torch.tensor([e['start_ms'] for e in events], device=device)
            ends = torch.tensor([e['end_ms'] for e in events], device=device)
            kinds = torch.tensor([e['kind'] for e in events], device=device)
            condition = event_condition(audio, starts, ends, kinds, profile)
            attributes = predict_topology(geometry, condition, starts)
            hypothesis = hypotheses[i % len(hypotheses)]
            plan, points, constrained = native_plan(events, attributes, timing_points_from_hypothesis(hypothesis), settings)
            coordinates = geometry_for_plan(geometry, condition, plan, steps=a.steps, seed=seed+100000)
            spec = compile_plan(plan, coordinates, points, settings,
                audio_filename='audio'+a.audio.suffix.lower(), title=a.audio.stem, stars=a.stars)
            text = compile_osu(spec)
            parsed = mapspec_from_osu(text)
            if not validate_map_spec(parsed).valid: raise ValueError('round-trip native validation failed')
            audit = audit_candidate(Candidate(f'flow-{i}', parsed, seed=seed), target_star=a.stars,
                                    calculator=calculator, policy=policy)
            entry.update(valid=audit.valid, errors=list(audit.errors), warnings=list(audit.warnings),
                         difficulty=audit.difficulty.to_dict(), metrics=dict(audit.metrics),
                         constrained_slider_lengths=constrained, clock=hypothesis.to_dict())
            if audit.valid:
                accepted.append((abs(audit.difficulty.stars-a.stars), i, text))
        except (ValueError, RuntimeError) as exc:
            entry.update(valid=False, errors=[str(exc)])
        report['candidates'].append(entry)
        (a.output_dir/'audit.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'candidate': i, 'valid': entry['valid'], 'errors': entry.get('errors', [])[:2],
                          'stars': entry.get('difficulty', {}).get('stars')}), flush=True)
    if not accepted:
        print(json.dumps({'exported': False, 'reason': 'no candidate passed hard gates'}), flush=True)
        raise SystemExit(2)
    _, selected, text = min(accepted)
    osu_path = a.output_dir/'generated.osu'
    osu_path.write_text(text)
    audio_name = 'audio'+a.audio.suffix.lower()
    # Package original audio bytes directly; audio is never regenerated.
    package = GeneratedPackage(a.output_dir, osu_path, a.output_dir/'audit.json',
        hashlib.sha256(osu_path.read_bytes()).hexdigest(), audio_path=a.audio, audio_sha256=hashes['audio'])
    # Archive paths must match AudioFilename even when the source name is different.
    import os
    import shutil
    audio_path = a.output_dir/audio_name
    try: os.link(a.audio, audio_path)
    except OSError: shutil.copyfile(a.audio, audio_path)
    package = replace(package, audio_path=audio_path)
    archive = write_osz_archive(package, a.output_dir/'generated.osz')
    report['selected'] = selected
    report['artifacts'] = {'osu': osu_path.name, 'osz': archive.name,
        'osu_sha256': package.osu_sha256, 'osz_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    (a.output_dir/'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'exported': True, 'osz': str(archive), 'candidate': selected}), flush=True)


if __name__ == '__main__': main()

"""Generate full-song playtest archives and preserve strict and diagnostic audits."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time
import zipfile

import torch

from src.v2.clock import TimingHypothesis
from src.v2.coverage import policy_for_audio
from src.v2.evaluation import Candidate, RosuDifficultyCalculator, StrictAuditPolicy, build_generation_result, rerank_candidates
from src.v2.generation import GenerationRequest
from src.v2.inference import GeneratedDraft, _sample_candidate_map, extract_audio_features, load_audio, load_v2_models
from src.v2.mapspec import MapMetadata, mapspec_from_osu, validate_map_spec
from scripts.compare_v2_maps import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--songs-json')
    parser.add_argument('--kaguya-audio')
    parser.add_argument('--candidate-count', type=int, default=4)
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(1)
    models = load_v2_models(args.checkpoint, device='cuda')
    calculator = RosuDifficultyCalculator()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    songs = json.loads(Path(args.songs_json).read_text()) if args.songs_json else []
    if args.kaguya_audio:
        songs.append(dict(id='kaguya', audio=args.kaguya_audio, title='Kaguya-sama Opening',
                          artist='Masayuki Suzuki', target_star=4.5))
    summary = []
    for song_index, song in enumerate(songs):
        folder = root / song['id']
        folder.mkdir(exist_ok=True)
        audio_path = folder / 'audio.mp3'
        if audio_path.exists():
            raise FileExistsError('Use a fresh output root; existing playtests are preserved')
        shutil.copyfile(song['audio'], audio_path)
        print(json.dumps({'song': song['id'], 'status': 'extracting_audio', 'title': song['title']}), flush=True)
        waveform, sr = load_audio(audio_path, target_sr=24000)
        features, automatic = extract_audio_features(waveform, sr, models=models)
        modes = {'audio_only': automatic}
        if 'oracle' in song:
            modes['known_timing'] = (TimingHypothesis.from_dict(song['oracle']),)
        song_result = dict(id=song['id'], title=song['title'], target_star=song['target_star'], modes={})
        if 'reference' in song:
            shutil.copyfile(song['reference'], folder / 'human-reference.osu')
            song_result['human_reference'] = summarize((folder / 'human-reference.osu').read_text(), calculator)
        generated = []
        for mode_index, (mode, hypotheses) in enumerate(modes.items()):
            seed = args.seed + song_index * 100 + mode_index * 20
            request = GenerationRequest(audio_filename='audio.mp3', target_star=song['target_star'],
                seed=seed, candidate_count=args.candidate_count,
                feature_revision=features.feature_cache.pipeline_revision, model_revision=models.model_revision,
                **{key:song[key] for key in ('ar','cs','od','hp','slider_multiplier','slider_tick_rate') if key in song})
            candidates, failures = [], []
            for i in range(args.candidate_count):
                hypothesis = hypotheses[i % len(hypotheses)]
                started = time.monotonic()
                print(json.dumps({'song':song['id'],'mode':mode,'candidate':i,'status':'sampling',
                                  'clock':hypothesis.to_dict()}), flush=True)
                try:
                    spec = _sample_candidate_map(planner=models.planner, geometry=models.geometry,
                        features=features, hypothesis=hypothesis, request=request, seed=seed+i,
                        temperature=0.9, geometry_steps=32, minimum_start_gap_ms=10.0,
                        metadata=MapMetadata(title=song['title'], artist=song['artist'],
                                             creator='osu-beatmap-generator V2', version='AI playtest - '+mode,
                                             beatmap_id=0, beatmap_set_id=-1))
                    candidates.append(Candidate(candidate_id=f'{mode}-{i}', map_spec=spec, seed=seed+i,
                                      provenance={'clock_hypothesis_id':hypothesis.hypothesis_id,
                                                  'clock_source':hypothesis.source,'mode':mode}))
                    print(json.dumps({'song':song['id'],'mode':mode,'candidate':i,'objects':len(spec.events),
                                      'seconds':time.monotonic()-started,'status':'sampled'}), flush=True)
                except (ValueError, RuntimeError) as exc:
                    failures.append({'candidate':i,'error':str(exc),'seconds':time.monotonic()-started})
                    print(json.dumps({'song':song['id'],'mode':mode,'candidate':i,'error':str(exc)}), flush=True)
            strict = rerank_candidates(candidates, target_star=request.target_star, calculator=calculator,
                                       policy=policy_for_audio(StrictAuditPolicy(max_star_error=0.5, minimum_start_gap_ms=10.0), features))
            diagnostic = rerank_candidates(candidates, target_star=request.target_star, calculator=calculator,
                                           policy=policy_for_audio(StrictAuditPolicy(max_star_error=None, minimum_start_gap_ms=10.0), features))
            passing = [a for a in strict if a.valid]
            calibrated = bool(passing)
            audits = strict if calibrated else diagnostic
            result = build_generation_result(request, audits, architecture_id=models.checkpoint_provenance.architecture_id,
                                              model_revision=models.model_revision)
            report = dict(strict_passes=len(passing), candidates=args.candidate_count,
                          sampling_failures=failures, calibrated=calibrated,
                          strict_audits=[a.to_manifest().to_dict() for a in strict], result=result.to_dict())
            if result.selected_candidate_id:
                selected_audit = next(a for a in audits if a.candidate.candidate_id == result.selected_candidate_id)
                actual = selected_audit.difficulty.stars
                label = f'AI {mode} - {actual:.2f} stars' + ('' if calibrated else ' - uncalibrated draft')
                selected = replace(selected_audit.candidate,
                                   map_spec=replace(selected_audit.candidate.map_spec,
                                                    metadata=replace(selected_audit.candidate.map_spec.metadata,version=label)))
                final_candidates = tuple(selected if c.candidate_id == selected.candidate_id else c for c in candidates)
                draft = GeneratedDraft(result=result, candidates=final_candidates, audits=tuple(audits))
                package = draft.package(folder, osu_filename=mode+'.osu', manifest_filename=mode+'.manifest.json',
                                        audio_source_path=audio_path)
                report['selected'] = summarize(package.osu_path.read_text(), calculator)
                generated.append(package.osu_path)
            song_result['modes'][mode] = report
            (folder / (mode+'.audit.json')).write_text(json.dumps(report, indent=2))
            print(json.dumps({'song':song['id'],'mode':mode,'calibrated':calibrated,'strict_passes':len(passing),
                              'selected':report.get('selected')}), flush=True)
        if generated:
            archive_path = root / ('kaguya_v2_trained.osz' if song['id']=='kaguya' else song['id']+'.osz')
            with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(audio_path, 'audio.mp3')
                for path in folder.glob('*.osu'):
                    archive.write(path, path.name)
            with zipfile.ZipFile(archive_path) as archive:
                if archive.testzip() is not None:
                    raise RuntimeError('Archive failed CRC validation')
                for name in archive.namelist():
                    if name.endswith('.osu'):
                        text = archive.read(name).decode('utf-8')
                        spec = mapspec_from_osu(text)
                        if not validate_map_spec(spec).valid or spec.general.audio_filename not in archive.namelist():
                            raise RuntimeError('Archive contains an invalid map or missing referenced audio')
            song_result['osz'] = str(archive_path)
            song_result['bytes'] = archive_path.stat().st_size
        summary.append(song_result)
        (root / ('summary-kaguya.json' if args.kaguya_audio else 'summary-heldout.json')).write_text(json.dumps(summary, indent=2))
    if any('osz' not in song for song in summary):
        raise RuntimeError('Some songs produced no structurally valid AI maps; inspect preserved audits')


if __name__ == '__main__':
    main()

"""Predeclared gates; missing evidence is a failure, never an implicit promotion."""
import json
from pathlib import Path
from .contracts import write_json,digest
from .metrics import bootstrap,summarize


def records(report,mode='generation',condition='aligned'):
    if not report.get('complete'):raise ValueError('incomplete report')
    return [r for r in report['rows'] if r['mode']==mode and r['condition']==condition]


def codec_gate(ae,vae):
    a=ae['summaries']['reconstruction/aligned'];v=vae['summaries']['reconstruction/aligned']
    rows=records(vae,'reconstruction')
    sampled=[{**r,'metrics':r['sampled_metrics']} for r in rows if 'sampled_metrics' in r]
    shuffled=[{**r,'metrics':r['shuffled_metrics']} for r in rows]
    cursor=sum(r['cursor_mse'] for r in rows);shuffled_cursor=sum(r['shuffled_cursor_mse'] for r in rows)
    ss=summarize(shuffled)
    checks={'onsets':v['f1_10']>=.97,'pairs':v['pair_reference_count']>0 and v['pair_recall']>=.95,
            'duration':v['slider_duration_median_ms'] is not None and v['slider_duration_median_ms']<=12,
            'no_regression':all(v[k]>=a[k]-.01 for k in ('f1_10','type_f1_10','pair_recall')),
            'uses_latent':v['f1_10']-ss['f1_10']>=.10 or shuffled_cursor>=1.25*max(cursor,1e-12),
            'sampled_posterior_measured':len(sampled)==len(rows)}
    if sampled:
        s=summarize(sampled)
        checks['sampled_detail_preserved']=s['f1_10']>=.97 and s['pair_recall']>=.95 and s['slider_duration_median_ms'] is not None and s['slider_duration_median_ms']<=12
    return {'passed':all(checks.values()),'checks':checks,'summary':v}


def promotion_gate(baseline,candidate,vae=False):
    b=records(baseline);c=records(candidate);old=summarize(b);new=summarize(c)
    delta=bootstrap(b,c)
    checks={'gain':delta['gain']>=(0 if vae else .03),'positive_interval':delta['ci95'][0]>0,
            'fine_timing':new['f1_10']>=old['f1_10']-.01,'pairs':new['pair_recall']>=old['pair_recall']-.01,
            'difficulty':new['star_error']<=old['star_error'],'validity':new['invalid_rate']<=old['invalid_rate']}
    wrong=records(candidate,condition='wrong')
    audio=bootstrap(wrong,c) if wrong else None
    checks['audio_dependence']=audio is not None and audio['ci95'][0]>0
    return {'passed':all(checks.values()),'checks':checks,'gain':delta,'audio_gain':audio}


def decide(kind,output,baseline,candidates):
    base=json.loads(Path(baseline).read_text());results=[]
    for c in candidates:
        report=json.loads(Path(c['report']).read_text())
        gate=codec_gate(base,report) if kind=='codec' else promotion_gate(base,report,kind=='vae-generator')
        results.append({**c,'gate':gate})
    passing=[r for r in results if r['gate']['passed']]
    selected=max(passing,key=lambda r: r['gate']['summary']['f1_10'] if kind=='codec' else r['gate']['gain']['gain']) if passing else None
    result={'kind':kind,'candidates':results,'selected':selected,
            'selected_generator':digest(selected['checkpoint']) if selected and kind!='codec' else 'upstream'}
    write_json(output,result);return result

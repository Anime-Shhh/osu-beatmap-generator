"""Label selected archives without altering gameplay; verify their original audio."""
import argparse, hashlib, json, re, zipfile
from pathlib import Path
from src.dreamer_adapter.contracts import digest,write_json
from src.dreamer_adapter.metrics import native_metrics
from src.v2.mapspec import mapspec_from_osu


def run(directory):
    directory=Path(directory);report=json.loads((directory/'comparison.json').read_text());audit={}
    for kind,result in report['results'].items():
        path=directory/(kind+'.osz')
        with zipfile.ZipFile(path) as z:
            if z.testzip():raise ValueError('broken archive')
            audio=z.read('audio.mp3');text=z.read('map.osu').decode()
        if hashlib.sha256(audio).hexdigest()!=report['audio_sha256']:raise ValueError('audio changed')
        for key,value in [('Title','DADDY ! DADDY ! DO ! feat. Suzuki Airi (TV Size)'),('Artist','Suzuki Masayuki'),('Creator','Experimental model comparison'),('BeatmapID','0'),('BeatmapSetID','-1'),('Version',f"{directory.name} {kind} {result['selected_stars']:.2f} stars (4.6 requested)")]:
            text=re.sub(r'(?m)^'+key+r'\s*:.*$',key+': '+value,text)
        # All gameplay remains the raw upstream export, including validation findings.
        with zipfile.ZipFile(path) as z:old=z.read('map.osu').decode()
        for section in ('TimingPoints','HitObjects','Difficulty'):
            pattern=r'(?ms)^\['+section+r'\]\s*\n(.*?)(?=^\[|\Z)'
            if re.search(pattern,old).group(1)!=re.search(pattern,text).group(1):raise ValueError('gameplay changed')
        (directory/(kind+'.osu')).write_text(text)
        temp=path.with_suffix('.tmp')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('audio.mp3',audio);z.writestr(kind+'.osu',text)
        temp.replace(path)
        audit[kind]=dict(osz_sha256=digest(path),osu_sha256=digest(directory/(kind+'.osu')),audio_verified=True,gameplay_changed=False,raw_archive_sha256=result['osz_sha256'])
    write_json(directory/'packaging-audit.json',audit)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');run(**vars(p.parse_args()))

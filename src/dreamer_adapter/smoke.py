"""One real-data optimizer update per codec, with gradient and posterior assertions."""
import copy
import json
from pathlib import Path
import torch
from .model import load_upstream,Codec
from .data import load_pair
from .contracts import write_json


def smoke(upstream,cache,output,seconds=600):
    torch.set_num_threads(1);torch.manual_seed(1731)
    p=Path(cache);progress=json.loads((p/('manifest.json' if (p/'manifest.json').exists() else 'progress.json')).read_text())
    row=next(r for r in progress.get('rows',progress.get('charts')) if r['split']=='train')
    a,x,_=load_pair(p,progress,row)
    a,x=a[None,:,:2052].cuda(),x[None,:,:2052].cuda()
    model=load_upstream(upstream,'cuda');results=[];exports=[]
    from .evaluate import sample,decode
    from .model import intervene
    from .metrics import native_metrics
    from src.v2.mapspec import mapspec_from_osu
    labels=torch.tensor([[4.5,8.8,6.5,4.,6.5]],device='cuda')
    with torch.inference_mode():
        for condition in ('aligned','zero','shift+500','shift-500','wrong'):
            changed=intervene(a[0],condition,a[0].flip(-1))
            chart,out_labels=sample(model,changed,labels,return_labels=True)
            assert chart.shape==x[0].shape and torch.isfinite(chart).all()
            text=decode(chart.cpu().numpy(),out_labels.cpu().numpy())
            measured=native_metrics(text,mapspec_from_osu(text))
            # Corrupted audio may produce an invalid map. Its scored failure is
            # an evaluation outcome, not a broken GPU/runtime path.
            record={'condition':condition,'stars':measured['stars'],
                    'invalid':measured['invalid'],'errors':measured['errors']}
            assert measured['stars'] is not None or measured['invalid']
            exports.append(record);print(json.dumps(record),flush=True)
            write_json(str(output)+'.progress.json',{'exports':exports,'seed':1731})
    torch.cuda.empty_cache()
    for kind in ('ae','vae'):
        codec=Codec(copy.deepcopy(model.latent),kind).cuda().train()
        with torch.no_grad():
            original,_=model.latent.encode_chart(x)
            adapted,_,_=codec.encode(x)
        assert torch.equal(original,adapted), 'identity mean adapter changed initial mapping'
        frozen={n:p.detach().clone() for n,p in codec.named_parameters() if not p.requires_grad}
        optimizer=torch.optim.AdamW((p for p in codec.parameters() if p.requires_grad),lr=1e-4)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,details=codec.loss(a,x,1,.001)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in codec.mean.parameters())
        assert all(p.grad is None for p in codec.latent.audio_encoder.parameters())
        if kind=='vae':assert any(p.grad is not None and p.grad.abs().sum()>0 for p in codec.logvar.parameters())
        optimizer.step()
        for n,p in codec.named_parameters():
            if n in frozen:assert torch.equal(p,frozen[n])
        codec.eval()
        with torch.inference_mode():
            pred,mu,lv=codec.reconstruct(a,x)
            pred2,_,_=codec.reconstruct(a,x,sample=True)
        assert pred.shape==x.shape and torch.isfinite(pred).all()
        assert not torch.equal(pred,pred2)
        results.append({'kind':kind,'loss':float(loss.detach()),'shape':list(pred.shape),**details})
    write_json(output,{'passed':True,'results':results,'exports':exports,'seed':1731,
                       'device':torch.cuda.get_device_name(),'torch':str(torch.__version__)})

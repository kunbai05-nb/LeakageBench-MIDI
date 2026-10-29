#!/usr/bin/env python3
"""Fresh checkpoint inference for the frozen six-model, four-condition experiment."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'2')
import argparse,collections,csv,hashlib,importlib.util,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from leakagebench_midi.models import load_checkpoint
from leakagebench_midi.data import read_jsonl
from scripts.reproduce_cleaning_mitigation import fetch,verify_tree,sha,dump,read
from scripts.evaluate_checkpoint import sequence_metric
MODELS=['transformer_s','transformer_m','transformer_l','midigpt','lstm','tcn_384']
CONDS=['clean','unrelated_donor','repeated_unrelated','same_family_donor']
SEEDS=[202608040,202608041,202608042]
def original_identity(p):
 import re
 parts='/'.join(p.parts).lower().replace('-','_')
 model=next((m for m in MODELS if m in parts),None)
 if model is None and '/tcn/' in parts:model='tcn_384'
 condition=next((c for c in ['clean','unrelated_donor','same_family_donor'] if any(x==c or x.endswith('_'+c) for x in parts.split('/'))),None)
 match=re.search(r'seed[_-](\d+)',p.name)
 return (model,condition,int(match[1])) if model and condition and match else None

def summarize(out,models,reference,inputs,include_control):
 rows=[];checks=[];per_group={};split='treated'
 for model in models:
  data={c:{} for c in CONDS}
  for c in CONDS:
   for seed in SEEDS:
    p=out/'raw'/model/c/f'seed_{seed}'/'evaluation.json';ev=read(p)
    data[c][seed]={x['family_id']:x['metric'] for x in ev if x['split']==split}
  families=sorted(data['clean'][SEEDS[0]])
  if len(families)!=100:raise ValueError('Expected the same 100 intervention groups')
  for c in CONDS:
   for seed in SEEDS:assert sorted(data[c][seed])==families
  avg={c:np.mean([[data[c][s][w] for w in families] for s in SEEDS],axis=0) for c in CONDS}
  row={'model':model,'families':100}
  for c in CONDS:
   vals=[np.mean(list(data[c][s].values())) for s in SEEDS]
   row[c+'_nll']=float(np.mean(vals));row[c+'_seed_sd']=float(np.std(vals,ddof=1))
   if include_control:
    vals=[]
    for seed in SEEDS:
     ev=read(out/'raw'/model/c/f'seed_{seed}'/'evaluation.json');vals.append(np.mean([x['metric'] for x in ev if x['split']=='control']))
    row[c+'_control_nll']=float(np.mean(vals))
  unrelated=avg['unrelated_donor'];same=avg['same_family_donor'];repeat=avg['repeated_unrelated']
  row['same_minus_unrelated_nll']=float((same-unrelated).mean())
  row['relative_reduction_percent']=float((100*(unrelated-same)/unrelated).mean())
  row['same_minus_repeated_nll']=float((same-repeat).mean())
  idx=np.random.default_rng(20260906).integers(0,len(families),(10000,len(families)))
  for key,values in [('relative_reduction',100*(unrelated-same)/unrelated),('same_minus_repeated',same-repeat)]:
   lo,hi=np.quantile(values[idx].mean(1),[.025,.975]);row[key+'_ci_low']=float(lo);row[key+'_ci_high']=float(hi)
  row['same_work_ppl']=float(np.exp(same.mean()));rows.append(row)
  per_group[model]={'family_ids':families,'seed_averaged_nll':{c:avg[c].tolist() for c in CONDS}}
  for key,expected in reference.get(model,{}).items():
   if key not in row:continue
   diff=abs(row[key]-expected);tol=.01 if key.endswith('percent') else .0001
   checks.append(dict(model=model,metric=key,reference=expected,new=row[key],difference=diff,tolerance=tol,match=diff<=tol))
 tables=out/'tables';tables.mkdir(parents=True,exist_ok=True)
 with (tables/'four_condition_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 dump(tables/'four_condition_results.json',dict(rows=rows,per_group=per_group,seeds=SEEDS,conditions=CONDS,bootstrap='10000 paired intervention-group draws, seed20260906, conditional on these fitted training seeds'))
 labels={'transformer_s':'Transformer-S','transformer_m':'Transformer-M','transformer_l':'Transformer-L','midigpt':'MIDI-GPT','lstm':'LSTM','tcn_384':'TCN-384'}
 md=['# Six-model four-condition controlled-exposure experiment','','NLL: mean ± sample SD across three training seeds. Group-relative effects average seed NLL within each of 100 intervention groups before taking paired contrasts.','', '| Model | Clean NLL | Unrelated NLL | Repeated unrelated NLL | Same-work NLL | Same − unrelated | Reduction (%) | Same − repeated [95% interval] | PPL (same-work) |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:
  cells=[labels[r['model']]]+[f'{r[c+"_nll"]:.3f} ± {r[c+"_seed_sd"]:.3f}' for c in CONDS]+[f'{r["same_minus_unrelated_nll"]:+.3f}',f'{r["relative_reduction_percent"]:.2f}',f'{r["same_minus_repeated_nll"]:+.4f} [{r["same_minus_repeated_ci_low"]:+.4f}, {r["same_minus_repeated_ci_high"]:+.4f}]',f'{r["same_work_ppl"]:.2f}']
  md.append('| '+' | '.join(cells)+' |')
 md+=['','The LSTM repeated-unrelated run for seed202608041 used the recorded FP32 training repair; evaluation remains BF16. VAE and diffusion have no fourth-condition checkpoints and are not part of this table. Control groups, when requested, are the original frozen control cohort, not a newly assembled external clean test set.']
 (tables/'four_condition_results.md').write_text('\n'.join(md)+'\n')
 matched=all(x['match'] for x in checks)
 dump(out/'verification.json',dict(generation_status='SUCCESS',numeric_status='PASS' if matched else 'DIFFERENT',fresh_checkpoint_inference=True,models=models,checkpoints=len(models)*12,numerical_checks=checks,fourth_condition_per_group_checks=[dict(model=m,seed=s,details=read(out/'raw'/m/'repeated_unrelated'/f'seed_{s}'/'inference_report.json')['fourth_reference']) for m in models for s in SEEDS]))
 return matched

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['all']+MODELS,default='all');p.add_argument('--condition',choices=['all']+CONDS,default='all');p.add_argument('--include-control',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'reproduced/four_condition');p.add_argument('--force-evaluate',action='store_true');p.add_argument('--strict-match',action='store_true');p.add_argument('--check-only',action='store_true');a=p.parse_args()
 if a.check_only:
  print('Environment and reproduction entry points are ready; checkpoint inference has not been run.');return
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('A BF16-capable CUDA GPU is required')
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 spec=read(ROOT/'reproduction/four_condition.json');cache=ROOT/'reproduction/four_condition';inputs=cache/'inputs'
 fetch(spec['inputs'],inputs);verify_tree(inputs);manifest=read(inputs/'MANIFEST.json')
 models=MODELS if a.model=='all' else [a.model]
 for asset in spec['original_assets']:
  if a.condition!='repeated_unrelated' and any(m in asset['models'] for m in models):fetch(asset,cache/'original')
 fetch(spec['repeated_asset'],cache/'repeat')
 found={original_identity(p):p for p in (cache/'original').rglob('*.pt') if original_identity(p)}
 repeat=read(cache/'repeat/BUNDLE.json');meta={('tcn_384' if r['model']=='tcn_384' else r['model'],r['seed']):r for r in repeat['checkpoints']}
 modspec=importlib.util.spec_from_file_location('fourth_frozen_loader',cache/'repeat/verify_checkpoint_bundle.py');adapter=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(adapter)
 groups=collections.defaultdict(list)
 for row in read_jsonl(inputs/'evaluation_windows.jsonl'):
  if row['split']=='treated' or a.include_control:groups[row['split'],row['family_id']].append(row)
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 for model in models:
  for condition in (CONDS if a.condition=='all' else [a.condition]):
   for seed in SEEDS:
    extra=condition=='repeated_unrelated'
    r=meta[model,seed] if extra else None
    ck=cache/'repeat'/r['path'] if extra else found[model,condition,seed]
    digest=sha(ck);expected=r['sha256'] if extra else next(asset['file_sha256'][str(ck.relative_to(cache/'original'))] for asset in spec['original_assets'] if str(ck.relative_to(cache/'original')) in asset.get('file_sha256',{}))
    assert digest==expected
    dest=out/'raw'/model/condition/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True);done=dest/'inference_report.json'
    signature=dict(checkpoint_sha256=digest,windows_sha256=sha(inputs/'evaluation_windows.jsonl'),torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),control=a.include_control,micro_batch=8)
    if done.exists() and not a.force_evaluate and read(done)['signature']==signature and sha(dest/'evaluation.json')==read(done)['evaluation_sha256']:
     print('REUSE',model,condition,seed,flush=True);continue
    if extra:
     payload=torch.load(ck,map_location='cpu',weights_only=False);net=adapter.model_for(r,repeat);net.load_state_dict(payload['model'],strict=True);del payload
    else:net,metadata=load_checkpoint(ck,map_location='cuda');assert metadata['condition']==condition and metadata['seed']==seed
    net.to('cuda').eval();began=time.monotonic();ev=[];fourth_check=[]
    with torch.inference_mode():
     for (split,gid),windows in groups.items():
      value,tokens=sequence_metric(net,windows,model,torch.device('cuda'),torch.bfloat16)
      assert math.isfinite(value) and tokens>0
      expected_tokens=sum(max(0,min(len(w['token_ids']),1024)-max(1,w['prompt_token_count'])-1) for w in windows)
      assert tokens==expected_tokens, 'Scored-token mask changed'
      ev.append(dict(split=split,family_id=gid,metric=value,token_count=tokens,windows=len(windows)))
    if extra:
     ref=read(inputs/'fourth_condition_reference.json')[f'{model}/{seed}']
     for split,refname in [('treated','treated_probe'),('control','control_probe')]:
      vals=[x for x in ev if x['split']==split]
      if not vals:continue
      expected={x['family_id']:x for x in ref[refname]};assert {x['family_id'] for x in vals}==set(expected)
      assert all(x['token_count']==expected[x['family_id']]['token_count'] for x in vals)
      fourth_check.append(dict(split=split,groups=len(vals),maximum_group_nll_difference=max(abs(x['metric']-expected[x['family_id']]['nll']) for x in vals),macro_nll_difference=float(np.mean([x['metric']-expected[x['family_id']]['nll'] for x in vals]))))
    dump(dest/'evaluation.json',ev);dump(done,dict(signature=signature,evaluation_sha256=sha(dest/'evaluation.json'),elapsed_seconds=time.monotonic()-began,fourth_reference=fourth_check))
    print('COMPLETE',model,condition,seed,round(time.monotonic()-began,1),'seconds',flush=True)
    del net;torch.cuda.empty_cache()
 if a.condition!='all':
  dump(out/f'partial_{a.condition}_verification.json',dict(generation_status='SUCCESS',models=models,condition=a.condition,checkpoints=len(models)*3,full_table_generated=False))
  print('PARTIAL INFERENCE COMPLETE:',a.condition,flush=True);return
 matched=summarize(out,models,manifest['reference_table'],inputs,a.include_control)
 print('GENERATED:',out/'tables/four_condition_results.csv',flush=True);print('NUMERICAL:', 'PASS' if matched else 'DIFFERENT; see verification.json',flush=True)
 if a.strict_match and not matched:raise RuntimeError('Strict numerical comparison failed; fresh results preserved')
if __name__=='__main__':main()

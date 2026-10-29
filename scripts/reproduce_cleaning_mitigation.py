#!/usr/bin/env python3
"""Fresh inference of the frozen 2-dataset/2-model/five-condition cleaning study."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'2')
import argparse,csv,hashlib,importlib.util,json,math,subprocess,sys,tarfile,tempfile,time
from collections import defaultdict
from pathlib import Path,PurePosixPath
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
REPO='kunbai05-nb/LeakageBench-MIDI'
def read(p):return json.loads(p.read_text())
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def verify_tree(root):
 for line in (root/'SHA256SUMS').read_text().splitlines():
  expected,name=line.split('  ',1);rel=PurePosixPath(name)
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe checksum path')
  if sha(root/name)!=expected:raise ValueError(f'Checksum mismatch: {name}')
def fetch(spec,target):
 stamp=target/'.archive_sha256'
 known=spec.get('file_sha256',{})
 if known and all((target/name).is_file() and sha(target/name)==h for name,h in known.items()):return
 if not known and stamp.exists() and stamp.read_text().strip()==spec['sha256']:return
 target.mkdir(parents=True,exist_ok=True)
 # Process-local direct transport; never changes system/chat proxy settings.
 env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
 with tempfile.TemporaryDirectory(prefix='cleaning-download-') as tmp:
  path=Path(tmp)/spec['name'];tag=spec['url'].split('/download/',1)[1].split('/',1)[0]
  # The repository is private: gh also supports authenticated release downloads.
  subprocess.run(['gh','release','download',tag,'--repo',REPO,'--pattern',spec['name'],'--dir',tmp],env=env,check=True,timeout=1800)
  if sha(path)!=spec['sha256']:raise ValueError(f'Archive SHA256 mismatch: {path.name}')
  with tarfile.open(path,'r:gz') as archive:
   for member in archive:
    rel=PurePosixPath(member.name)
    if rel.is_absolute() or '..' in rel.parts or not (member.isfile() or member.isdir()):raise ValueError('Unsafe archive')
    if member.isdir():continue
    dst=target/str(rel);dst.parent.mkdir(parents=True,exist_ok=True)
    with archive.extractfile(member) as src,dst.open('wb') as dstfile:
     import shutil
     shutil.copyfileobj(src,dstfile)
 stamp.write_text(spec['sha256']+'\n')
 if known and not all((target/name).is_file() and sha(target/name)==h for name,h in known.items()):raise ValueError('Extracted checkpoint files do not match the frozen manifest')
def macro(e,labels):
 groups=defaultdict(list)
 for f,x in e.items():groups[str(labels[f])].append(x['nll'])
 return {w:float(np.mean(vals)) for w,vals in groups.items()}
def summarize(manifest,inputs,out,jobs):
 rows=[];checks=[]
 for ds in sorted({j['dataset'] for j in jobs}):
  audit=read(inputs/ds/'INPUT_AUDIT.json');labels=read(inputs/ds/'labels.json')
  for model in sorted({j['model'] for j in jobs if j['dataset']==ds}):
   chosen=[j for j in jobs if j['dataset']==ds and j['model']==model]
   data={};expected={}
   for j in chosen:
    key=(j['condition'],j['seed']);data[key]={};expected[key]={}
    for part in ['test','independent_test']:
     p=out/'raw'/ds/model/j['condition']/f'seed_{j["seed"]}'/f'{part}_evaluation.json'
     data[key][part]=macro(read(p),labels)
     expected[key][part]=macro(read(inputs/j['reference']/f'{part}_evaluation.json'),labels)
   for c in manifest['conditions']:
    row=dict(dataset=ds,model=model,condition=c)
    a=audit['conditions'][c]
    row.update(training_files=a['files'],natural_training_windows=a['windows'],exposed_test_works=a['exposed_test_works'],test_works=a['test_works'],exposure_percent=100*a['exposed_test_works']/a['test_works'],clean_token_retention_percent=100*a['clean_token_retention'])
    values={}
    for name,source in [('new',data),('reference',expected)]:
     works=sorted(source[c,manifest['seeds'][0]]['test'])
     matrix=np.array([[source[c,s]['test'][w] for w in works] for s in manifest['seeds']]);cur=matrix.mean(0)
     ref=np.mean([[source['label_reference',s]['test'][w] for w in works] for s in manifest['seeds']],axis=0)
     rel=100*(cur-ref)/ref
     ind=np.array([np.mean(list(source[c,s]['independent_test'].values())) for s in manifest['seeds']])
     values[name]=dict(work_macro_nll=float(cur.mean()),nll_seed_sd=float(matrix.mean(1).std(ddof=1)),delta_nll=float((cur-ref).mean()),signed_relative_percent=float(rel.mean()),absolute_relative_percent=float(np.abs(rel).mean()),independent_clean_nll=float(ind.mean()),independent_nll_seed_sd=float(ind.std(ddof=1)))
    row.update(values['new']);rows.append(row)
    for metric,value in values['new'].items():
     tolerance=manifest['tolerances']['table_percentage_points' if metric.endswith('_percent') else 'table_nll']
     difference=abs(value-values['reference'][metric]);checks.append(dict(dataset=ds,model=model,condition=c,metric=metric,absolute_difference=difference,tolerance=tolerance,pass_=difference<=tolerance))
 tables=out/'tables';tables.mkdir(exist_ok=True)
 with (tables/'cleaning_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 dump(tables/'cleaning_results.json',dict(rows=rows,aggregation=manifest['aggregation'],steps=3000,training_scored_tokens=49152000,seeds=manifest['seeds']))
 columns=['dataset','model','condition','training_files','exposure_percent','work_macro_nll','nll_seed_sd','delta_nll','signed_relative_percent','absolute_relative_percent','independent_clean_nll','clean_token_retention_percent']
 md=['# Five-condition cleaning experiment','',manifest['aggregation'],'']
 for ds,model in sorted({(r['dataset'],r['model']) for r in rows}):
  md.extend([f'## {ds} / {model}','','| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |'])
  for r in [r for r in rows if (r['dataset'],r['model'])==(ds,model)]:md.append('| '+' | '.join(f'{r[k]:.6f}' if isinstance(r[k],float) else str(r[k]) for k in columns)+' |')
  md.append('')
 (tables/'cleaning_results.md').write_text('\n'.join(md)+'\n')
 dump(out/'verification.json',dict(status='PASS' if all(c['pass_'] for c in checks) else 'DIFFERENT',generation_status='SUCCESS',fresh_checkpoint_inference=True,completed_checkpoints=len(jobs),table_checks=checks,per_file_diagnostics=[dict(dataset=j['dataset'],model=j['model'],condition=j['condition'],seed=j['seed'],checks=read(out/'raw'/j['dataset']/j['model']/j['condition']/f'seed_{j["seed"]}'/'inference_report.json')['file_checks']) for j in jobs],scope=manifest['scope'],torch_version=torch.__version__,gpu=torch.cuda.get_device_name()))
 return all(c['pass_'] for c in checks)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--dataset',choices=['all','atepp','lmd'],default='all');p.add_argument('--model',choices=['all','gpt','transformer'],default='all')
 p.add_argument('--output',type=Path,default=ROOT/'reproduced/cleaning_mitigation');p.add_argument('--micro-batch',type=int,default=16)
 p.add_argument('--strict-match',action='store_true',help='Return an error when generated table values exceed the frozen numerical tolerances')
 p.add_argument('--force-evaluate',action='store_true',help='Reevaluate completed checkpoints; verified downloads are reused')
 args=p.parse_args()
 if args.micro_batch<1:raise ValueError('micro-batch must be positive')
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('A BF16-capable CUDA GPU is required')
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 # Keep the same Transformer execution policy as the frozen BF16 evaluation.
 torch.backends.mha.set_fastpath_enabled(True)
 spec=read(ROOT/'reproduction/cleaning_mitigation.json')
 inputs=ROOT/'reproduction/cleaning_mitigation/inputs';fetch(spec['inputs'],inputs);verify_tree(inputs)
 manifest=read(inputs/'MANIFEST.json');out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 jobs=[j for j in manifest['jobs'] if args.dataset in ['all',j['dataset']] and args.model in ['all',j['model']]]
 for asset in manifest['checkpoint_assets']:
  if not any(j['dataset']==asset['dataset'] and j['model']==asset['model'] for j in jobs):continue
  base=ROOT/'reproduction/cleaning_mitigation'/asset['dataset']/asset['model'];fetch(asset,base)
  architecture=base/'architecture'
  for name in ['core','gpt_model','gpt2']:sys.modules.pop(name,None)
  sys.path.insert(0,str(architecture));modspec=importlib.util.spec_from_file_location('frozen_cleaning',architecture/'run.py');module=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(module)
  for j in [j for j in jobs if (j['dataset'],j['model'])==(asset['dataset'],asset['model'])]:
   checkpoint=base/j['checkpoint'];assert sha(checkpoint)==j['sha256']
   dest=out/'raw'/j['dataset']/j['model']/j['condition']/f'seed_{j["seed"]}';dest.mkdir(parents=True,exist_ok=True)
   signature=dict(checkpoint_sha256=j['sha256'],runner_sha256=sha(Path(__file__)),micro_batch=args.micro_batch,torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),input_manifest_sha256=sha(inputs/'MANIFEST.json'))
   done=dest/'inference_report.json'
   if done.exists() and not args.force_evaluate:
    report=read(done)
    # Frozen architecture files (including evaluate) were verified above.
    # Wrapper-only table/report changes do not invalidate existing inference.
    comparable=lambda sig:{k:v for k,v in sig.items() if k!='runner_sha256'}
    if comparable(report['signature'])==comparable(signature) and all(sha(dest/name)==h for name,h in report['output_sha256'].items()):
     print('REUSE',j['dataset'],j['model'],j['condition'],j['seed'],flush=True);continue
   payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
   assert payload['step']==3000 and payload['trained_tokens']==49152000
   assert payload['identity']['seed']==j['seed'] and payload['identity']['condition']==j['condition']
   net=module.StandardTransformer(payload['model_config']);net.load_state_dict(payload['model'],strict=True);del payload
   net.to('cuda').eval();started=time.monotonic();torch.cuda.reset_peak_memory_stats();file_checks=[]
   for part in ['test','independent_test']:
    with np.load(inputs/j['dataset']/'prepared'/f'{part}.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    ev=module.evaluate(net,data,args.micro_batch,'cuda');reference=read(inputs/j['reference']/f'{part}_evaluation.json');assert set(ev)==set(reference)
    for f,x in ev.items():
     assert math.isfinite(x['nll']) and x['tokens']==reference[f]['tokens'] and x['windows']==reference[f]['windows']
    maxdiff=max(abs(x['nll']-reference[f]['nll']) for f,x in ev.items())
    dump(dest/f'{part}_evaluation.json',ev)
    file_checks.append(dict(part=part,max_file_nll_difference=maxdiff,files=len(ev),scored_tokens=sum(v['tokens'] for v in ev.values())))
    # Per-file floating-point differences are diagnostics, not table equivalence.
    # Token/window identity is exact; final table tolerances are checked separately.
    file_checks[-1]['exceeds_diagnostic_threshold']=maxdiff>manifest['tolerances']['per_file_nll']
   names=['test_evaluation.json','independent_test_evaluation.json']
   dump(done,dict(signature=signature,file_checks=file_checks,elapsed_seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),output_sha256={name:sha(dest/name) for name in names}))
   print('COMPLETE',j['dataset'],j['model'],j['condition'],j['seed'],round(time.monotonic()-started,1),'seconds',flush=True)
   del net;torch.cuda.empty_cache()
  sys.path.remove(str(architecture))
 matched=summarize(manifest,inputs,out,jobs)
 print('GENERATED:',out/'tables/cleaning_results.csv',flush=True)
 print('Numerical comparison:', 'PASS' if matched else 'DIFFERENT; inspect verification.json (fresh values preserved)',flush=True)
 if args.strict_match and not matched:raise RuntimeError('Strict numerical match failed; generated outputs are preserved')
if __name__=='__main__':main()

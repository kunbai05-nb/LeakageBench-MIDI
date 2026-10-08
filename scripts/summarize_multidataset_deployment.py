#!/usr/bin/env python3
"""Reinfer frozen candidates, rebuild families and regenerate the audit table."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from leakagebench_midi.detector import build_components
ORDER=('pop909','asap','pdmx','gigamidi','maestro','lmd','aria')
DISPLAY=dict(zip(ORDER,('POP909','ASAP','PDMX','GigaMIDI','MAESTRO','LMD','Aria-MIDI')))

def array_digest(value):
    value=np.ascontiguousarray(value);h=hashlib.sha256()
    h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()

def json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def exposure(labels,assignment):
    n=int(labels.max(initial=-1))+1
    train=np.bincount(labels[assignment==0],minlength=n)>0
    test=np.bincount(labels[assignment==2],minlength=n);exposed=train & (test>0)
    return dict(test_families=int((test>0).sum()),exposed_families=int(exposed.sum()),test_files=int(test.sum()),
                exposed_files=int(test[exposed].sum()),family_exposure=float(exposed.sum()/max(1,(test>0).sum())),
                file_exposure=float(test[exposed].sum()/max(1,test.sum())))

def split_simulation(labels,repetitions,seed_base):
    n=len(labels);train,validation=int(n*.8),int(n*.1);samples=[]
    for i in range(repetitions):
        order=np.random.default_rng(seed_base+i).permutation(n)
        assignment=np.full(n,2,dtype=np.int8);assignment[order[:train]]=0
        assignment[order[train:train+validation]]=1
        samples.append(dict(seed=seed_base+i,**exposure(labels,assignment)))
    summary={}
    for key in ('family_exposure','file_exposure'):
        values=np.asarray([r[key] for r in samples])
        summary[key]=dict(mean=float(values.mean()),empirical_95_interval=np.quantile(values,[.025,.975]).tolist())
    return dict(repetitions=repetitions,seed_base=seed_base,ratios=[.8,.1,.1],
                denominator='all test families/files, including isolated failed files',
                interval='empirical 2.5%-97.5% quantiles, not bootstrap CI',summary=summary,samples=samples)

def replay(evidence,output,datasets=ORDER):
    output.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((evidence/'FROZEN_CONFIG.json').read_text())
    config=json.loads((evidence/'model/AUDIT_CONFIG.json').read_text())
    model_path=evidence/'model'/config['audit_verifier_file']
    if hashlib.sha256(model_path.read_bytes()).hexdigest()!=config['audit_verifier_sha256']:
        raise ValueError('checkpoint checksum mismatch')
    if config['decision_threshold']!=frozen['decision_threshold']:raise ValueError('threshold mismatch')
    model=joblib.load(model_path);rows=[];checks={};started=time.perf_counter()
    for name in datasets:
        print(f'{DISPLAY[name]}: checkpoint inference, protected graph, 200 splits',flush=True)
        spec=frozen['datasets'][name]
        with np.load(evidence/'inputs'/f'{name}.npz',allow_pickle=False) as z:pairs,features=z['pairs'],z['features']
        if features.shape!=(len(pairs),57) or pairs.shape!=(len(features),2):raise ValueError('invalid feature schema')
        if np.any(pairs<0) or np.any(pairs>=spec['files']) or not np.isfinite(features).all():raise ValueError('invalid candidate values')
        parts=[model.predict_proba(features[i:i+10000,config['audit_verifier_columns']])[:,1] for i in range(0,len(features),10000)]
        scores=np.concatenate(parts) if parts else np.empty(0)
        selected=np.flatnonzero(scores>=config['decision_threshold'])
        accepted,_,_,labels=build_components(pairs,scores,selected,spec['files'],**frozen['family_graph'])
        for field,value in [('selected',selected),('accepted',accepted),('labels',labels)]:
            if array_digest(value)!=spec['expected'][field]:raise ValueError(f'{name}: graph output differs: {field}')
        random=split_simulation(labels,frozen['repetitions'],frozen['seed_base'])
        with gzip.open(evidence/'manifests'/f'{name}.jsonl.gz','rt') as h:manifests=[json.loads(x) for x in h if x.strip()]
        if len(manifests)!=spec['files']:raise ValueError('manifest count mismatch')
        official=None
        if all(r.get('split') in ('train','validation','test') for r in manifests):
            assignment=np.asarray([{'train':0,'validation':1,'test':2}[r['split']] for r in manifests]);official=exposure(labels,assignment)
        if json_digest(random)!=spec['expected']['random'] or json_digest(official)!=spec['expected']['official']:
            raise ValueError(f'{name}: recomputed exposure differs')
        result=dict(files=spec['files'],valid_files=spec['valid_files'],manifest_sha256=spec['manifest_sha256'],
                    detector=frozen['detector'],candidate_pairs=len(pairs),positive_pairs=len(selected),
                    family_count=int(labels.max(initial=-1))+1,detected_random_split=random,
                    detected_official_split=official,scope=frozen['scope'])
        (output/f'{name}_results.json').write_text(json.dumps(result,indent=2)+'\n')
        counts=np.bincount(labels);family=random['summary']['family_exposure'];file=random['summary']['file_exposure']
        rows.append(dict(dataset=DISPLAY[name],midi_files=spec['files'],valid_midi_percent=100*spec['valid_files']/spec['files'],
                         multi_version_files_percent=100*float((counts[labels]>1).mean()),family_exposure_percent=100*family['mean'],
                         family_low=100*family['empirical_95_interval'][0],family_high=100*family['empirical_95_interval'][1],
                         file_exposure_percent=100*file['mean'],file_low=100*file['empirical_95_interval'][0],
                         file_high=100*file['empirical_95_interval'][1],
                         official_family_exposure_percent=None if official is None else 100*official['family_exposure'],
                         official_file_exposure_percent=None if official is None else 100*official['file_exposure']))
        checks[name]=dict(status='PASS',checkpoint_inference=True,graph_rebuild=True,split_replay=True,
                          scores_digest_matches=array_digest(scores)==spec['expected']['scores'])
    with (output/'multidataset_deployment.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['| Dataset | MIDI files | Valid MIDI (%) | Multi-version files (%) | Family exposure, % [95% EI] | File exposure, % [95% EI] | Official family/file exposure (%) |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        official='—' if r['official_family_exposure_percent'] is None else f"{r['official_family_exposure_percent']:.1f} / {r['official_file_exposure_percent']:.1f}"
        lines.append(f"| {r['dataset']} | {r['midi_files']:,} | {r['valid_midi_percent']:.1f} | {r['multi_version_files_percent']:.1f} | {r['family_exposure_percent']:.1f} [{r['family_low']:.1f}, {r['family_high']:.1f}] | {r['file_exposure_percent']:.1f} [{r['file_low']:.1f}, {r['file_high']:.1f}] | {official} |")
    (output/'multidataset_deployment.md').write_text('\n'.join(lines)+'\n')
    report=dict(status='PASS',scope=frozen['scope'],datasets=checks,elapsed_seconds=time.perf_counter()-started)
    (output/'multidataset_verification.json').write_text(json.dumps(report,indent=2)+'\n');return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--datasets',nargs='+',choices=ORDER,default=ORDER)
    a=p.parse_args();print(json.dumps(replay(a.evidence,a.output,a.datasets),indent=2))

if __name__=='__main__':main()

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import joblib
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]

class Verifier:
    def predict_proba(self,features):
        values=features[:,0].astype(np.float64)
        return np.column_stack((1-values,values))

def fixture(tmp_path):
    spec=importlib.util.spec_from_file_location('deployment',ROOT/'scripts/summarize_multidataset_deployment.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for name in ('inputs','model','manifests'):(tmp_path/name).mkdir()
    features=np.zeros((3,57),dtype=np.float32);features[:,0]=[.9,.1,.8]
    pairs=np.array([[0,1],[1,2],[3,4]],dtype=np.int64)
    scores=Verifier().predict_proba(features)[:,1];selected=np.flatnonzero(scores>=.5)
    graph=dict(maximum_size=50,soft_size=8,minimum_bridge_edges=3)
    accepted,_,_,labels=m.build_components(pairs,scores,selected,6,**graph)
    path=tmp_path/'model/audit_verifier.joblib';joblib.dump(Verifier(),path)
    config=dict(audit_verifier_file=path.name,audit_verifier_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                decision_threshold=.5,audit_verifier_columns=list(range(57)))
    (tmp_path/'model/AUDIT_CONFIG.json').write_text(json.dumps(config))
    np.savez(tmp_path/'inputs/asap.npz',features=features,pairs=pairs)
    with gzip.open(tmp_path/'manifests/asap.jsonl.gz','wt') as h:
        for i in range(6):h.write(json.dumps(dict(relative_path=f'{i}.mid'))+'\n')
    expected={k:m.array_digest(v) for k,v in dict(scores=scores,selected=selected,accepted=accepted,labels=labels).items()}
    expected.update(random=m.json_digest(m.split_simulation(labels,3,123)),official=m.json_digest(None))
    frozen=dict(detector='test auditor',scope='candidate checkpoint',decision_threshold=.5,repetitions=3,seed_base=123,
                family_graph=graph,datasets=dict(asap=dict(files=6,valid_files=6,manifest_sha256='test',expected=expected)))
    (tmp_path/'FROZEN_CONFIG.json').write_text(json.dumps(frozen));return m

def test_actual_inference_without_frozen_scores_or_families(tmp_path):
    m=fixture(tmp_path);out=tmp_path/'out';result=m.replay(tmp_path,out,['asap'])
    assert result['status']=='PASS' and result['datasets']['asap']['scores_digest_matches']
    assert set(np.load(tmp_path/'inputs/asap.npz').files)=={'pairs','features'}
    assert (out/'multidataset_deployment.csv').exists()
    assert len(json.loads((out/'asap_results.json').read_text())['detected_random_split']['samples'])==3

def test_checkpoint_tampering_rejected(tmp_path):
    m=fixture(tmp_path);(tmp_path/'model/audit_verifier.joblib').write_bytes(b'invalid')
    with pytest.raises(ValueError,match='checkpoint checksum'):m.replay(tmp_path,tmp_path/'out',['asap'])

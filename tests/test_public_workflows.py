from __future__ import annotations
import hashlib,json,os,subprocess,sys
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from leakagebench_midi import cross_probability,conditional_sibling_probability,family_aware_split,audit_split,build_contamination,analyze_effect,read_jsonl,build_family_map
from leakagebench_midi import classify_pair

ROOT=Path(__file__).resolve().parents[1]

def test_checkpoint_verifier_rejects_paths_outside_bundle(tmp_path):
 spec = importlib.util.spec_from_file_location('verify_model_checkpoints', ROOT/'scripts/verify_model_checkpoints.py')
 module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
 with pytest.raises(ValueError, match='escapes bundle root'):
  module.safe_checkpoint_path(tmp_path, '../outside.pt')
def test_analytical_probability():
 assert abs(cross_probability(2,.8,.1)-.16)<1e-12
 assert abs(conditional_sibling_probability(3,.8)-.96)<1e-12
def test_family_atomic_split():
 rows=[{'id':f'{f}-{i}','family_id':f,'tokens':10+i} for f in 'abcd' for i in range(2)]
 out,a=family_aware_split(rows,(.5,.25,.25),17);assert a['cross_split_family_count']==0
 assert all(len({x['split'] for x in out if x['family_id']==f})==1 for f in 'abcd')
def test_component_map():
 m=build_family_map(['a','b','c'],[('a','b')]);assert m['a']==m['b'] and m['a']!=m['c']
def test_contamination_integrity_and_matching():
 base=[{'id':f'b{i}','family_id':f'b{i}','tokens':100+i} for i in range(5)];treated=[{'family_id':'t','members':[{'id':'r','family_id':'t','tokens':100},{'id':'d','family_id':'t','tokens':102}]}];control=[{'id':'c','family_id':'c','tokens':100}];val=[{'id':'v','family_id':'v','tokens':100}];x=build_contamination(base,treated,control,val,[{'family_id':'t','receiver_id':'r','donor_id':'d'}],1)
 assert x['integrity']['receiver_in_train']==0 and x['integrity']['non_designated_treated_in_train']==0
 assert x['token_budget']['relative_total_difference']==0
def test_bootstrap_determinism():
 rows=[]
 for s in range(3):
  for split,fs,e in [('treated',['t1','t2'],-.1),('control',['c1','c2'],0),('clean_validation',['v1','v2'],0)]:
   for f in fs:
    common={'dataset':'synthetic','architecture':'toy','model_size':'tiny','seed':s,'split':split,'family_id':f}
    rows += [{**common,'condition':'clean','nll':2},{**common,'condition':'family_leak','nll':2+e}]
 result=analyze_effect(rows,1000,5)
 assert result==analyze_effect(rows,1000,5)
 assert result['p_two_sided']==2/1001
def test_synthetic_integration_and_structure(tmp_path):
 out=tmp_path/'demo';env=dict(os.environ,PYTHON=sys.executable);subprocess.run(['bash',str(ROOT/'scripts/run_synthetic_demo.sh'),str(out)],cwd=ROOT,env=env,check=True)
 assert json.loads((out/'family_aware_leakage_audit.json').read_text())['contaminated_test_families']==0
 assert json.loads((out/'contamination/integrity.json').read_text())['receiver_in_train']==0
 cases=json.loads((out/'structural_cases.json').read_text())
 for expected,(a,b) in cases.items():assert classify_pair(out/a,out/b)['classification']==expected

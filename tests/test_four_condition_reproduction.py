import importlib.util
from pathlib import Path
import json
import pytest
ROOT=Path(__file__).resolve().parents[1]
def module():
 spec=importlib.util.spec_from_file_location('four_condition',ROOT/'scripts/reproduce_four_condition.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_original_checkpoint_identity_retains_seed_and_condition():
 m=module()
 assert m.original_identity(Path('midigpt-clean/seed_202608040.pt'))==('midigpt','clean',202608040)
 assert m.original_identity(Path('lmd/tcn/unrelated_donor/seed_202608041.pt'))==('tcn_384','unrelated_donor',202608041)
 assert m.original_identity(Path('lmd/transformer_l/same_family_donor/seed_202608042.pt'))==('transformer_l','same_family_donor',202608042)

def test_four_conditions_use_same_groups_and_seed_before_ratio(tmp_path):
 m=module();refs={};models=['midigpt'];families=[f'w{i:03}' for i in range(100)]
 for c in m.CONDS:
  for seed,v in zip(m.SEEDS,[1.,3.,5.]):
   # Same work is exactly one lower; averaging seed-wise ratios would be wrong.
   value=v-1 if c=='same_family_donor' else v
   rows=[dict(split='treated',family_id=w,metric=value,token_count=10,windows=1) for w in families]
   dest=tmp_path/'raw'/'midigpt'/c/f'seed_{seed}';m.dump(dest/'evaluation.json',rows)
   m.dump(dest/'inference_report.json',{'fourth_reference':[]})
 assert m.summarize(tmp_path,models,refs,tmp_path,False)
 r=json.loads((tmp_path/'tables/four_condition_results.json').read_text())['rows'][0]
 assert r['relative_reduction_percent']==pytest.approx(100/3)
 assert r['same_minus_repeated_nll']==-1.
 assert r['same_minus_repeated_ci_low']==-1.
 # Missing a group in only one seed must stop aggregation.
 p=tmp_path/'raw/midigpt/clean/seed_202608040/evaluation.json';rows=json.loads(p.read_text());m.dump(p,rows[:-1])
 with pytest.raises((AssertionError,ValueError)):m.summarize(tmp_path,models,refs,tmp_path,False)

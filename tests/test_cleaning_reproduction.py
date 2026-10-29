"""Critical aggregation and input-integrity checks for checkpoint reproduction."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
def module():
 spec=importlib.util.spec_from_file_location('cleaning_reproduction',ROOT/'scripts/reproduce_cleaning_mitigation.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_work_macro_does_not_overweight_long_files():
 # Two files of the same work have very different scored-token counts.
 ev={'a':{'nll':1.,'tokens':10},'b':{'nll':3.,'tokens':1000},'c':{'nll':8.,'tokens':100}}
 assert module().macro(ev,{'a':'first','b':'first','c':'second'})=={'first':2.,'second':8.}

def test_downloaded_input_corruption_is_detected(tmp_path):
 p=tmp_path/'test.npz';p.write_bytes(b'original')
 (tmp_path/'SHA256SUMS').write_text(hashlib.sha256(b'original').hexdigest()+'  test.npz\n')
 m=module();m.verify_tree(tmp_path);p.write_bytes(b'changed')
 with pytest.raises(ValueError,match='Checksum mismatch'):m.verify_tree(tmp_path)

def test_relative_deviation_averages_seeds_before_ratio(tmp_path,monkeypatch):
 m=module();monkeypatch.setattr(m.torch.cuda,'get_device_name',lambda:'unit-test, no inference')
 inputs=tmp_path/'inputs';out=tmp_path/'output';inputs.mkdir();out.mkdir()
 (inputs/'d').mkdir()
 m.dump(inputs/'d/labels.json',{'file':'work'})
 conds=['uncorrected','label_reference'];jobs=[]
 audit={'conditions':{c:{'files':1,'windows':1,'exposed_test_works':0,'test_works':1,'clean_token_retention':1.} for c in conds}}
 m.dump(inputs/'d/INPUT_AUDIT.json',audit)
 for c in conds:
  for seed,value in enumerate([1.,3.,5.] if c=='label_reference' else [2.,4.,6.]):
   rel=f'reference/{c}/{seed}';jobs.append({'dataset':'d','model':'gpt','condition':c,'seed':seed,'reference':rel})
   for part in ['test','independent_test']:
    ev={'file':{'nll':value,'tokens':10,'windows':1}}
    m.dump(inputs/rel/f'{part}_evaluation.json',ev)
    m.dump(out/f'raw/d/gpt/{c}/seed_{seed}/{part}_evaluation.json',ev)
   m.dump(out/f'raw/d/gpt/{c}/seed_{seed}/inference_report.json',{'file_checks':[]})
 manifest={'conditions':conds,'seeds':[0,1,2],'aggregation':'test','scope':'test','tolerances':{'table_percentage_points':.01,'table_nll':.0001}}
 m.summarize(manifest,inputs,out,jobs)
 row=json.loads((out/'tables/cleaning_results.json').read_text())['rows'][0]
 assert row['signed_relative_percent']==pytest.approx(100/3)
 assert row['work_macro_nll']==4.
 assert row['delta_nll']==1.

 # Numeric disagreement must stay visible without discarding successful outputs.
 ref=inputs/'reference/uncorrected/0/test_evaluation.json'
 changed=json.loads(ref.read_text());changed['file']['nll']=9.;m.dump(ref,changed)
 assert m.summarize(manifest,inputs,out,jobs) is False
 report=json.loads((out/'verification.json').read_text())
 assert report['generation_status']=='SUCCESS'
 assert report['status']=='DIFFERENT'
 assert (out/'tables/cleaning_results.csv').is_file()

from __future__ import annotations
import csv,json,shutil,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def run(root,check=True):
 return subprocess.run([sys.executable,str(root/'scripts/reproduce_paper_results.py'),'--verify-only'],cwd=root,text=True,capture_output=True,check=check)

def test_raw_estimator_registry_and_table_agree():
 run(ROOT)
 registry={x['result_id']:x for x in json.loads((ROOT/'metadata/result_registry.json').read_text())['results']}
 derived=json.loads((ROOT/'reproducibility/derived/transformer_l_analysis.json').read_text())
 assert abs(derived['tau']-registry['lmd_transformer_l']['effect'])<5e-11
 subprocess.run([sys.executable,str(ROOT/'scripts/reproduce_paper_results.py')],cwd=ROOT,check=True)
 with (ROOT/'reproducibility/tables/confirmatory_effect.csv').open() as f:row=next(csv.DictReader(f))
 assert float(row['tau'])==registry['lmd_transformer_l']['effect']

def copied(tmp_path):
 target=tmp_path/'candidate';shutil.copytree(ROOT,target);return target

def test_modified_evidence_fails_integrity(tmp_path):
 root=copied(tmp_path);p=root/'reproducibility/evidence/transformer_family_effect.json';p.write_text(p.read_text()+' ')
 result=run(root,False);assert result.returncode!=0 and 'integrity check failed' in result.stderr

def test_modified_raw_result_fails_integrity(tmp_path):
 root=copied(tmp_path);p=root/'reproducibility/raw_results/transformer_l.jsonl';p.write_text(p.read_text().replace('"nll":','"nll":9, "old_nll":',1))
 result=run(root,False);assert result.returncode!=0 and 'integrity check failed' in result.stderr

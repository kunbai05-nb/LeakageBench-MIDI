from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_public_v2_is_sanitized():
 value=json.loads((ROOT/'results/manuscript_results_v2_public.json').read_text())
 assert value['result_count']==73
 assert value['formal_results_changed'] is False
 forbidden={'source_artifact','source_field','source_sha256'}
 assert all(not (forbidden & set(row)) for row in value['results'])

def test_public_result_verifier(tmp_path):
 target=tmp_path/'reproduced'
 subprocess.run([sys.executable,str(ROOT/'scripts/reproduce_public_results.py'),'--verify'],cwd=ROOT,check=True)
 subprocess.run([sys.executable,str(ROOT/'scripts/reproduce_public_results.py'),'--output',str(target)],cwd=ROOT,check=True)
 manifest=json.loads((ROOT/'results/RESULTS_MANIFEST.json').read_text())
 assert all((target/item['path']).is_file() for item in manifest['files'])

def test_no_packaged_research_data_or_weights():
 forbidden={'.mid','.midi','.wav','.flac','.mp3','.pt','.pth','.ckpt','.safetensors'}
 assert not [p for p in ROOT.rglob('*') if p.is_file() and p.suffix.lower() in forbidden]

def test_readme_release_boundaries():
 text=(ROOT/'README.md').read_text().lower()
 assert 'data are not distributed' in text
 assert 'reference relation, not a universal detector' in text
 assert 'checkpoints are not included' in text
 assert 'reproduce_public_results.py' in text

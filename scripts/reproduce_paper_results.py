#!/usr/bin/env python3
"""Recompute model effects from frozen public rows and render paper tables."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('public_core', ROOT / 'leakagebench_midi/core.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(p):
    return json.loads(p.read_text())

def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x]

def write(name, records, directory='tables'):
    p = ROOT / 'reproducibility' / directory / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify-only', action='store_true')
    a = ap.parse_args()
    manifest = load(ROOT / 'reproducibility/INTEGRITY_MANIFEST.json')
    for item in manifest['artifacts']:
        p = ROOT / item['relative_path']
        if item['artifact_type'] in {'RAW_RESULT', 'EVIDENCE', 'REGISTRY'} and (not p.is_file() or sha(p) != item['sha256']):
            raise ValueError(f"integrity check failed: {item['relative_path']}")
    registry = load(ROOT / 'metadata/result_registry.json')
    by = {x['result_id']: x for x in registry['results']}
    for r in registry['results']:
        ep = ROOT / r['source_audit_path']
        if sha(ep) != r['source_artifact_sha256']:
            raise ValueError(f"registry/evidence hash mismatch: {r['result_id']}")
        if 'provenance' in r:
            q = r['provenance']
            rp = ROOT / q['raw_result_path']
            if sha(rp) != q['raw_result_sha256']:
                raise ValueError(f"raw result hash mismatch: {r['result_id']}")
            out = core.analyze_effect(rows(rp), q['bootstrap_samples'], q['bootstrap_seed'], family_manifest=load(ROOT / q['family_manifest_path']), bootstrap_draw_mode=q['bootstrap_draw_mode'], family_order=q.get('family_order', 'sorted'))
            if abs(out['tau'] - r['effect']) > 5e-11 or max((abs(x - y) for (x, y) in zip(out['ci95'], r['ci95']))) > 5e-10:
                raise ValueError(f"derived result mismatch: {r['result_id']}")
    if a.verify_only:
        return
    cap = []
    for rid in ('tcn_384', 'transformer_s', 'transformer_m', 'lmd_transformer_l'):
        r = by[rid]
        cap.append({'model': r['model'], 'parameters': r['parameter_count'], 'tau': r['effect'], 'ci_low': r['ci95'][0], 'ci_high': r['ci95'][1], 'relative_improvement': r['relative_effect'], 'status': r['status']})
    write('architecture_capacity.csv', cap)
    r = by['lmd_transformer_l']
    write('confirmatory_effect.csv', [{'model': r['model'], 'parameters': r['parameter_count'], 'treated_families': r['sample_or_family_count'], 'tau': r['effect'], 'ci_low': r['ci95'][0], 'ci_high': r['ci95'][1], 'relative_improvement': r['relative_effect'], 'status': r['status']}])
    r = by['pdmx_reduced']
    write('pdmx_external.csv', [{'cohort': r['experiment'], 'treated': r['sample_or_family_count'], 'tau': r['effect'], 'ci_low': r['ci95'][0], 'ci_high': r['ci95'][1], 'relative_improvement': r['relative_effect'], 'status': r['status']}])
    census = load(ROOT / by['lmd_census_80_10_10']['source_audit_path'])['aggregate']
    records = []
    for (protocol, stages) in census.items():
        s = stages['S0_FILE_SPLIT']
        records.append({'protocol': protocol, 'test_family_contamination': s['test_family_contamination_rate']['mean'], 'test_file_contamination': s['test_file_contamination_rate']['mean']})
    write('lmd_census.csv', records)
    split = load(ROOT / 'reproducibility/evidence/split_comparison.json')
    cost = load(ROOT / 'reproducibility/evidence/family_aware_data_cost.json')
    mitigation = [{'protocol': v['protocol'], 'files': v['files_retained'], 'cross_split_known_families': v['family_overlap_count'], 'token_loss': v['discarded_token_ratio'], 'reassigned_files': cost['component_assignment']['file_reassignment_ratio'] if k == 'S2' else ''} for (k, v) in split.items() if k.startswith('S')]
    write('mitigation.csv', mitigation)
    write('prevalence.csv', [{'split': x['protocol'], 'test_family_contamination': x['test_family_contamination']} for x in records], 'figures')
    write('capacity.csv', [{'model': x['model'], 'parameters': x['parameters'], 'tau': x['tau']} for x in cap], 'figures')
    write('mitigation.csv', [{'protocol': x['protocol'], 'cross_split_known_families': x['cross_split_known_families']} for x in mitigation], 'figures')
    lmd = by['lmd_transformer_l']
    pdmx = by['pdmx_reduced']
    write('external_evidence.csv', [{'dataset': lmd['dataset'], 'tau': lmd['effect'], 'relative_improvement': lmd['relative_effect']}, {'dataset': pdmx['dataset'], 'tau': pdmx['effect'], 'relative_improvement': pdmx['relative_effect']}], 'figures')
if __name__ == '__main__':
    main()

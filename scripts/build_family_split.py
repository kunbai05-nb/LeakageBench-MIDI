#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from leakagebench_midi import read_jsonl,write_jsonl,family_aware_split
p=argparse.ArgumentParser();p.add_argument('--input_manifest',required=True);p.add_argument('--family_map');p.add_argument('--ratios',nargs='+',type=float,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--output',required=True);a=p.parse_args();rows=read_jsonl(a.input_manifest)
if a.family_map:
 fm=json.loads(Path(a.family_map).read_text());rows=[dict(r,family_id=fm[r['id']]) for r in rows]
out,audit=family_aware_split(rows,tuple(a.ratios),a.seed,names=('train','validation','test') if len(a.ratios)==3 else ('train','test'));op=Path(a.output);op.mkdir(parents=True,exist_ok=True);write_jsonl(op/'split_manifest.jsonl',out);(op/'split_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')

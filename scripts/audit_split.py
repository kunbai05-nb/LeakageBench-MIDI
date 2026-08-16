#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from leakagebench_midi import read_jsonl,audit_split
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--family_map');p.add_argument('--train',default='train');p.add_argument('--test',default='test');p.add_argument('--output',required=True);a=p.parse_args();rows=read_jsonl(a.manifest)
if a.family_map:
 fm=json.loads(Path(a.family_map).read_text());rows=[dict(r,family_id=fm[r['id']]) for r in rows]
Path(a.output).write_text(json.dumps(audit_split(rows,a.train,a.test),indent=2,sort_keys=True)+'\n')

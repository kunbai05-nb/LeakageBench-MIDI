#!/usr/bin/env python3
import argparse,json
from collections import Counter
from pathlib import Path
from leakagebench.release.public import census
p=argparse.ArgumentParser();p.add_argument('--family_map',required=True);p.add_argument('--ratios',nargs='+',type=float,required=True);p.add_argument('--num_seeds',type=int,default=1000);p.add_argument('--seed',type=int,default=0);p.add_argument('--output',required=True);a=p.parse_args();x=json.loads(Path(a.family_map).read_text());sizes=list(Counter(x.values()).values());Path(a.output).write_text(json.dumps(census(sizes,a.ratios,a.num_seeds,a.seed),indent=2,sort_keys=True)+'\n')

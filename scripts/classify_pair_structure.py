#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from leakagebench_midi import classify_pair
p=argparse.ArgumentParser();p.add_argument('midi_a');p.add_argument('midi_b');p.add_argument('--output',required=True);a=p.parse_args();Path(a.output).write_text(json.dumps(classify_pair(a.midi_a,a.midi_b),indent=2,sort_keys=True)+'\n')

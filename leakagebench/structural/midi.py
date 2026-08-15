from __future__ import annotations
import hashlib,json
from pathlib import Path
import mido

def _digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def midi_hashes(path):
 p=Path(path);m=mido.MidiFile(p,clip=False);tracks=[]
 for track in m.tracks:
  tick=0;events=[]
  for msg in track:
   tick+=msg.time
   if msg.is_meta and msg.type not in {"time_signature","set_tempo"}:continue
   d=msg.dict();d.pop("time",None);events.append((tick,tuple(sorted(d.items()))))
  tracks.append(tuple(events))
 return {"byte_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"canonical_event_sha256":_digest(tracks),"track_order_invariant_sha256":_digest(sorted(tracks))}
def classify_pair(a,b):
 x,y=midi_hashes(a),midi_hashes(b)
 if x["byte_sha256"]==y["byte_sha256"]:c="BYTE_EXACT"
 elif x["canonical_event_sha256"]==y["canonical_event_sha256"]:c="CANONICAL_EQUIVALENT"
 elif x["track_order_invariant_sha256"]==y["track_order_invariant_sha256"]:c="TRACK_ORDER_EQUIVALENT"
 else:c="STRUCTURALLY_NONEXACT"
 return {"classification":c,"a":x,"b":y,"frozen_hierarchy":["BYTE_EXACT","CANONICAL_EQUIVALENT","TRACK_ORDER_EQUIVALENT","STRUCTURALLY_NONEXACT"]}

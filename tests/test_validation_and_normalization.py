from __future__ import annotations
import math
from pathlib import Path
import mido, pytest
from leakagebench_midi import analyze_effect, audit_split, build_contamination, family_aware_split, classify_pair_normalized

def sample():
 base=[{'id':'b','family_id':'b','tokens':100}]
 treated=[{'family_id':'t','members':[{'id':'r','family_id':'t','tokens':100},{'id':'d','family_id':'t','tokens':101}]}]
 control=[{'id':'c','family_id':'c','tokens':100}]; val=[{'id':'v','family_id':'v','tokens':100}]
 assignment=[{'family_id':'t','receiver_id':'r','donor_id':'d'}]
 return base,treated,control,val,assignment

@pytest.mark.parametrize('mutation,message',[
 (lambda x:x[4][0].update(donor_id='c'),'donor'),
 (lambda x:x[4][0].update(receiver_id='missing'),'receiver'),
 (lambda x:x[1][0]['members'][0].update(family_id='wrong'),'family'),
 (lambda x:x[4][0].update(donor_id='r'),'donor'),
 (lambda x:x[4].append(dict(x[4][0])),'duplicate'),
])
def test_bad_assignments(mutation,message):
 x=list(sample());mutation(x)
 with pytest.raises(ValueError,match=message):build_contamination(*x,seed=1)

def test_duplicate_receiver_and_token_tolerance():
 x=list(sample());x[1].append({'family_id':'u','members':[{'id':'u1','family_id':'u','tokens':20},{'id':'u2','family_id':'u','tokens':21}]});x[4].append({'family_id':'u','receiver_id':'r','donor_id':'u2'})
 with pytest.raises(ValueError,match='receiver'):build_contamination(*x,seed=1)
 x=list(sample());x[1][0]['members'][1]['tokens']=10000
 with pytest.raises(ValueError,match='token'):build_contamination(*x,seed=1,max_pair_token_rel_diff=.1)

@pytest.mark.parametrize('ratios', [(-.1,.1,1.0),(math.nan,.5,.5),(math.inf,0,0),(1.1,0,0)])
def test_bad_ratios(ratios):
 with pytest.raises(ValueError):family_aware_split([{'id':'a','family_id':'a','tokens':1}],ratios,1)

def result_rows():
 out=[]
 for seed in range(3):
  for split,ids in [('treated',['t']),('control',['c']),('clean_validation',['v'])]:
   for condition,nll in [('clean',2.0),('family_leak',1.9 if split=='treated' else 2.0)]:
    for fid in ids:out.append({'dataset':'d','architecture':'a','model_size':'s','seed':seed,'condition':condition,'split':split,'family_id':fid,'nll':nll})
 return out

def test_duplicate_identical_and_conflicting_rows():
 r=result_rows();
 with pytest.raises(ValueError,match='duplicate'):analyze_effect(r+[dict(r[0])])
 bad=dict(r[0],nll=9)
 with pytest.raises(ValueError,match='duplicate'):analyze_effect(r+[bad])

def test_missing_pairs_empty_groups_and_nonfinite():
 r=result_rows();
 with pytest.raises(ValueError,match='missing result pair'):analyze_effect([x for x in r if not (x['family_id']=='t' and x['seed']==0 and x['condition']=='clean')])
 with pytest.raises(ValueError,match='treated cohort is empty'):analyze_effect([x for x in r if x['split']!='treated'])
 with pytest.raises(ValueError,match='control cohort is empty'):analyze_effect([x for x in r if x['split']!='control'])
 for val in (math.nan,math.inf):
  q=[dict(x) for x in r];q[0]['nll']=val
  with pytest.raises(ValueError,match='non-finite'):analyze_effect(q)

def test_zero_denominator_is_undefined():
 out=audit_split([{'id':'a','family_id':'a','tokens':1,'split':'train'}])
 assert out['test_file_contamination_rate'] is None and out['test_family_contamination_rate'] is None

def save(path,ppq,notes,meta=None,reverse=False,zero_off=False):
 m=mido.MidiFile(ticks_per_beat=ppq); tracks=[]
 for track_notes in notes:
  t=mido.MidiTrack()
  if meta:t.append(mido.MetaMessage('track_name',name=meta,time=0))
  for onset,duration,pitch in track_notes:
   t.append(mido.Message('note_on',note=pitch,velocity=64,time=onset))
   t.append(mido.Message('note_on' if zero_off else 'note_off',note=pitch,velocity=0,time=duration))
  tracks.append(t)
 if reverse:tracks.reverse()
 m.tracks.extend(tracks);m.save(path)

def exact(a,b):return classify_pair_normalized(a,b)['classification']=='NORMALIZED_STRUCTURAL_EXACT'
def simultaneous(path,order):
 m=mido.MidiFile(ticks_per_beat=480);t=mido.MidiTrack()
 for pitch in order:t.append(mido.Message('note_on',note=pitch,velocity=64,time=0))
 for index,pitch in enumerate(order):t.append(mido.Message('note_off',note=pitch,velocity=0,time=480 if index==0 else 0))
 m.tracks.append(t);m.save(path)
def test_normalized_equivalences(tmp_path):
 a=tmp_path/'a.mid';b=tmp_path/'b.mid'
 save(a,480,[[(0,480,60)]]);save(b,960,[[(0,960,60)]]);assert exact(a,b)
 save(a,480,[[(0,480,60)]]);save(b,480,[[(0,480,60)]],zero_off=True);assert exact(a,b)
 save(a,480,[[(0,480,60)]],meta='a');save(b,480,[[(0,480,60)]],meta='b');assert exact(a,b)
 save(a,480,[[(0,480,60)],[(0,480,64)]]);save(b,480,[[(0,480,60)],[(0,480,64)]],reverse=True);assert exact(a,b)
 simultaneous(a,[60,64]);simultaneous(b,[64,60]);assert exact(a,b)

@pytest.mark.parametrize('left,right', [([(0,480,60)],[(0,480,61)]), ([(0,480,60)],[(240,480,60)]), ([(0,480,60)],[(0,240,60)])])
def test_normalized_musical_differences(tmp_path,left,right):
 a=tmp_path/'a.mid';b=tmp_path/'b.mid';save(a,480,[left]);save(b,480,[right]);assert not exact(a,b)

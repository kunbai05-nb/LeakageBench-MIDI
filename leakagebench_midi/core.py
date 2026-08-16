from __future__ import annotations
import hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

def stable(*parts): return hashlib.sha256("\0".join(map(str,parts)).encode()).hexdigest()
def read_jsonl(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def write_jsonl(path, rows):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text("".join(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n" for x in rows))

def build_family_map(ids, edges):
 parent={x:x for x in ids}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 for a,b in edges:
  if a not in parent or b not in parent:raise ValueError("edge references unknown id")
  x,y=find(a),find(b)
  if x!=y:parent[max(x,y)]=min(x,y)
 groups=defaultdict(list)
 for x in sorted(parent):groups[find(x)].append(x)
 return {x:stable('family-component-v1',*members) for members in groups.values() for x in members}

def family_aware_split(rows, ratios=(.8,.1,.1), seed=0, names=("train","validation","test")):
 if len(ratios)!=len(names) or abs(sum(ratios)-1)>1e-9: raise ValueError("ratios must sum to one")
 by=defaultdict(list)
 for r in rows: by[r["family_id"]].append(r)
 total_tokens=sum(int(r.get("tokens",1)) for r in rows);target=np.asarray(ratios)*total_tokens;current=np.zeros(len(ratios));assigned={}
 order=sorted(by,key=lambda f:(-sum(int(x.get("tokens",1)) for x in by[f]),stable(seed,f)))
 for f in order:
  weight=sum(int(x.get("tokens",1)) for x in by[f]); choice=min(range(len(ratios)),key=lambda j:(sum(((current[k]+(weight if k==j else 0)-target[k])/max(1,target[k]))**2 for k in range(len(ratios))),j));assigned[f]=names[choice];current[choice]+=weight
 out=[dict(r,split=assigned[r["family_id"]]) for r in rows]; original={r["id"]:r.get("split") for r in rows};changed=sum(original[r["id"]] not in (None,r["split"]) for r in out)
 audit={"seed":seed,"family_components":len(by),"files":len(rows),"cross_split_family_count":0,"file_counts":dict(Counter(r["split"] for r in out)),"token_counts":{n:sum(int(r.get("tokens",1)) for r in out if r["split"]==n) for n in names},"target_ratios":dict(zip(names,ratios)),"file_reassignment_count":changed,"file_reassignment_ratio":changed/max(1,len(rows))}
 return out,audit

def audit_split(rows, train="train", test="test"):
 by=defaultdict(list)
 for r in rows: by[r["family_id"]].append(r)
 contaminated=[];test_fams=[];test_files=0;cont_files=0
 for f,m in by.items():
  tr=[x["id"] for x in m if x["split"]==train];te=[x["id"] for x in m if x["split"]==test]
  if te:test_fams.append(f);test_files+=len(te)
  if tr and te: contaminated.append({"family_id":f,"test_files":te,"train_siblings":tr,"family_size":len(m)});cont_files+=len(te)
 return {"test_families":len(test_fams),"test_files":test_files,"contaminated_test_families":len(contaminated),"contaminated_test_files":cont_files,"test_family_contamination_rate":len(contaminated)/max(1,len(test_fams)),"test_file_contamination_rate":cont_files/max(1,test_files),"family_size_statistics":dict(Counter(len(x) for x in by.values())),"contaminated":contaminated}

def cross_probability(k,p_train,p_test):
 p_other=1-p_train-p_test;return 1-(1-p_train)**k-(1-p_test)**k+p_other**k
def conditional_sibling_probability(k,p_train): return 1-(1-p_train)**(k-1)
def census(family_sizes,ratios=(.8,.1,.1),num_seeds=1000,seed=0):
 sizes=np.asarray(family_sizes,int); probs=np.asarray(ratios,float);rng=np.random.default_rng(seed);rows=[]
 for i in range(num_seeds):
  cf=ct=tf=tt=0
  for k in sizes:
   a=rng.choice(len(probs),int(k),p=probs);has_test=np.any(a==len(probs)-1);has_train=np.any(a==0);tf+=has_test;tt+=np.sum(a==len(probs)-1)
   if has_test and has_train:cf+=1;ct+=np.sum(a==len(probs)-1)
  rows.append((cf/max(1,tf),ct/max(1,tt)))
 a=np.asarray(rows);return {"num_seeds":num_seeds,"seed":seed,"test_family_contamination":{"mean":float(a[:,0].mean()),"q025":float(np.quantile(a[:,0],.025)),"q975":float(np.quantile(a[:,0],.975))},"test_file_contamination":{"mean":float(a[:,1].mean()),"q025":float(np.quantile(a[:,1],.025)),"q975":float(np.quantile(a[:,1],.975))},"analytical_by_size":{str(k):{"cross_probability":cross_probability(int(k),ratios[0],ratios[-1]),"test_member_has_train_sibling":conditional_sibling_probability(int(k),ratios[0])} for k in sorted(set(sizes.tolist()))},"family_aware":{"known_cross_split_families":0}}

def build_contamination(base, treated, control, validation, assignments, seed=0):
 tids={x["family_id"] for x in treated};cids={x["family_id"] for x in control};vids={x["family_id"] for x in validation}
 if tids&cids or tids&vids or cids&vids:raise ValueError("probe family sets overlap")
 receivers={x["receiver_id"] for x in assignments};donors={x["donor_id"] for x in assignments};rows=[r for r in base if r["family_id"] not in tids|cids|vids and r["id"] not in receivers|donors]
 donor_rows={r["id"]:r for r in treated for r in r.get("members",[])}; selected=[]
 for a in sorted(assignments,key=lambda x:x["family_id"]):
  if a["family_id"] not in tids or a["donor_id"] not in donor_rows:raise ValueError("invalid designated donor")
  selected.append(dict(donor_rows[a["donor_id"]],source_role="designated_donor"))
 clean=list(rows);leak=list(rows);unused=set(range(len(rows)));replacement=[]
 for d in selected:
  i=min(unused,key=lambda j:(abs(int(rows[j].get("tokens",1))-int(d.get("tokens",1))),stable(seed,rows[j]["id"])));unused.remove(i);replacement.append({"removed":rows[i]["id"],"donor":d["id"]});leak[i]=d
 tc=sum(int(x.get("tokens",1)) for x in clean);tl=sum(int(x.get("tokens",1)) for x in leak)
 return {"train_clean":clean,"train_family_leak":leak,"treated_probe":treated,"control_probe":control,"clean_validation":validation,"replacements":replacement,"token_budget":{"clean_tokens":tc,"family_leak_tokens":tl,"relative_mismatch":abs(tl-tc)/max(1,tc)},"integrity":{"receiver_in_train":sum(x["id"] in receivers for x in clean+leak),"non_designated_treated_in_train":sum(x["family_id"] in tids and x["id"] not in donors for x in clean+leak),"family_sets_disjoint":True}}

def analyze_effect(rows,bootstrap_samples=10000,bootstrap_seed=0):
 by={(r["seed"],r["condition"],r["split"],r["family_id"]):float(r["nll"]) for r in rows};seeds=sorted({r["seed"] for r in rows});fams={s:sorted({r["family_id"] for r in rows if r["split"]==s}) for s in ["treated","control","clean_validation"]}
 delta=lambda seed,split,f:by[seed,"family_leak",split,f]-by[seed,"clean",split,f]
 seed_effect={str(s):float(np.mean([delta(s,"treated",f) for f in fams["treated"]])-np.mean([delta(s,"control",f) for f in fams["control"]])) for s in seeds}
 pooled={split:{f:float(np.mean([delta(s,split,f) for s in seeds])) for f in fams[split]} for split in fams};tv=np.asarray(list(pooled["treated"].values()));cv=np.asarray(list(pooled["control"].values()));tau=float(tv.mean()-cv.mean());rng=np.random.default_rng(bootstrap_seed);draw=np.asarray([rng.choice(tv,len(tv),True).mean()-rng.choice(cv,len(cv),True).mean() for _ in range(bootstrap_samples)]);vv=np.asarray(list(pooled["clean_validation"].values()))
 return {"tau":tau,"ci95":[float(np.quantile(draw,.025)),float(np.quantile(draw,.975))],"seed_effects":seed_effect,"three_of_three_seed_negative":len(seeds)==3 and all(x<0 for x in seed_effect.values()),"treated_negative_sign_rate":float((tv<0).mean()),"treated_mean_delta":float(tv.mean()),"control_drift":float(cv.mean()),"clean_validation_drift":float(vv.mean()),"bootstrap_unit":"family","bootstrap_samples":bootstrap_samples,"bootstrap_seed":bootstrap_seed,"seed_fixed_effect":True}

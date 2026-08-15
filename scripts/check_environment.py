#!/usr/bin/env python3
import importlib.metadata as md,json,platform,sys
required={'numpy':'1.26','scipy':'1.15','mido':'1.3','torch':'2.3'};observed={};ok=True
for package,prefix in required.items():
 try:v=md.version(package);observed[package]=v;ok &= v.startswith(prefix)
 except md.PackageNotFoundError:observed[package]='MISSING';ok=False
out={'python':platform.python_version(),'python_compatible':sys.version_info[:2]==(3,10),'packages':observed,'compatible':bool(ok and sys.version_info[:2]==(3,10))};print(json.dumps(out,indent=2,sort_keys=True));raise SystemExit(0 if out['compatible'] else 1)


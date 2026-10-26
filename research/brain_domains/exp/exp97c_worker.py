"""exp97c worker: shard via EXP97C_SHARD/EXP97C_TOKENS => FINDINGS 38."""
import hashlib,importlib.util,json,os,pathlib
P="/domains/scripts/exp97c_perband_labels.py"
assert hashlib.sha256(pathlib.Path(P).read_bytes()).hexdigest()=="9411765b9730a25117a779266da4f4316e6f8fd78ff6c50deb72591a1d2e7cc6"
sp=importlib.util.spec_from_file_location("j",P);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
import numpy as np
sh=os.environ["EXP97C_SHARD"];T=os.environ["EXP97C_TOKENS"].split(",")
o=pathlib.Path("/domains/runs/illusionbrain-align/exp97c_"+sh);o.mkdir(parents=True,exist_ok=True)
for f in sorted(pathlib.Path("/domains/steinmetz_trials_primary").glob("*.npz")):
    if not any(t in f.stem for t in T):continue
    r=m.session_transdiff(np.load(f,allow_pickle=True))
    (o/("session_"+f.stem+".json")).write_text(json.dumps(r,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    print("OK",f.name,r.get("primary_stat"),flush=True)

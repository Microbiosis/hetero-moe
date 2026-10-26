import runpy,json,pathlib,hashlib
P="/domains/scripts/exp97c_perband_labels.py"
assert hashlib.sha256(pathlib.Path(P).read_bytes()).hexdigest()=="9411765b9730a25117a779266da4f4316e6f8fd78ff6c50deb72591a1d2e7cc6"
m=runpy.run_path(P);R=pathlib.Path("/domains/runs/illusionbrain-align")
per=[];seen={}
for f in sorted(R.glob("exp97c_w*/session_*.json")):
    r=json.loads(f.read_text());n=r.get("session");c=json.dumps(r,sort_keys=True,default=str)
    if n in seen:
        assert seen[n]==c
        continue
    seen[n]=c;per.append(r);print("===REC==="+c,flush=True)
v=m["decide"](per);d=R/"exp97c_final";d.mkdir(parents=True,exist_ok=True)
(d/"exp97c_perband_labels.json").write_text(json.dumps({"per_session":per,"verdict":v},indent=2,ensure_ascii=False,default=str))
print("===VERDICT==="+json.dumps(v,ensure_ascii=False,default=str),flush=True)

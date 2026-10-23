"""rerun_missing.py — 分批重跑缺失的被试（低并发，避免内存互抢）→ 收尾合并。

用法：python rerun_missing.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUCKET = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"

# 已完成的被试（stable/dev）
DONE = {"s06": ("stable", "dev"), "s07": ("stable",), "s08": ("stable", "dev"),
        "s09": ("stable",)}
PASSES = [("stable", "stable"), ("dev", "1.8.1.dev184031250")]

todo = []
for s in range(1, 11):
    tag = f"s{s:02d}"
    for pname, pver in PASSES:
        if pname in DONE.get(tag, ()):
            continue
        todo.append((tag, pname, pver))
print(f"需要重跑 {len(todo)} 个 Job：{[t[0]+'/'+t[1] for t in todo]}", flush=True)

BATCH = 5
VOL = ["-v", "hf://buckets/illusion-brain/training-data:/training:ro",
       "-v", "hf://buckets/illusion-brain/brain-domains:/domains",
       "--flavor", "cpu-upgrade", "--timeout", "90m", "-d"]


def stages(ids):
    if not ids:
        return {}
    r = subprocess.run(["hf", "jobs", "inspect"] + [f"redmingwei/{i}" for i in ids]
                       + ["--format", "json"], capture_output=True, text=True,
                       timeout=240)
    try:
        return {j["id"]: j["status"]["stage"] for j in json.loads(r.stdout)}
    except Exception:
        return {}


for bi in range(0, len(todo), BATCH):
    batch = todo[bi:bi + BATCH]
    ids, names = [], []
    for tag, pname, pver in batch:
        cmd = ["hf", "jobs", "run", "hf.co/spaces/redmingwei/IllusionBrain",
               "python", "-u", "/domains/scripts/eval_brain_alignment.py",
               "--only-subjects", str(int(tag[1:])), "--bd-version", pver,
               "--out-dir", f"{BUCKET}/{pname}", "--train-images", "2000"] + VOL \
              + ["--name", f"t3-{tag}-{pname}"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        jid = ""
        for line in (r.stdout or "").splitlines():
            if "id:" in line and len(line.split("id:")[-1].strip()) == 24:
                jid = line.split("id:")[-1].strip()
        ids.append(jid)
        names.append(f"{tag}/{pname}")
        print(f"  启动 {tag}/{pname} -> {jid}", flush=True)
    deadline = time.time() + 3000
    while time.time() < deadline:
        st = stages(ids)
        if st:
            done = [n for n, i in zip(names, ids) if st.get(i, "").startswith("COMPLETED")]
            bad = [n for n, i in zip(names, ids)
                   if st.get(i, "") in ("ERROR", "FAILED", "CANCELLED")]
            run = [n for n, i in zip(names, ids) if st.get(i, "") not in
                   ("COMPLETED", "ERROR", "FAILED", "CANCELLED")]
            print(f"  批次：完成 {len(done)}/{len(names)}  失败 {bad}  在跑 {run}", flush=True)
            if not run:
                break
        time.sleep(40)
    else:
        print("  ⚠ 批次超时", flush=True)

print("\n=== 下载 + 合并 ===", flush=True)
subprocess.run([sys.executable, str(HERE / "collect_alignment.py"),
                str(HERE / "alignment-results")])
r = subprocess.run([sys.executable, str(HERE.parent / "exp" / "merge_alignment.py"),
                    "--dir", str(HERE / "alignment-results" / "stable"),
                    "--dir", str(HERE / "alignment-results" / "dev"),
                    "--out", str(HERE / "alignment-results")],
                   capture_output=True, text=True)
print(r.stdout[-4000:])

"""在远端跑 v60 单测。"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("V60T_ART_DIR", "v60t_artifacts")

r = subprocess.run(
    [sys.executable, os.path.join(ROOT, "research/aligner/v60_token_locality/tests/test_package.py")],
    cwd=ROOT, capture_output=True, text=True)
print(r.stdout)
print("===== STDERR =====")
print(r.stderr[-3000:])
print(f"rc={r.returncode}")

os.makedirs(OUT, exist_ok=True)
json.dump({"rc": r.returncode, "stdout": r.stdout[-20000:], "stderr": r.stderr[-6000:]},
          open(os.path.join(OUT, "tests.json"), "w"), ensure_ascii=False, indent=1)
print(f"落盘 {OUT}/tests.json")
sys.exit(r.returncode)

"""在远端跑 v61 单测。"""
import json, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("V61T_ART_DIR", "v61t_artifacts")
r = subprocess.run([sys.executable, os.path.join(ROOT, "research/aligner/v61_delta_replication/tests/test_package.py")],
                   cwd=ROOT, capture_output=True, text=True)
print(r.stdout); print("===== STDERR ====="); print(r.stderr[-3000:]); print(f"rc={r.returncode}")
os.makedirs(OUT, exist_ok=True)
json.dump({"rc": r.returncode, "stdout": r.stdout[-20000:], "stderr": r.stderr[-6000:]},
          open(os.path.join(OUT, "tests.json"), "w"), ensure_ascii=False, indent=1)
sys.exit(r.returncode)

"""在远端跑 v58 单测 —— 本机不跑。

用法: V48_RUNNER=tools/run_v58_tests_remote.py bash ... (由 Space 调起)
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("V58T_ART_DIR", "v58t_artifacts")

r = subprocess.run([sys.executable, os.path.join(ROOT, "tests/test_v58_entanglement_area.py")],
                   cwd=ROOT, capture_output=True, text=True)
print(r.stdout[-6000:])
print("===== STDERR =====")
print(r.stderr[-2000:])
print(f"\nrc={r.returncode}")

os.makedirs(OUT, exist_ok=True)
json.dump({"rc": r.returncode, "stdout": r.stdout[-20000:], "stderr": r.stderr[-5000:]},
          open(os.path.join(OUT, "tests.json"), "w"), ensure_ascii=False, indent=1)
print(f"落盘 {OUT}/tests.json")
sys.exit(r.returncode)

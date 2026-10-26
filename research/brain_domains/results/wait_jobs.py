import json
import subprocess
import sys
import time
from pathlib import Path

IDS = """6adae305c191903149a7a3d3 6adae307925af0e27d4fa9e2 6adae309c191903149a7a3d6
6adae30ac191903149a7a3d8 6adae30c925af0e27d4fa9e7 6adae30e925af0e27d4fa9e9
6adae310925af0e27d4fa9ef 6adae312c191903149a7a3df 6adae314c191903149a7a3e2
6adae316925af0e27d4fa9f3 6adae318c191903149a7a3e7 6adae31ac191903149a7a3ea
6adae31cc191903149a7a3ee 6adae31ec191903149a7a3f0 6adae320c191903149a7a3f5
6adae322925af0e27d4fa9fb 6adae324c191903149a7a3fa 6adae326c191903149a7a3fe
6adae328c191903149a7a402 6adae32ac191903149a7a404""".split()

def stages():
    r = subprocess.run(["hf", "jobs", "inspect"] + [f"redmingwei/{i}" for i in IDS]
                       + ["--format", "json"], capture_output=True, text=True,
                       timeout=180)
    try:
        return {j["id"]: j["status"]["stage"] for j in json.loads(r.stdout)}
    except Exception:
        return {}

TERM = ("COMPLETED", "ERROR", "FAILED", "CANCELLED")
deadline = time.time() + 3000
while time.time() < deadline:
    st = stages()
    if not st:
        time.sleep(30)
        continue
    done = [i for i in IDS if st.get(i, "").startswith("COMPLETED")]
    bad = [i for i in IDS if st.get(i, "") in ("ERROR", "FAILED", "CANCELLED")]
    run = [i for i in IDS if st.get(i, "") not in TERM]
    print(f"完成 {len(done)}/{len(IDS)}  失败 {len(bad)}  在跑 {len(run)}", flush=True)
    if not run:
        print("DONE", flush=True)
        print("失败:", bad, flush=True)
        sys.exit(0)
    time.sleep(45)
print("TIMEOUT", flush=True)

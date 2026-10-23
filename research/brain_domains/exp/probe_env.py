import json
import platform
import sys

info = {"python": sys.version.split()[0], "platform": platform.platform()}
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
except Exception as e:
    info["torch_error"] = repr(e)
for m in ("numpy", "scipy", "sklearn"):
    try:
        mod = __import__(m)
        info[m] = getattr(mod, "__version__", "?")
    except Exception as e:
        info[m] = f"MISSING: {e!r}"
try:
    import safetensors
    info["safetensors"] = safetensors.__version__
except Exception as e:
    info["safetensors"] = f"MISSING: {e!r}"

# 不依赖 torch，直接看驱动认不认卡
try:
    import subprocess
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
    info["nvidia_smi_rc"] = r.returncode
    info["nvidia_smi_out"] = r.stdout.strip() or r.stderr.strip()[:300]
except Exception as e:
    info["nvidia_smi_error"] = repr(e)

print("=== JSON BEGIN ===")
print(json.dumps(info, indent=2, ensure_ascii=False))
print("=== JSON END ===")

"""实验环境信息模块。

所有实验脚本应在头部导入并调用 log_experiment_env() 记录环境信息。
实际运行环境（已通过命令查证）:
    - 容器: Docker 容器（hostname=941507f22847）运行在 WSL2 上
    - OS: Debian GNU/Linux 12 (bookworm)
    - Kernel: Linux 6.18.33.2-microsoft-standard-WSL2
    - CPU: AMD Ryzen 9 8945HX with Radeon Graphics, 32 cores
    - Python: 3.11.2
    - PyTorch: 2.13.0+cpu
    - CUDA: 不可用 (CPU only)
"""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime

import torch


def log_experiment_env(experiment_name: str) -> dict:
    """记录并打印实验环境信息。

    Args:
        experiment_name: 实验名称 (如 "v9_v30_v31_compare")

    Returns:
        dict with environment info for logging/reporting
    """
    env_info = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cpu_count": __import__("os").cpu_count(),
        "hostname": platform.node(),
    }

    # 打印环境信息
    print("=" * 78)
    print(f"实验环境: {experiment_name}")
    print("=" * 78)
    print(f"  时间:       {env_info['timestamp']}")
    print(f"  Python:     {env_info['python_version'].split()[0]}")
    print(f"  平台:       {env_info['platform']}")
    print(f"  PyTorch:    {env_info['pytorch_version']}")
    print(f"  CUDA:       {env_info['cuda_available']} ({env_info['cuda_version']})")
    print(f"  设备:       {env_info['device']}")
    print(f"  CPU 核心:   {env_info['cpu_count']}")
    print(f"  Hostname:   {env_info['hostname']}")
    print("=" * 78)

    return env_info


def get_git_commit() -> str | None:
    """获取当前 git commit hash (如果可用)。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=__import__("os").path.dirname(__file__),
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else None
    except Exception:
        return None

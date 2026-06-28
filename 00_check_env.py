"""
Checkpoint 0: Environment check (Apple Silicon / MPS version).
Run this FIRST to confirm your setup is ready before downloading the 4B model.
"""
import sys
import subprocess

print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    mps_available = torch.backends.mps.is_available()
    print(f"MPS (Apple GPU) available: {mps_available}")
    if not mps_available:
        print("WARNING: MPS not available. Check you're on macOS 12.3+ and have "
              "a recent torch build.")
except ImportError:
    print("PyTorch not installed yet. Install instructions will follow.")

try:
    import mlx.core as mx
    print(f"MLX installed: yes (device default: {mx.default_device()})")
except ImportError:
    print("MLX not installed yet (only needed if using mlx-tune path).")

# Check which fine-tuning package is actually present
for pkg in ["unsloth", "mlx_tune"]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            version_line = [l for l in result.stdout.splitlines() if l.startswith("Version")]
            print(f"{pkg} installed: yes ({version_line[0] if version_line else 'version unknown'})")
        else:
            print(f"{pkg} installed: no")
    except Exception as e:
        print(f"{pkg} check failed: {e}")

print("\nUnified memory note: your Mac shares RAM between CPU and GPU. "
      "Close other heavy apps before training to maximize usable memory.")
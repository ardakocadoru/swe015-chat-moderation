"""Reproduce every model and every number from scratch: steps 1-6 in order."""
import subprocess
import sys
from pathlib import Path

STEPS = ["01_prepare_data.py", "02_train_classical.py", "03_train_w2v_ffn.py",
         "03b_train_w2v_cnn.py", "04_finetune_bert.py", "05_export_quantize.py", "06_benchmark.py", "08_error_analysis.py"]

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for step in STEPS:
        print(f"\n######## {step}", flush=True)
        subprocess.run([sys.executable, str(here / step)], check=True)

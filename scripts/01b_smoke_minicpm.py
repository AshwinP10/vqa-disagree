"""
Minimal MiniCPM-V-2.6 access check — 1 sample, ~1 GPU-minute.
Run after accepting the gated-repo terms on HuggingFace.

Usage:
    modal run scripts/01b_smoke_minicpm.py
"""
import modal
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from modal_app.common import app, image, volume, VOL_PATH, hf_secret
from modal_app.pipeline import load_dataset_samples
from modal_app.inference import MiniCPMVLM


@app.local_entrypoint()
def main():
    print("=== MiniCPM-V-2.6 Access Check ===")
    items = load_dataset_samples.remote(dataset_keys=["vqav2"])
    item = items[:1]
    print(f"Question: {item[0]['question']}  GT={item[0]['gt'][:40]}")

    model = MiniCPMVLM()
    results = model.run_batch.remote(item)
    r = results[0]
    print(f"pred={r['prediction'][:80]!r}  ms={r['latency_ms']:.0f}")
    if r["prediction"]:
        print("OK MiniCPM access confirmed.")
    else:
        print("FAIL empty prediction — check model access.")

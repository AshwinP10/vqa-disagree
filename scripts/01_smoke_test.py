"""
Smoke test: 5 VQAv2 samples through all 4 VLMs.

Usage:
    modal run scripts/01_smoke_test.py
"""
import modal
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from modal_app.common import app, image, volume, VOL_PATH, hf_secret
from modal_app.pipeline import load_dataset_samples
from modal_app.inference import QwenVLM, LLaVAVLM, InternVLM, MiniCPMVLM


@app.local_entrypoint()
def main():
    print("=== VQA-Disagree Smoke Test ===")
    print("Loading 5 VQAv2 samples…")

    # Load a tiny batch from VQAv2 only
    items = load_dataset_samples.remote(dataset_keys=["vqav2"])
    items = items[:5]
    print(f"Got {len(items)} items. Questions:")
    for it in items:
        print(f"  [{it['idx']}] {it['question'][:80]}  GT={it['gt'][:40]}")

    models = {
        "qwen":     QwenVLM(),
        "llava":    LLaVAVLM(),
        "internvl": InternVLM(),
        "minicpm":  MiniCPMVLM(),
    }

    for name, cls in models.items():
        print(f"\n--- {name} ---")
        results = cls.run_batch.remote(items)
        for r in results:
            print(f"  idx={r['idx']}  pred={r['prediction'][:60]!r}"
                  f"  logp={r['mean_logprob']}  ms={r['latency_ms']:.0f}")

    print("\nOK Smoke test passed — all 4 models ran successfully.")


"""
Full VLM inference: 2,000 samples x 4 models.
Saves qwen.csv, llava.csv, internvl.csv, minicpm.csv to Modal volume.

The entire orchestration loop runs inside Modal — safe to close your laptop
immediately after running this command.

Usage:
    modal run scripts/02_run_all_vlms.py            # full run
    modal run scripts/02_run_all_vlms.py --dry-run  # 20 samples per model
"""
import modal
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from modal_app.common import app, image, volume, VOL_PATH, hf_secret
from modal_app.pipeline import load_dataset_samples
from modal_app.inference import QwenVLM, LLaVAVLM, InternVLM, MiniCPMVLM

CHUNK_SIZE = 50


def _chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


@app.function(
    image=image,
    volumes={VOL_PATH: volume},
    secrets=[hf_secret],
    timeout=28800,   # 8 hours — plenty for all 4 models
    cpu=2,
)
def run_all_inference(dry_run: bool = False):
    """
    Orchestration runs inside Modal — caller can exit immediately.
    Downloads all datasets, runs all 4 VLMs in sequence, saves CSVs to volume.
    """
    import json
    import pandas as pd
    from pathlib import Path as P

    # Load samples (calls another Modal function from inside Modal)
    print("Loading dataset samples...")
    all_items = load_dataset_samples.remote(dataset_keys=None)
    if dry_run:
        all_items = all_items[:20]
        print(f"[dry-run] Using {len(all_items)} items")
    else:
        print(f"Loaded {len(all_items)} items total")

    model_map = {
        "qwen":     QwenVLM(),
        "llava":    LLaVAVLM(),
        "internvl": InternVLM(),
        "minicpm":  MiniCPMVLM(),
    }

    vol_path = P(VOL_PATH)

    for model_name, cls in model_map.items():
        print(f"\n{'='*50}")
        print(f"Running {model_name} on {len(all_items)} samples...")
        all_results = []
        chunks = list(_chunked(all_items, CHUNK_SIZE))
        for i, chunk in enumerate(chunks):
            results = cls.run_batch.remote(chunk)
            all_results.extend(results)
            done = min((i + 1) * CHUNK_SIZE, len(all_items))
            print(f"  {model_name}: {done}/{len(all_items)} done")

        # Deserialize gt back to list if needed
        for r in all_results:
            try:
                r["gt"] = json.loads(r["gt"])
            except (json.JSONDecodeError, TypeError):
                pass

        csv_name = f"{model_name}.csv"
        csv_path = vol_path / csv_name
        df = pd.DataFrame(all_results)
        df.to_csv(csv_path, index=False)
        volume.commit()
        print(f"  Saved {len(df)} rows -> {csv_name}")

    print("\nOK All VLM inference complete.")
    return len(all_items)


@app.local_entrypoint()
def main(dry_run: bool = False):
    print("=== VQA-Disagree: Full VLM Inference ===")
    print("Run with --detach flag to safely close your terminal mid-run.")
    n = run_all_inference.remote(dry_run=dry_run)
    print(f"Done. {n} samples processed across all 4 models.")

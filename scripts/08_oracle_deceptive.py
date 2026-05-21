"""
Run Qwen2-VL-72B oracle on all Deceptive-stratum items (174 total).

Saves results to a NEW file `oracle_deceptive.csv` (does NOT overwrite the
existing `oracle_correct` column in `disagreement_full.csv`).  A separate
local merge step writes the final combined CSV.

Usage:
    modal run scripts/08_oracle_deceptive.py
"""
import modal
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from modal_app.common import app, image, volume, VOL_PATH, hf_secret
from modal_app.inference import Qwen72BVLM


@app.function(
    image=image,
    volumes={VOL_PATH: volume},
    secrets=[hf_secret],
    timeout=900,
    cpu=2,
)
def sample_deceptive_items() -> list[dict]:
    """Fetch images for all Deceptive items in disagreement_full.csv."""
    import pandas as pd
    from collections import defaultdict
    from datasets import load_dataset
    from modal_app.pipeline import DATASET_CONFIGS, _extract_image_id, _get_image_bytes

    vol = Path(VOL_PATH)
    df = pd.read_csv(vol / "disagreement_full.csv")
    sub = df[df["stratum"] == "deceptive"].reset_index(drop=True)
    print(f"Found {len(sub)} Deceptive items in disagreement_full.csv")
    print("Source distribution:")
    print(sub["source_dataset"].value_counts().to_string())

    # Image cache: one streaming pass per source dataset
    needed: dict[str, set] = defaultdict(set)
    for _, row in sub.iterrows():
        needed[row["source_dataset"]].add(str(row["image_id"]))

    image_cache: dict[tuple, bytes] = {}
    for ds_key, ids_needed in needed.items():
        hf_path, hf_name, split, _ = DATASET_CONFIGS[ds_key]
        print(f"[sample] Scanning {ds_key} for {len(ids_needed)} images…")
        load_kwargs = dict(split=split, streaming=True)
        if hf_name:
            load_kwargs["name"] = hf_name
        hf_ds = load_dataset(hf_path, **load_kwargs)
        found = 0
        for i, row in enumerate(hf_ds):
            row_id = _extract_image_id(row, ds_key, i)
            if row_id in ids_needed:
                img = _get_image_bytes(row, ds_key)
                if img:
                    image_cache[(ds_key, row_id)] = img
                    found += 1
            if found >= len(ids_needed):
                break
        print(f"[sample]   → found {found}/{len(ids_needed)}")

    items = []
    for _, row in sub.iterrows():
        key = (row["source_dataset"], str(row["image_id"]))
        img = image_cache.get(key)
        if img is None:
            continue
        items.append({
            "idx":            int(row["idx"]),
            "source_dataset": row["source_dataset"],
            "image_id":       str(row["image_id"]),
            "question":       row["question"],
            "gt":             row["gt"],
            "stratum":        row["stratum"],
            "image_bytes":    img,
        })
    print(f"[sample] {len(items)} items ready for oracle")
    return items


@app.function(
    image=image,
    volumes={VOL_PATH: volume},
    secrets=[hf_secret],
    timeout=300,
    cpu=2,
)
def save_deceptive_oracle(results: list[dict]) -> int:
    """Save oracle predictions for Deceptive items to oracle_deceptive.csv."""
    import json
    import pandas as pd
    from analysis.normalize import vqa_soft_match

    vol = Path(VOL_PATH)
    rows = []
    for r in results:
        gt = r["gt"]
        try:
            gt = json.loads(gt)
            if isinstance(gt, (int, float)):
                gt = str(int(gt)) if float(gt) == int(gt) else str(gt)
        except Exception:
            pass
        correct = float(vqa_soft_match(r["prediction"], gt) > 0)
        rows.append({
            "idx":            r["idx"],
            "source_dataset": r.get("source_dataset"),
            "stratum":        "deceptive",
            "oracle_pred":    r["prediction"],
            "oracle_correct": correct,
        })

    out_df = pd.DataFrame(rows)
    out_path = vol / "oracle_deceptive.csv"
    out_df.to_csv(out_path, index=False)
    volume.commit()

    print(f"\nSaved {len(out_df)} rows → {out_path}")
    print(f"Deceptive oracle accuracy: {out_df['oracle_correct'].mean():.4f}")
    print("\nBy source dataset:")
    print(out_df.groupby("source_dataset")["oracle_correct"].agg(["mean", "count"]).round(4).to_string())
    return len(out_df)


@app.local_entrypoint()
def main():
    print("=== Deceptive Stratum Oracle Validation (Qwen2-VL-72B) ===")

    items = sample_deceptive_items.remote()
    print(f"Got {len(items)} Deceptive items with images")
    if not items:
        print("No items — aborting.")
        return

    oracle = Qwen72BVLM()
    CHUNK = 20
    results = []
    chunks = [items[i: i + CHUNK] for i in range(0, len(items), CHUNK)]
    for i, chunk in enumerate(chunks):
        preds = oracle.run_batch.remote(chunk)
        for p, orig in zip(preds, chunk):
            p["gt"]             = orig["gt"]
            p["stratum"]        = orig["stratum"]
            p["source_dataset"] = orig["source_dataset"]
        results.extend(preds)
        print(f"  oracle {min((i+1)*CHUNK, len(items))}/{len(items)} done")

    n = save_deceptive_oracle.remote(results)
    print(f"\nOracle complete. {n} Deceptive items validated.")

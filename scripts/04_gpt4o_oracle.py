"""
Scale-based oracle validation via Qwen2.5-VL-72B (or 32B fallback).
Samples 300 items (100 Easy + 100 Medium + 100 Hard, balanced across datasets).
Adds oracle_correct column to disagreement_full.csv.

The key argument: if disagreement-stratified hardness persists when evaluated
by a ~10x larger same-family model, it reflects genuine task difficulty rather
than architectural idiosyncrasies.

Usage:
    modal run scripts/04_gpt4o_oracle.py
    modal run scripts/04_gpt4o_oracle.py --n-per-stratum 50   # cheaper test
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
    timeout=600,
    cpu=2,
)
def sample_oracle_items(n_per_stratum: int = 100) -> list[dict]:
    """Sample items from disagreement_full.csv and fetch their image bytes."""
    import json
    import pandas as pd
    from pathlib import Path
    from datasets import load_dataset
    from collections import defaultdict
    from modal_app.pipeline import DATASET_CONFIGS, _extract_image_id, _get_image_bytes

    vol = Path(VOL_PATH)
    df  = pd.read_csv(vol / "disagreement_full.csv")

    # Balanced sample across easy/medium/hard, spread across source datasets
    target_strata = ["easy", "medium", "hard"]
    selected = []
    for stratum in target_strata:
        sub = df[df["stratum"] == stratum]
        n   = min(n_per_stratum, len(sub))
        # Spread across source datasets
        per_src = max(1, n // sub["source_dataset"].nunique())
        idxs = []
        for _, grp in sub.groupby("source_dataset"):
            idxs.extend(grp.sample(n=min(per_src, len(grp)), random_state=42).index.tolist())
        # top up if needed
        remaining = [i for i in sub.index if i not in set(idxs)]
        idxs.extend(remaining[: n - len(idxs)])
        selected.extend(idxs[:n])

    sample_df = df.loc[selected].reset_index(drop=True)
    print(f"Oracle sample: {len(sample_df)} rows — "
          + ", ".join(f"{s}={len(sample_df[sample_df.stratum==s])}"
                      for s in target_strata))

    # Build image cache (one streaming pass per dataset)
    needed: dict[str, set] = defaultdict(set)
    for _, row in sample_df.iterrows():
        needed[row["source_dataset"]].add(str(row["image_id"]))

    image_cache: dict[tuple, bytes] = {}
    for ds_key, ids_needed in needed.items():
        hf_path, hf_name, split, _ = DATASET_CONFIGS[ds_key]
        print(f"[sample] Scanning {ds_key} for {len(ids_needed)} images…")
        try:
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
        except Exception as e:
            print(f"[sample] Error scanning {ds_key}: {e}")

    print(f"[sample] Cached {len(image_cache)} images")

    items = []
    for _, row in sample_df.iterrows():
        key = (row["source_dataset"], str(row["image_id"]))
        img = image_cache.get(key)
        if img is None:
            continue
        gt_raw = row["gt"]
        items.append({
            "idx":            int(row["idx"]),
            "source_dataset": row["source_dataset"],
            "image_id":       str(row["image_id"]),
            "question":       row["question"],
            "gt":             gt_raw,
            "stratum":        row["stratum"],
            "image_bytes":    img,
        })

    print(f"[sample] {len(items)} items with images ready for oracle")
    return items


@app.function(
    image=image,
    volumes={VOL_PATH: volume},
    secrets=[hf_secret],
    timeout=300,
    cpu=2,
)
def save_oracle_results(results: list[dict]):
    """Merge oracle predictions back into disagreement_full.csv."""
    import json
    import pandas as pd
    from pathlib import Path
    from analysis.normalize import vqa_soft_match

    vol = Path(VOL_PATH)
    df  = pd.read_csv(vol / "disagreement_full.csv")

    oracle_rows = []
    for r in results:
        gt = r["gt"]
        try:
            gt = json.loads(gt)
            if isinstance(gt, (int, float)):
                gt = str(int(gt)) if float(gt) == int(gt) else str(gt)
        except Exception:
            pass
        correct = float(vqa_soft_match(r["prediction"], gt) > 0)
        oracle_rows.append({
            "idx":            r["idx"],
            "oracle_pred":    r["prediction"],
            "oracle_correct": correct,
        })

    oracle_df = pd.DataFrame(oracle_rows)
    df["oracle_pred"]    = None
    df["oracle_correct"] = None
    df = df.set_index("idx")
    df.update(oracle_df.set_index("idx"))
    df = df.reset_index()

    df.to_csv(vol / "disagreement_full.csv", index=False)
    volume.commit()

    val = df[df["oracle_correct"].notna()]
    print("\nScale oracle (Qwen2.5-VL-72B) accuracy per stratum:")
    print(val.groupby("stratum")["oracle_correct"].agg(["mean", "count"]).round(3).to_string())

    easy_acc = val[val["stratum"] == "easy"]["oracle_correct"].mean()
    hard_acc = val[val["stratum"] == "hard"]["oracle_correct"].mean()
    if easy_acc and hard_acc:
        gap = (easy_acc - hard_acc) * 100
        print(f"\nEasy-Hard gap: {gap:.1f} pp  (target >= 15 pp)")
        if gap >= 15:
            print("Validation PASSED — disagreement is a genuine difficulty signal.")
        else:
            print("WARNING: gap < 15 pp. Consider reviewing stratum assignments.")

    return len(oracle_rows)


@app.local_entrypoint()
def main(n_per_stratum: int = 100):
    print("=== Scale-based Oracle Validation (Qwen2.5-VL-72B) ===")

    # Step 1: sample items with images (cheap CPU function)
    items = sample_oracle_items.remote(n_per_stratum=n_per_stratum)
    print(f"Got {len(items)} oracle items")

    if not items:
        print("No items — run 03_compute_disagreement.py first.")
        return

    # Step 2: run 72B oracle (expensive GPU function)
    oracle = Qwen72BVLM()
    CHUNK  = 20   # smaller batches for the large model
    results = []
    chunks = [items[i: i + CHUNK] for i in range(0, len(items), CHUNK)]
    for i, chunk in enumerate(chunks):
        preds = oracle.run_batch.remote(chunk)
        # strip image_bytes before merging (save bandwidth)
        for p, orig in zip(preds, chunk):
            p["gt"]      = orig["gt"]
            p["stratum"] = orig["stratum"]
        results.extend(preds)
        print(f"  oracle {min((i+1)*CHUNK, len(items))}/{len(items)} done")

    # Step 3: save back to volume
    n = save_oracle_results.remote(results)
    print(f"Oracle done. {n} samples validated.")

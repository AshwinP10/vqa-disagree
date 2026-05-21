"""
SBERT threshold sensitivity analysis.

Re-cluster the existing per-model predictions in disagreement_full.csv at
thresholds {0.7, 0.85, 0.9} and compare stratum-assignment stability against
the paper's default of 0.8.

Runs on CPU inside Modal (no GPU needed).  Saves a summary CSV to the volume.

Usage:
    modal run scripts/09_sbert_sensitivity.py
"""
import modal
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from modal_app.common import app, image, volume, VOL_PATH


@app.function(
    image=image,
    volumes={VOL_PATH: volume},
    timeout=1800,
    cpu=4,
    memory=8192,
)
def run_sensitivity():
    import json
    import pandas as pd
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from modal_app.pipeline import disagreement_score

    vol = Path(VOL_PATH)
    df = pd.read_csv(vol / "disagreement_full.csv")
    print(f"Loaded {len(df)} rows")

    model_names = ["qwen", "llava", "internvl", "minicpm"]
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    thresholds = [0.70, 0.80, 0.85, 0.90]
    stratum_assignments = {}

    for thr in thresholds:
        print(f"\nClustering at threshold {thr}…")
        from analysis.clustering import compute_clusters
        from analysis.normalize import vqa_soft_match

        records = []
        for _, row in df.iterrows():
            preds = [str(row.get(f"{m}_pred", "") or "") for m in model_names]
            gt_raw = row["gt"]
            try:
                gt = json.loads(gt_raw)
                if isinstance(gt, (int, float)):
                    gt = str(int(gt)) if float(gt) == int(gt) else str(gt)
            except (json.JSONDecodeError, TypeError):
                gt = str(gt_raw)

            clusters = compute_clusters(preds, sbert_model=sbert, threshold=thr)
            n = len(preds)
            largest = max(len(c) for c in clusters)
            disagree = 1.0 - (largest / n)
            correctness = [vqa_soft_match(p, gt) for p in preds]
            all_correct  = all(c > 0 for c in correctness)
            none_correct = all(c == 0 for c in correctness)
            is_easy      = (largest == n and all_correct)
            is_deceptive = (largest == n and none_correct)
            is_hard      = (disagree >= 0.5)
            is_medium    = (disagree == 0.25)
            if is_easy:        stratum = "easy"
            elif is_deceptive: stratum = "deceptive"
            elif is_hard:      stratum = "hard"
            elif is_medium:    stratum = "medium"
            else:              stratum = "easy"
            records.append({"idx": row["idx"], "stratum": stratum, "disagreement": disagree})

        out = pd.DataFrame(records).set_index("idx")
        stratum_assignments[thr] = out

    # Reference: paper's default (0.80) — should match df["stratum"]
    paper = df.set_index("idx")[["stratum"]].rename(columns={"stratum": "stratum_paper"})

    summary_rows = []
    for thr in thresholds:
        s = stratum_assignments[thr]
        joined = paper.join(s[["stratum"]].rename(columns={"stratum": f"stratum_{thr}"}))
        same = (joined["stratum_paper"] == joined[f"stratum_{thr}"]).sum()
        total = len(joined)

        counts = s["stratum"].value_counts().to_dict()
        for stratum in ["easy", "medium", "hard", "deceptive"]:
            counts.setdefault(stratum, 0)

        summary_rows.append({
            "threshold":      thr,
            "n_easy":         counts["easy"],
            "n_medium":       counts["medium"],
            "n_hard":         counts["hard"],
            "n_deceptive":    counts["deceptive"],
            "agreement_with_paper": round(same / total, 4),
            "n_total":        total,
        })

    summary = pd.DataFrame(summary_rows)
    out_path = vol / "sbert_sensitivity.csv"
    summary.to_csv(out_path, index=False)
    volume.commit()

    print("\n=== SBERT Threshold Sensitivity Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved → {out_path}")
    return summary.to_dict(orient="records")


@app.local_entrypoint()
def main():
    print("=== SBERT Threshold Sensitivity (CPU) ===")
    rows = run_sensitivity.remote()
    print("\nFinal summary:")
    for r in rows:
        print(r)

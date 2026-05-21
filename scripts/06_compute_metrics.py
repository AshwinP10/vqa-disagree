"""
Compute all summary statistics for the paper.
Downloads disagreement_full.csv from Modal volume and generates metrics_summary.csv.

Usage:
    python scripts/06_compute_metrics.py               # reads from results/
    modal run scripts/06_compute_metrics.py --remote   # reads from volume
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from analysis.stats import (
    chi_square_stratum_vs_type,
    stratum_accuracy_summary,
    bootstrap_ci,
)

MODEL_NAMES = ["qwen", "llava", "internvl", "minicpm"]
ACC_COLS    = [f"{m}_correct" for m in MODEL_NAMES]
STRATA      = ["easy", "medium", "hard", "deceptive"]


def compute_all(df: pd.DataFrame) -> dict:
    out = {}

    # ── Stratum counts & fractions ──────────────────────────────────────────
    vc = df["stratum"].value_counts()
    total = len(df)
    out["total_samples"] = total
    for s in STRATA:
        out[f"n_{s}"]    = int(vc.get(s, 0))
        out[f"frac_{s}"] = round(vc.get(s, 0) / total, 4)

    # ── Per-model accuracy per stratum ──────────────────────────────────────
    for model in MODEL_NAMES:
        col = f"{model}_correct"
        if col not in df.columns:
            continue
        for s in STRATA:
            sub = df[df["stratum"] == s][col].dropna()
            if len(sub) == 0:
                continue
            out[f"acc_{model}_{s}"] = round(float(sub.mean()), 4)

    # ── Mean accuracy across models per stratum ──────────────────────────────
    avail_acc = [c for c in ACC_COLS if c in df.columns]
    for s in STRATA:
        sub = df[df["stratum"] == s]
        if len(sub) == 0:
            continue
        vals = sub[avail_acc].values.flatten()
        vals = vals[~np.isnan(vals)]
        out[f"acc_mean_{s}"] = round(float(np.mean(vals)), 4) if len(vals) > 0 else None

    # ── Per-dataset stratum fractions ────────────────────────────────────────
    for ds, grp in df.groupby("source_dataset"):
        for s in STRATA:
            out[f"frac_{ds}_{s}"] = round(len(grp[grp["stratum"] == s]) / len(grp), 4)

    # ── GPT-4o-mini oracle ───────────────────────────────────────────────────
    if "oracle_correct" in df.columns:
        oracle = df[df["oracle_correct"].notna()]
        for s in STRATA:
            sub = oracle[oracle["stratum"] == s]["oracle_correct"]
            if len(sub) == 0:
                continue
            mean = float(sub.mean())
            lo, hi = bootstrap_ci(sub.tolist())
            out[f"oracle_acc_{s}"]    = round(mean, 4)
            out[f"gpt4o_ci_lo_{s}"]  = round(lo, 4)
            out[f"gpt4o_ci_hi_{s}"]  = round(hi, 4)

        easy_acc = out.get("oracle_acc_easy")
        hard_acc = out.get("oracle_acc_hard")
        if easy_acc and hard_acc:
            out["oracle_easy_hard_gap_pp"] = round((easy_acc - hard_acc) * 100, 1)

    # ── Chi-square test ──────────────────────────────────────────────────────
    if "question_type" in df.columns:
        chi = chi_square_stratum_vs_type(df)
        out.update({f"chi2_{k}": v for k, v in chi.items()})

    # ── Per question-type disagrement ────────────────────────────────────────
    if "question_type" in df.columns:
        for qt, grp in df.groupby("question_type"):
            out[f"mean_disagree_{qt}"] = round(grp["disagreement"].mean(), 4)
            out[f"frac_hard_{qt}"]     = round(len(grp[grp["stratum"] == "hard"]) / len(grp), 4)

    return out


def main(input_path: str = "results/disagreement_full.csv",
         output_path: str = "results/metrics_summary.json"):
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    metrics = compute_all(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {output_path}")

    # Pretty print key numbers
    print("\n=== Key Metrics ===")
    for s in STRATA:
        n    = metrics.get(f"n_{s}", 0)
        frac = metrics.get(f"frac_{s}", 0)
        acc  = metrics.get(f"acc_mean_{s}")
        gpt  = metrics.get(f"oracle_acc_{s}")
        print(f"  {s:12s}: n={n:5d}  frac={frac:.1%}  "
              f"model_acc={acc:.3f}  oracle_acc={gpt}")
    if "oracle_easy_hard_gap_pp" in metrics:
        print(f"\n  Oracle easy->hard gap: {metrics['oracle_easy_hard_gap_pp']:.1f} pp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/disagreement_full.csv")
    parser.add_argument("--output", default="results/metrics_summary.json")
    args = parser.parse_args()
    main(args.input, args.output)




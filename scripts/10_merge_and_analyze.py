"""
After 08_oracle_deceptive.py finishes:
  1. Download oracle_deceptive.csv from Modal volume (run `modal volume get`).
  2. Run this script LOCALLY to merge into disagreement_full.csv (preserves
     the original 281-item oracle results) and print fresh analysis.

Usage:
    modal volume get vqa-disagree-data oracle_deceptive.csv results/oracle_deceptive.csv --force
    python scripts/10_merge_and_analyze.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"


def main():
    full = pd.read_csv(RESULTS / "disagreement_full.csv")
    new  = pd.read_csv(RESULTS / "oracle_deceptive.csv")

    print(f"disagreement_full.csv: {len(full)} rows")
    print(f"oracle_deceptive.csv:   {len(new)} rows")
    print(f"Existing oracle rows in full CSV: {full['oracle_correct'].notna().sum()}")

    # Merge: update oracle_pred and oracle_correct for Deceptive rows
    full_idx = full.set_index("idx")
    new_idx  = new.set_index("idx")

    # Only overwrite cells where we have NEW data, never wipe existing ones.
    for col in ["oracle_pred", "oracle_correct"]:
        if col in new_idx.columns:
            full_idx.loc[new_idx.index, col] = new_idx[col].values

    full = full_idx.reset_index()
    full.to_csv(RESULTS / "disagreement_full.csv", index=False)
    print(f"\nWrote merged CSV. Oracle rows now: {full['oracle_correct'].notna().sum()}")

    # ── Print verified analysis ──
    print("\n=== Oracle accuracy per stratum (post-merge) ===")
    oracle = full[full["oracle_correct"].notna()]
    for s in ["easy", "medium", "hard", "deceptive"]:
        sub = oracle[oracle["stratum"] == s]["oracle_correct"].astype(float)
        if len(sub) == 0:
            continue
        m = sub.mean()
        se = np.sqrt(m * (1 - m) / len(sub))
        print(f"  {s:10s}: acc={m:.4f}  n={len(sub):3d}  "
              f"95%CI=[{max(0,m-1.96*se):.3f}, {min(1,m+1.96*se):.3f}]")

    # Deceptive by source dataset
    dec = oracle[oracle["stratum"] == "deceptive"]
    print("\nDeceptive oracle accuracy by source dataset:")
    print(dec.groupby("source_dataset")["oracle_correct"].agg(["mean", "count"]).round(4).to_string())

    # Easy vs Deceptive: this is the key comparison for the artifact claim
    easy_acc = oracle[oracle["stratum"] == "easy"]["oracle_correct"].mean()
    dec_acc  = dec["oracle_correct"].mean()
    print(f"\nEasy oracle accuracy:      {easy_acc:.3f}")
    print(f"Deceptive oracle accuracy: {dec_acc:.3f}")
    if dec_acc > 0.30:
        print("→ A large fraction of Deceptive items are solvable by the 72B oracle.")
        print("  Most are likely answer-format artifacts rather than shared blind spots.")
    elif dec_acc < 0.15:
        print("→ Deceptive items are genuinely hard even for the 72B oracle.")
        print("  Confirms architectural blind-spot interpretation.")
    else:
        print("→ Mixed signal; Deceptive contains both artifacts and genuine errors.")


if __name__ == "__main__":
    main()

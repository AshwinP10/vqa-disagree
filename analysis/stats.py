"""
Statistical tests for VQA-Disagree results.

Usage:
    python analysis/stats.py --input results/disagreement_full.csv
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def chi_square_stratum_vs_type(df: pd.DataFrame) -> dict:
    """Chi-square test of independence: stratum × question_type."""
    ct = pd.crosstab(df["stratum"], df["question_type"])
    chi2, pval, dof, _ = scipy_stats.chi2_contingency(ct)
    return {"chi2": float(chi2), "pval": float(pval), "dof": int(dof)}


def stratum_accuracy_summary(df: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Mean accuracy per stratum per model."""
    rows = []
    for stratum, grp in df.groupby("stratum"):
        row = {"stratum": stratum, "n": len(grp)}
        for col in model_cols:
            if col in grp.columns:
                row[col] = grp[col].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("stratum")


def bootstrap_ci(values: list[float], n_boot: int = 2000,
                 ci: float = 0.95) -> tuple[float, float]:
    arr = np.array(values)
    boot = [np.mean(np.random.choice(arr, size=len(arr), replace=True))
            for _ in range(n_boot)]
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def run(input_path: str):
    df = pd.read_csv(input_path)
    print(f"\nLoaded {len(df)} rows from {input_path}")

    # Stratum counts
    print("\n--- Stratum distribution ---")
    print(df["stratum"].value_counts().to_string())

    # Source dataset distribution
    print("\n--- Stratum × source_dataset ---")
    print(pd.crosstab(df["source_dataset"], df["stratum"]).to_string())

    # Accuracy per stratum
    acc_cols = [c for c in df.columns if c.endswith("_correct")]
    if acc_cols:
        print("\n--- Mean accuracy per stratum ---")
        print(df.groupby("stratum")[acc_cols].mean().round(3).to_string())

    # Chi-square: stratum vs question type
    if "question_type" in df.columns:
        result = chi_square_stratum_vs_type(df)
        print(f"\n--- Chi-square (stratum × question_type) ---")
        print(f"  χ²={result['chi2']:.2f}, p={result['pval']:.4f}, dof={result['dof']}")

    # GPT-4o accuracy per stratum if present
    if "gpt4o_correct" in df.columns:
        gpt_df = df[df["gpt4o_correct"].notna()]
        print("\n--- GPT-4o-mini accuracy per stratum ---")
        print(gpt_df.groupby("stratum")["gpt4o_correct"].agg(["mean", "count"]).round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/disagreement_full.csv")
    args = parser.parse_args()
    run(args.input)

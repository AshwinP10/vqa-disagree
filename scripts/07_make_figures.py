"""
Generate all 4 paper figures + 2 LaTeX tables.
Reads results/disagreement_full.csv.

Usage:
    python scripts/07_make_figures.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

FIGURES_DIR = ROOT / "figures"
PAPER_FIG_DIR = ROOT / "paper" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
PAPER_FIG_DIR.mkdir(exist_ok=True)

MODEL_NAMES  = ["qwen", "llava", "internvl", "minicpm"]
MODEL_LABELS = {"qwen": "Qwen2-VL-7B", "llava": "LLaVA-1.6-M",
                "internvl": "InternVL2-8B", "minicpm": "MiniCPM-V 2.6"}
STRATA       = ["easy", "medium", "hard", "deceptive"]
STRATA_COLORS = {"easy": "#4caf50", "medium": "#ff9800", "hard": "#f44336", "deceptive": "#9c27b0"}

DATASET_LABELS = {
    "vqav2": "VQAv2", "gqa": "GQA", "textvqa": "TextVQA",
    "chartqa": "ChartQA", "realworldqa": "RealWorldQA",
}

sns.set_theme(style="whitegrid", font_scale=1.0)


def save(fig, name: str):
    for d in [FIGURES_DIR, PAPER_FIG_DIR]:
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight", dpi=150)
    print(f"  Saved {name}.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Teaser: per-stratum accuracy per model
# ─────────────────────────────────────────────────────────────────────────────

def fig1_teaser(df: pd.DataFrame):
    acc_data = {}
    for s in STRATA:
        sub = df[df["stratum"] == s]
        row = {}
        for m in MODEL_NAMES:
            col = f"{m}_correct"
            if col in sub.columns:
                row[MODEL_LABELS[m]] = sub[col].mean() * 100
        if row:
            acc_data[s] = row

    models = list(MODEL_LABELS.values())
    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, s in enumerate(STRATA):
        if s not in acc_data:
            continue
        vals = [acc_data[s].get(m, 0) for m in models]
        bars = ax.bar(x + i * width, vals, width, label=s.capitalize(),
                      color=STRATA_COLORS[s], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-Stratum Model Accuracy", fontweight="bold")
    ax.legend(title="Stratum", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.axhline(50, ls="--", color="gray", lw=0.8, alpha=0.5)
    plt.tight_layout()
    save(fig, "fig1_teaser")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Stacked stratum distribution per source dataset
# ─────────────────────────────────────────────────────────────────────────────

def fig2_stacked_distribution(df: pd.DataFrame):
    datasets = sorted(df["source_dataset"].unique())
    data = {}
    for ds in datasets:
        grp = df[df["source_dataset"] == ds]
        data[DATASET_LABELS.get(ds, ds)] = {
            s: len(grp[grp["stratum"] == s]) / len(grp) * 100 for s in STRATA
        }

    ds_labels = list(data.keys())
    x = np.arange(len(ds_labels))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bottoms = np.zeros(len(ds_labels))
    for s in STRATA:
        vals = [data[d].get(s, 0) for d in ds_labels]
        ax.bar(x, vals, bottom=bottoms, color=STRATA_COLORS[s],
               label=s.capitalize(), edgecolor="white", linewidth=0.5)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Sample fraction (%)")
    ax.set_title("Stratum Distribution per Benchmark", fontweight="bold")
    ax.legend(title="Stratum", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    save(fig, "fig2_stacked_distribution")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Heatmap: question_type × stratum (disagreement rate)
# ─────────────────────────────────────────────────────────────────────────────

def fig3_heatmap(df: pd.DataFrame):
    if "question_type" not in df.columns:
        print("  [skip fig3] no question_type column")
        return
    from analysis.question_typing import QUESTION_TYPES
    pivot = pd.crosstab(
        df["question_type"], df["stratum"], normalize="index"
    ) * 100
    # Reorder
    cols = [s for s in STRATA if s in pivot.columns]
    rows = [qt for qt in QUESTION_TYPES if qt in pivot.index]
    pivot = pivot.reindex(index=rows, columns=cols, fill_value=0)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "% of question type"},
                annot_kws={"size": 8})
    ax.set_title("Stratum Distribution by Question Type", fontweight="bold")
    ax.set_xlabel("Stratum")
    ax.set_ylabel("Question type")
    plt.tight_layout()
    save(fig, "fig3_heatmap")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Scale Oracle Accuracy (Qwen2.5-VL-72B) vs stratum (validation)
# ─────────────────────────────────────────────────────────────────────────────

def fig4_oracle_validation(df: pd.DataFrame):
    if "oracle_correct" not in df.columns or df["oracle_correct"].isna().all():
        print("  [skip fig4] no oracle_correct column")
        return

    oracle = df[df["oracle_correct"].notna()]
    accs, errs = [], []
    strata_avail = [s for s in STRATA if s in oracle["stratum"].values]
    for s in strata_avail:
        sub = oracle[oracle["stratum"] == s]["oracle_correct"].astype(float)
        accs.append(sub.mean() * 100)
        sem = sub.std() / np.sqrt(len(sub)) * 100
        errs.append(sem)

    fig, ax = plt.subplots(figsize=(4.5, 3))
    colors = [STRATA_COLORS[s] for s in strata_avail]
    bars = ax.bar(strata_avail, accs, yerr=errs, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.5, error_kw={"lw": 1.5})
    ax.set_ylabel("Scale Oracle Accuracy (Qwen2-VL-72B) (%)")
    ax.set_title("Oracle Accuracy per Stratum", fontweight="bold")
    ax.set_ylim(0, 105)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 1.5,
                f"{acc:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save(fig, "fig4_oracle_validation")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 — Per-model accuracy per stratum
# ─────────────────────────────────────────────────────────────────────────────

def table1_accuracy(df: pd.DataFrame) -> str:
    rows = []
    for m in MODEL_NAMES:
        col = f"{m}_correct"
        if col not in df.columns:
            continue
        row = [MODEL_LABELS[m]]
        for s in STRATA:
            sub = df[df["stratum"] == s][col].dropna()
            val = f"{sub.mean()*100:.1f}" if len(sub) else "--"
            row.append(val)
        rows.append(row)

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Easy & Medium & Hard & Deceptive \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 2 — Per-dataset stratum statistics
# ─────────────────────────────────────────────────────────────────────────────

def table2_datasets(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{lrcccc}",
        r"\toprule",
        r"Dataset & N & Easy\% & Medium\% & Hard\% & Deceptive\% \\",
        r"\midrule",
    ]
    for ds in sorted(df["source_dataset"].unique()):
        grp = df[df["source_dataset"] == ds]
        n   = len(grp)
        fracs = [f"{len(grp[grp.stratum==s])/n*100:.1f}" for s in STRATA]
        label = DATASET_LABELS.get(ds, ds)
        lines.append(f"{label} & {n} & " + " & ".join(fracs) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(input_path: str = "results/disagreement_full.csv"):
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows")

    print("Generating figures…")
    fig1_teaser(df)
    fig2_stacked_distribution(df)
    fig3_heatmap(df)
    fig4_oracle_validation(df)

    print("Generating tables…")
    t1 = table1_accuracy(df)
    t2 = table2_datasets(df)
    for name, content in [("table1_accuracy.tex", t1), ("table2_datasets.tex", t2)]:
        for d in [FIGURES_DIR, PAPER_FIG_DIR]:
            (d / name).write_text(content)
        print(f"  Saved {name}")

    print("\nOK All figures and tables generated.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/disagreement_full.csv")
    args = parser.parse_args()
    main(args.input)



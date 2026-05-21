"""
Merge per-model CSVs, compute disagreement scores, stratify samples.
Saves disagreement_full.csv to Modal volume.

Usage:
    modal run scripts/03_compute_disagreement.py
    # or locally after downloading CSVs:
    python scripts/03_compute_disagreement.py --local
"""
import modal
import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from modal_app.common import app, image, volume, VOL_PATH


@app.function(
    image=image,
    volumes={VOL_PATH: volume},
    timeout=600,
    cpu=4,
)
def compute_disagreement_remote():
    import pandas as pd
    from modal_app.pipeline import disagreement_score
    from sentence_transformers import SentenceTransformer

    vol = Path(VOL_PATH)

    # Load all 4 model CSVs
    model_names = ["qwen", "llava", "internvl", "minicpm"]
    dfs = {}
    for name in model_names:
        path = vol / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run 02_run_all_vlms.py first.")
        dfs[name] = pd.read_csv(path)
        print(f"Loaded {name}: {len(dfs[name])} rows")

    # Merge on idx
    base = dfs["qwen"][["idx", "source_dataset", "image_id", "question", "gt"]].copy()
    for name in model_names:
        base = base.merge(
            dfs[name][["idx", "prediction", "mean_logprob", "latency_ms"]].rename(columns={
                "prediction":  f"{name}_pred",
                "mean_logprob": f"{name}_logp",
                "latency_ms":  f"{name}_ms",
            }),
            on="idx", how="left"
        )

    print(f"Merged dataframe: {len(base)} rows")

    # Load sentence-BERT for semantic clustering
    print("Loading sentence-BERT…")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    # Compute disagreement per row
    records = []
    for _, row in base.iterrows():
        preds = [str(row.get(f"{m}_pred", "") or "") for m in model_names]
        gt_raw = row["gt"]
        try:
            gt = json.loads(gt_raw)
            if isinstance(gt, (int, float)):
                gt = str(int(gt)) if float(gt) == int(gt) else str(gt)
        except (json.JSONDecodeError, TypeError):
            gt = str(gt_raw)

        ds = disagreement_score(preds, gt, sbert_model=sbert)

        rec = row.to_dict()
        rec.update({
            "disagreement":          ds["disagreement"],
            "num_clusters":          ds["num_clusters"],
            "largest_cluster_size":  ds["largest_cluster_size"],
            "stratum":               ds["stratum"],
            "qwen_correct":    ds["correctness"][0],
            "llava_correct":   ds["correctness"][1],
            "internvl_correct":ds["correctness"][2],
            "minicpm_correct": ds["correctness"][3],
        })
        records.append(rec)

    out_df = pd.DataFrame(records)

    # Add question types (rule-based, fast)
    from analysis.question_typing import classify_batch
    out_df["question_type"] = classify_batch(out_df["question"].tolist())

    out_path = vol / "disagreement_full.csv"
    out_df.to_csv(out_path, index=False)
    volume.commit()
    print(f"Saved {len(out_df)} rows → {out_path}")

    # Print stratum summary
    print("\nStratum distribution:")
    print(out_df["stratum"].value_counts().to_string())
    print("\nMean accuracy per stratum:")
    acc_cols = ["qwen_correct", "llava_correct", "internvl_correct", "minicpm_correct"]
    print(out_df.groupby("stratum")[acc_cols].mean().round(3).to_string())

    return len(out_df)


@app.local_entrypoint()
def main(local: bool = False):
    if local:
        import pandas as pd, json
        from modal_app.pipeline import disagreement_score
        from analysis.question_typing import classify_batch

        model_names = ["qwen", "llava", "internvl", "minicpm"]
        dfs = {n: pd.read_csv(f"results/{n}.csv") for n in model_names}
        base = dfs["qwen"][["idx", "source_dataset", "image_id", "question", "gt"]].copy()
        for name in model_names:
            base = base.merge(
                dfs[name][["idx", "prediction"]].rename(columns={"prediction": f"{name}_pred"}),
                on="idx", how="left"
            )
        records = []
        for _, row in base.iterrows():
            preds = [str(row.get(f"{m}_pred", "") or "") for m in model_names]
            gt_raw = row["gt"]
            try:
                gt = json.loads(gt_raw)
                if isinstance(gt, (int, float)):
                    gt = str(int(gt)) if float(gt) == int(gt) else str(gt)
            except Exception:
                gt = str(gt_raw)
            ds = disagreement_score(preds, gt)
            rec = row.to_dict()
            rec.update({"disagreement": ds["disagreement"], "stratum": ds["stratum"],
                        **{f"{m}_correct": ds["correctness"][i]
                           for i, m in enumerate(model_names)}})
            records.append(rec)
        out = pd.DataFrame(records)
        out["question_type"] = classify_batch(out["question"].tolist())
        out.to_csv("results/disagreement_full.csv", index=False)
        print(f"Saved {len(out)} rows → results/disagreement_full.csv")
    else:
        n = compute_disagreement_remote.remote()
        print(f"Done. {n} rows processed.")


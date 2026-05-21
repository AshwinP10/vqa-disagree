"""
Build and push the VQA-Disagree HuggingFace dataset.
Selects 125 samples per stratum (500 total), balancing source datasets.

Usage:
    modal run scripts/05_make_dataset.py --hf-repo YOUR_HF_USERNAME/VQA-Disagree
"""
import modal
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from modal_app.common import app, image, volume, VOL_PATH, hf_secret


@app.function(
    image=image,
    volumes={VOL_PATH: volume},
    secrets=[hf_secret],
    timeout=600,
    cpu=2,
)
def build_and_push(hf_repo: str, n_per_stratum: int = 125):
    import os, json
    import pandas as pd
    from pathlib import Path
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi, whoami

    if hf_repo == "auto":
        username = whoami(token=os.environ["HF_TOKEN"])["name"]
        hf_repo = f"{username}/VQA-Disagree"
        print(f"Auto-detected HF repo: {hf_repo}")

    vol = Path(VOL_PATH)
    df  = pd.read_csv(vol / "disagreement_full.csv")
    print(f"Loaded {len(df)} rows")

    # Sample n_per_stratum per stratum with dataset balance
    selected = []
    for stratum, grp in df.groupby("stratum"):
        n_avail = len(grp)
        n_take  = min(n_per_stratum, n_avail)
        # Try to spread across source datasets
        source_counts = grp["source_dataset"].value_counts()
        per_source = max(1, n_take // len(source_counts))
        sampled_idxs = []
        for src, src_grp in grp.groupby("source_dataset"):
            take = min(per_source, len(src_grp))
            sampled_idxs.extend(src_grp.sample(n=take, random_state=42).index.tolist())
        # Top up if needed
        remaining = [i for i in grp.index if i not in set(sampled_idxs)]
        extra = n_take - len(sampled_idxs)
        if extra > 0 and remaining:
            sampled_idxs.extend(remaining[:extra])
        selected.extend(sampled_idxs[:n_take])

    subset = df.loc[selected].copy().reset_index(drop=True)
    print(f"Selected {len(subset)} samples:")
    print(subset["stratum"].value_counts().to_string())

    # Build HF Dataset
    records = []
    for _, row in subset.iterrows():
        records.append({
            "question":       row["question"],
            "ground_truth":   row["gt"],
            "source_benchmark": row["source_dataset"],
            "image_id":       str(row["image_id"]),
            "question_type":  row.get("question_type", "other"),
            "qwen_pred":      str(row.get("qwen_pred", "")),
            "llava_pred":     str(row.get("llava_pred", "")),
            "internvl_pred":  str(row.get("internvl_pred", "")),
            "minicpm_pred":   str(row.get("minicpm_pred", "")),
            "qwen_conf":      float(row["qwen_logp"]) if pd.notna(row.get("qwen_logp")) else None,
            "llava_conf":     float(row["llava_logp"]) if pd.notna(row.get("llava_logp")) else None,
            "internvl_conf":  None,
            "minicpm_conf":   None,
            "disagreement_score": float(row["disagreement"]),
            "stratum":        row["stratum"],
            "oracle_correct":  float(row["oracle_correct"]) if pd.notna(row.get("oracle_correct")) else None,
        })

    hf_dataset = Dataset.from_list(records)
    dd = DatasetDict({"test": hf_dataset})

    print(f"Pushing to {hf_repo}…")
    dd.push_to_hub(hf_repo, token=os.environ["HF_TOKEN"])
    print(f"OK VQA-Disagree pushed to https://huggingface.co/datasets/{hf_repo}")

    # Also save locally in volume
    subset.to_csv(vol / "vqa_disagree_500.csv", index=False)
    volume.commit()
    return len(subset)


@app.local_entrypoint()
def main(hf_repo: str = "auto", n_per_stratum: int = 125):
    n = build_and_push.remote(hf_repo=hf_repo, n_per_stratum=n_per_stratum)
    print(f"Done. {n} samples in VQA-Disagree.")


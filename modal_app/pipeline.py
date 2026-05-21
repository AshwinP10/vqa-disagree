"""
Data loading and disagreement scoring pipeline for VQA-Disagree.

Key functions:
  load_dataset_samples()  — Modal function: loads all 5 datasets, returns serialized items
  disagreement_score()    — pure Python: computes per-sample disagreement dict
  stratify()              — assigns stratum label from disagreement dict
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from modal_app.common import app, image, volume, VOL_PATH, hf_secret

# ─────────────────────────────────────────────────────────────────────────────
# Disagreement score (pure Python — also used locally in scripts/03_*)
# ─────────────────────────────────────────────────────────────────────────────

def disagreement_score(predictions: list[str], gt: str | list[str],
                       sbert_model=None) -> dict:
    """
    Compute disagreement score for a set of 4 VLM predictions.

    Returns dict with keys:
      disagreement, num_clusters, largest_cluster_size,
      correctness, is_easy, is_deceptive, is_hard, is_medium, stratum
    """
    from analysis.normalize import vqa_soft_match
    from analysis.clustering import compute_clusters

    clusters = compute_clusters(predictions, sbert_model=sbert_model)
    n = len(predictions)
    largest = max(len(c) for c in clusters)
    disagree = 1.0 - (largest / n)

    correctness = [vqa_soft_match(p, gt) for p in predictions]
    all_correct  = all(c > 0 for c in correctness)
    none_correct = all(c == 0 for c in correctness)

    is_easy       = (largest == n and all_correct)
    is_deceptive  = (largest == n and none_correct)
    is_hard       = (disagree >= 0.5)
    is_medium     = (disagree == 0.25)

    if is_easy:
        stratum = "easy"
    elif is_deceptive:
        stratum = "deceptive"
    elif is_hard:
        stratum = "hard"
    elif is_medium:
        stratum = "medium"
    else:
        stratum = "easy"   # disagree==0 but mixed correctness → treat as easy

    return {
        "disagreement":         round(disagree, 4),
        "num_clusters":         len(clusters),
        "largest_cluster_size": largest,
        "correctness":          correctness,
        "is_easy":       is_easy,
        "is_deceptive":  is_deceptive,
        "is_hard":       is_hard,
        "is_medium":     is_medium,
        "stratum":       stratum,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader (runs inside Modal container — has HF internet access)
# ─────────────────────────────────────────────────────────────────────────────

# Tuple: (hf_path, hf_config_name_or_None, split, max_n)
# GQA needs an explicit config name; TextVQA replaces the unavailable lmms-lab/MMVP.
DATASET_CONFIGS = {
    "vqav2":      ("lmms-lab/VQAv2",           None,                           "validation", 500),
    "gqa":        ("lmms-lab/GQA",             "testdev_balanced_instructions", "testdev",    500),
    "textvqa":    ("lmms-lab/textvqa",          None,                           "validation", 300),
    "chartqa":    ("HuggingFaceM4/ChartQA",     None,                           "val",        500),
    "realworldqa":("xai-org/RealWorldQA",       None,                           "test",       200),
}


def _get_image_bytes(row: dict, dataset_key: str) -> bytes | None:
    """Extract image bytes from a dataset row (handles various field formats)."""
    from io import BytesIO
    from PIL import Image as PILImage

    pil = None
    for field in ("image", "img", "Image"):
        val = row.get(field)
        if val is None:
            continue
        if isinstance(val, PILImage.Image):
            pil = val
            break
        if isinstance(val, dict) and "bytes" in val:
            pil = PILImage.open(BytesIO(val["bytes"]))
            break
        if isinstance(val, bytes):
            pil = PILImage.open(BytesIO(val))
            break

    if pil is None:
        return None
    pil = pil.convert("RGB")
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _extract_gt(row: dict, dataset_key: str):
    """Return ground truth as string or list of strings."""
    if dataset_key in ("vqav2", "textvqa"):
        answers = row.get("answers", [])
        if isinstance(answers, list) and len(answers) > 0:
            if isinstance(answers[0], dict):
                return [a["answer"] for a in answers]
            return [str(a) for a in answers]
        return str(row.get("multiple_choice_answer", ""))
    # Single-answer datasets
    for field in ("answer", "label", "Answer"):
        val = row.get(field)
        if val is not None:
            return str(val)
    return ""


def _extract_image_id(row: dict, dataset_key: str, fallback_idx: int) -> str:
    for field in ("question_id", "image_id", "imageId", "id", "idx", "image_name"):
        val = row.get(field)
        if val is not None:
            return str(val)
    return f"{dataset_key}_{fallback_idx}"


def _extract_question(row: dict, dataset_key: str) -> str:
    for field in ("question", "query", "Question"):
        val = row.get(field)
        if val is not None:
            return str(val)
    return ""


@app.function(
    image=image,
    volumes={VOL_PATH: volume},
    secrets=[hf_secret],
    timeout=600,
    cpu=4,
)
def load_dataset_samples(dataset_keys: list[str] | None = None) -> list[dict]:
    """
    Load samples from all (or selected) source datasets.
    Returns list of item dicts ready for VLM inference.
    Each item has: idx, source_dataset, image_id, question, gt, image_bytes.
    """
    import json
    from datasets import load_dataset

    keys = dataset_keys or list(DATASET_CONFIGS.keys())
    all_items: list[dict] = []
    global_idx = 0

    for key in keys:
        hf_path, hf_name, split, max_n = DATASET_CONFIGS[key]
        print(f"[loader] Loading {key} ({hf_path}, name={hf_name}, split={split}, max={max_n})")

        try:
            load_kwargs = dict(split=split, streaming=True)
            if hf_name:
                load_kwargs["name"] = hf_name
            ds = load_dataset(hf_path, **load_kwargs)
        except Exception as e:
            print(f"[loader] Failed to load {key}: {e}. Skipping.")
            continue

        count = 0
        for row in ds:
            if count >= max_n:
                break
            img_bytes = _get_image_bytes(row, key)
            if img_bytes is None:
                continue
            question = _extract_question(row, key)
            gt       = _extract_gt(row, key)
            img_id   = _extract_image_id(row, key, count)

            all_items.append({
                "idx":            global_idx,
                "source_dataset": key,
                "image_id":       img_id,
                "question":       question,
                "gt":             json.dumps(gt) if isinstance(gt, list) else gt,
                "image_bytes":    img_bytes,
            })
            global_idx += 1
            count += 1

        print(f"[loader]   → loaded {count} samples (total so far: {global_idx})")

    print(f"[loader] Total samples: {len(all_items)}")
    return all_items


# ─────────────────────────────────────────────────────────────────────────────
# Volume I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_csv_to_volume(rows: list[dict], filename: str):
    import pandas as pd
    path = Path(VOL_PATH) / filename
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    volume.commit()
    print(f"[pipeline] Saved {len(df)} rows → {path}")


def load_csv_from_volume(filename: str):
    import pandas as pd
    path = Path(VOL_PATH) / filename
    return pd.read_csv(path)

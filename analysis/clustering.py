"""
Semantic clustering of VLM predictions using sentence-BERT + exact-match fallback.
Used by the disagreement score computation.
"""
from __future__ import annotations
from analysis.normalize import normalize_answer


def cluster_by_exact_match(normalized: list[str]) -> list[list[int]]:
    """Return clusters as lists of indices grouped by identical normalized string."""
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(normalized):
        groups.setdefault(s, []).append(i)
    return list(groups.values())


def merge_with_sbert(clusters: list[list[int]], raw_preds: list[str],
                     threshold: float = 0.8, model=None) -> list[list[int]]:
    """
    Merge clusters whose centroid sentence embeddings exceed `threshold`.
    `model` is a SentenceTransformer instance passed in to avoid re-loading.
    Falls back to no-op merge if model is None.
    """
    if model is None or len(clusters) <= 1:
        return clusters

    import numpy as np

    # One representative string per cluster (first raw pred in cluster)
    reps = [raw_preds[c[0]] for c in clusters]
    embeddings = model.encode(reps, normalize_embeddings=True)

    merged = list(range(len(clusters)))  # merge[i] = canonical cluster id

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if merged[j] != j:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= threshold:
                # merge j into i
                for k in range(len(merged)):
                    if merged[k] == j:
                        merged[k] = i

    result: dict[int, list[int]] = {}
    for old_idx, canonical in enumerate(merged):
        result.setdefault(canonical, []).extend(clusters[old_idx])
    return list(result.values())


def compute_clusters(predictions: list[str], sbert_model=None,
                     threshold: float = 0.8) -> list[list[int]]:
    """Full clustering pipeline: exact-match first, then sbert merge."""
    normalized = [normalize_answer(p) for p in predictions]
    clusters = cluster_by_exact_match(normalized)
    clusters = merge_with_sbert(clusters, predictions, threshold, sbert_model)
    return clusters

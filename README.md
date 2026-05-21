# VQA-Disagree

**Multi-Model Disagreement as an Annotation-Free Difficulty Signal for VQA Benchmarks**

A reproducible pipeline that scores 1,500 VQA samples from four public benchmarks for *multi-model disagreement* — the fraction of distinct answer clusters produced by an ensemble of four architecturally diverse 7B VLMs — and stratifies them into four difficulty buckets (Easy, Medium, Hard, Deceptive) with no human annotation. Released as a 456-sample stratified dataset on [HuggingFace Hub](https://huggingface.co/datasets/AshwinP10/VQA-Disagree).

Paper: submitted to the **DataMFM Workshop @ CVPR 2026** (non-archival).

## Key results

| Stratum   | n (of 1,500) | Per-model acc. (4 × 7B) | Qwen2-VL-72B oracle acc. |
|---|---:|---:|---:|
| Easy      | 81  (5.4%)   | 92.6%–96.3% | 38.3% |
| Medium    | 493 (32.9%)  | 6.1%–9.1%   | 7.0%  |
| Hard      | 752 (50.1%)  | 2.0%–2.7%   | 10.0% |
| Deceptive | 174 (11.6%)  | 0.0%        | **1.7%** |

The 36.6 pp Easy–Deceptive oracle gap (38.3% vs. 1.7%) validates multi-model disagreement as a real difficulty signal: the Deceptive stratum — where all four 7B models agree and are wrong — is also the hardest stratum for the 72B oracle.

## Models in the ensemble

| Tag | Model | Visual encoder | LM backbone |
|---|---|---|---|
| `qwen`     | Qwen2-VL-7B-Instruct     | Dynamic ViT     | Qwen2-7B |
| `llava`    | LLaVA-1.6-Mistral-7B     | CLIP ViT-L      | Mistral-7B |
| `internvl` | InternVL2-8B             | InternViT-300M  | InternLM2-7B |
| `minicpm`  | MiniCPM-V 2.6            | SigLIP          | MiniCPM-3B |

Oracle (validation only, not part of the ensemble): **Qwen2-VL-72B-Instruct** (4-bit NF4 via QLoRA, A100-80GB).

## Source benchmarks (1,500 samples total)

| Dataset      | Split        |    N |
|---|---|---:|
| VQAv2        | validation   |  500 |
| TextVQA      | validation   |  300 |
| ChartQA      | validation   |  500 |
| RealWorldQA  | test         |  200 |

## Pipeline

1. `scripts/02_run_all_vlms.py` — runs the four 7B VLMs on all 1,500 samples (Modal, A100-40GB).
2. `scripts/03_compute_disagreement.py` — merges per-model CSVs, computes the disagreement score `D = 1 − k*/4` over SBERT-clustered predictions (threshold 0.8), assigns each sample to a stratum.
3. `scripts/04_gpt4o_oracle.py` — runs Qwen2-VL-72B on a stratified 281-item sample (Easy/Medium/Hard). (Filename kept for historical reasons; the oracle is Qwen2-VL-72B, not GPT-4o.)
4. `scripts/05_make_dataset.py` — builds the 456-sample HF release.
5. `scripts/06_compute_metrics.py` — recomputes all summary statistics into `results/metrics_summary.json`.
6. `scripts/07_make_figures.py` — regenerates all paper figures from `results/disagreement_full.csv`.
7. `scripts/08_oracle_deceptive.py` — runs the 72B oracle on all 174 Deceptive items (extends oracle coverage to 455 total).
8. `scripts/09_sbert_sensitivity.py` — re-clusters at SBERT thresholds {0.70, 0.85, 0.90} for the stability check (84.8% label preservation at ±0.05 of default 0.80).
9. `scripts/10_merge_and_analyze.py` — merges new oracle results back into `results/disagreement_full.csv` (preserving existing values).

## Repository layout

```
modal_app/        Modal app: 4 ensemble VLM classes + Qwen2-VL-72B oracle + dataset loader
scripts/          Run scripts (numbered in execution order)
analysis/         Pure-Python utilities (VQA normalization, SBERT clustering, question typing, stats)
results/          All produced CSVs + metrics_summary.json (reproducibility)
paper/            CVPR 2026 LaTeX source + figures + bib + style files
figures/          Mirror of paper/figures used by 07_make_figures.py
```

## Setup

Requires Python 3.11+ and a Modal account for GPU inference. Inference is dispatched in fire-and-forget mode so you can close the laptop after launching.

```bash
pip install -e .
modal token new                                    # one-time
modal secret create huggingface HF_TOKEN=hf_xxx    # for gated models
```

Each Modal script can be launched with `modal run scripts/<name>.py`.

## License & citation

Code: MIT.
Dataset: see HuggingFace Hub card for the release license.

If you use this work, please cite the DataMFM @ CVPR 2026 paper (citation will be added after the workshop).

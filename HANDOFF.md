# VQA-Disagree — Handoff Document (2026-05-11)

## What this project is

4-page non-archival short paper for **CVPR 2026 DataMFM Workshop**.
Deadline: **May 20 2026 23:59 UTC**.

Core idea: run 4 architecturally diverse 7B VLMs on 2,000 VQA samples, use
multi-model disagreement to stratify samples into Easy/Medium/Hard/Deceptive
difficulty strata, validate with a scale-based oracle (Qwen2.5-VL-72B), and
release a 500-sample HuggingFace dataset.

## Current status: SMOKE TEST PASSED — ready for full inference

All 4 models confirmed working on 5-sample smoke test:
- Qwen2-VL-7B-Instruct       OK
- LLaVA-1.6-Mistral-7B       OK
- InternVL2-8B                OK
- MiniCPM-V-2.6               OK (gated repo — user already accepted terms)

No OpenAI/Anthropic keys needed. Only credentials required:
- HuggingFace token (already in Modal secret named `huggingface`)

## Next steps (run in this order)

### 1. Full VLM inference — ~$10, ~6 GPU-hours
```
cd C:\Users\ashwi\OneDrive\Desktop\coding_files\research_paper\vqa-disagree
modal run scripts/02_run_all_vlms.py
```
**Safe to close laptop** — orchestration runs inside Modal.
Saves: `qwen.csv`, `llava.csv`, `internvl.csv`, `minicpm.csv` to Modal volume `vqa-disagree-data`.

Dry-run first if you want to sanity-check cheaply (~$0.10):
```
modal run scripts/02_run_all_vlms.py --dry-run
```

### 2. Disagreement scoring — free (CPU only)
```
modal run scripts/03_compute_disagreement.py
```
Saves: `disagreement_full.csv` to volume.

### 3. Scale-based oracle validation — ~$1, A100-80GB
```
modal run scripts/04_gpt4o_oracle.py
```
Uses Qwen2.5-VL-72B-Instruct (4-bit NF4 quant, auto-fallback to 32B).
Adds `oracle_pred` + `oracle_correct` columns to `disagreement_full.csv`.
Target: Easy-Hard accuracy gap >= 15 percentage points.

### 4. Build HuggingFace dataset
```
modal run scripts/05_make_dataset.py --hf-repo YOUR_HF_NAME/VQA-Disagree
```

### 5. Download results + compute metrics + figures (local)
```
modal volume get vqa-disagree-data / ./results/
python scripts/06_compute_metrics.py
python scripts/07_make_figures.py
```

### 6. Fill in the paper
All real numbers go into `paper/main.tex`. Search for `\FILL` — every placeholder
is marked. Figures land in `figures/` after step 5.

### 7. Submit
OpenReview, before May 20 2026 23:59 UTC.

## Budget summary
| Step | Cost |
|------|------|
| Full VLM inference (4 models x 2K samples) | ~$10 |
| Oracle (Qwen2.5-72B, 300 samples, A100-80GB) | ~$1 |
| Buffer | ~$1.50 |
| **Total** | **~$12.50** |

## Key technical details

**Disagreement score:** D = 1 - (largest_cluster_size / 4), range [0, 1]

**Strata:**
- Easy: D=0, all 4 models correct
- Deceptive: D=0, all 4 models wrong
- Medium: D=0.25 (one model disagrees)
- Hard: D>=0.5 (models split 50/50 or worse)

**Semantic clustering:** exact-match normalization first, then sentence-BERT
(all-MiniLM-L6-v2) cosine similarity >= 0.8 to merge clusters.

**VQA soft accuracy:** min(matches/3, 1.0) for VQAv2 (10 annotator answers);
normalized exact match for GQA/MMVP/ChartQA/RealWorldQA.

**Oracle:** Qwen2.5-VL-72B-Instruct, 4-bit NF4 quant (bitsandbytes), A100-80GB.
Same model family as Qwen2-VL in ensemble — paper frames this as "scale-based
oracle": hardness persisting at 10x scale = structural task difficulty, not
architectural quirk.

## Source datasets (2,000 samples total)
| Dataset | HF path | Split | N |
|---------|---------|-------|---|
| VQAv2 | lmms-lab/VQAv2 | validation | 500 |
| GQA | lmms-lab/GQA | testdev_balanced | 500 |
| MMVP | lmms-lab/MMVP | test | ~300 |
| ChartQA | HuggingFaceM4/ChartQA | val | 500 |
| RealWorldQA | xai-org/RealWorldQA | test | 200 |

## Known non-issues (do not panic about these)
- `trust_remote_code` warning for lmms-lab/VQAv2 — harmless, dataset loads fine
- "A new version of files downloaded" warnings from InternVL2/MiniCPM — harmless
- InternVL2 and MiniCPM have `mean_logprob=None` in CSVs — expected, their
  `chat()` API does not expose token scores
- FlashAttention2 not installed warning — using eager attention, fine for correctness

## User preferences (important)
- **Store results extensively** — save every intermediate CSV. Never overwrite.
  If re-running a step, save to a new filename (e.g. `qwen_v2.csv`).
- **Be cost-conscious** — confirm before launching any run over ~$2.
- **Laptop-close-safe** — `02_run_all_vlms.py` was restructured so orchestration
  runs inside Modal. All subsequent scripts also use Modal functions for heavy work.
  The user can close their laptop once `modal run` dispatches the job.

## File structure
```
vqa-disagree/
  modal_app/
    common.py        # Modal app, image (includes sentencepiece+protobuf), volume, hf_secret
    inference.py     # QwenVLM, LLaVAVLM, InternVLM, MiniCPMVLM, Qwen72BVLM
    pipeline.py      # load_dataset_samples(), disagreement_score(), DATASET_CONFIGS
  analysis/
    normalize.py     # vqa_soft_match(), normalize_answer()
    clustering.py    # compute_clusters() with sentence-BERT
    question_typing.py  # rule-based 7-type classifier
    stats.py         # chi-square, bootstrap CI
  scripts/
    01_smoke_test.py      # DONE — all 4 models passed
    01b_smoke_minicpm.py  # minimal MiniCPM-only check (already passed)
    02_run_all_vlms.py    # NEXT — full inference, laptop-close-safe
    03_compute_disagreement.py
    04_gpt4o_oracle.py    # uses Qwen2.5-VL-72B, not GPT-4o
    05_make_dataset.py
    06_compute_metrics.py
    07_make_figures.py
  paper/
    main.tex         # 4-page CVPR template, double-blind, all \FILL placeholders
    refs.bib         # 20 real citations
  HANDOFF.md         # this file
  README.md
```

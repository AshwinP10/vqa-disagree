"""
Four VLM inference classes for VQA-Disagree.
Each class: loads one model, exposes run_batch(items) -> results.

item dict keys (input):
    idx, source_dataset, image_id, question, gt, image_bytes (JPEG bytes)

result dict keys (output):
    idx, source_dataset, image_id, question, gt,
    prediction, mean_logprob, latency_ms, tokens
"""
from __future__ import annotations
import modal
from modal_app.common import app, image, volume, VOL_PATH, HF_CACHE, hf_secret

_GPU   = "A100-40GB"
_TO    = 3600
_SCALE = 300

# ─────────────────────────────────────────────────────────────────────────────
# Helpers (run inside container)
# ─────────────────────────────────────────────────────────────────────────────

def _decode_image(image_bytes: bytes):
    from io import BytesIO
    from PIL import Image
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def _mean_logprob(scores, gen_ids) -> float | None:
    """Compute mean log-probability from HF generate() output_scores."""
    try:
        import torch
        import torch.nn.functional as F
        import numpy as np
        lps = []
        for score, tok in zip(scores, gen_ids):
            lp = F.log_softmax(score[0], dim=-1)[tok].item()
            lps.append(lp)
        return float(np.mean(lps)) if lps else None
    except Exception:
        return None


def _base_result(item: dict, prediction: str, logprob, latency_ms: float, tokens: int) -> dict:
    return {
        "idx":            item["idx"],
        "source_dataset": item["source_dataset"],
        "image_id":       item["image_id"],
        "question":       item["question"],
        "gt":             item["gt"],
        "prediction":     prediction,
        "mean_logprob":   logprob,
        "latency_ms":     round(latency_ms, 1),
        "tokens":         tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Qwen2-VL-7B-Instruct
# ─────────────────────────────────────────────────────────────────────────────

@app.cls(
    gpu=_GPU, timeout=_TO, scaledown_window=_SCALE,
    image=image, volumes={VOL_PATH: volume}, secrets=[hf_secret],
)
class QwenVLM:
    @modal.enter()
    def load(self):
        import os, torch
        os.environ["HF_HOME"] = HF_CACHE
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
        ).eval()

    @modal.method()
    def run_batch(self, items: list[dict]) -> list[dict]:
        import time, torch
        results = []
        for item in items:
            t0 = time.time()
            try:
                pil = _decode_image(item["image_bytes"])
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": pil},
                    {"type": "text",  "text":  item["question"]},
                ]}]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                img_inputs, _ = self.process_vision_info(messages)
                inputs = self.processor(
                    text=[text], images=img_inputs, return_tensors="pt"
                ).to("cuda")

                with torch.no_grad():
                    out = self.model.generate(
                        **inputs, max_new_tokens=64, do_sample=False,
                        output_scores=True, return_dict_in_generate=True,
                    )
                in_len = inputs["input_ids"].shape[1]
                gen_ids = out.sequences[0][in_len:]
                pred  = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
                lp    = _mean_logprob(out.scores, gen_ids)
                toks  = len(gen_ids)
            except Exception as e:
                print(f"[qwen] idx={item['idx']} error: {e}")
                pred, lp, toks = "", None, 0
            results.append(_base_result(item, pred, lp, (time.time() - t0) * 1000, toks))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# LLaVA-1.6-Mistral-7B
# ─────────────────────────────────────────────────────────────────────────────

@app.cls(
    gpu=_GPU, timeout=_TO, scaledown_window=_SCALE,
    image=image, volumes={VOL_PATH: volume}, secrets=[hf_secret],
)
class LLaVAVLM:
    @modal.enter()
    def load(self):
        import os, torch
        os.environ["HF_HOME"] = HF_CACHE
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        self.processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf"
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
        ).eval()

    @modal.method()
    def run_batch(self, items: list[dict]) -> list[dict]:
        import time, torch
        results = []
        for item in items:
            t0 = time.time()
            try:
                pil = _decode_image(item["image_bytes"])
                conv = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": item["question"]},
                ]}]
                prompt = self.processor.apply_chat_template(conv, add_generation_prompt=True)
                inputs = self.processor(
                    images=[pil], text=prompt, return_tensors="pt"
                ).to("cuda")

                with torch.no_grad():
                    out = self.model.generate(
                        **inputs, max_new_tokens=64, do_sample=False,
                        output_scores=True, return_dict_in_generate=True,
                    )
                in_len = inputs["input_ids"].shape[1]
                gen_ids = out.sequences[0][in_len:]
                pred  = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
                lp    = _mean_logprob(out.scores, gen_ids)
                toks  = len(gen_ids)
            except Exception as e:
                print(f"[llava] idx={item['idx']} error: {e}")
                pred, lp, toks = "", None, 0
            results.append(_base_result(item, pred, lp, (time.time() - t0) * 1000, toks))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# InternVL2-8B
# ─────────────────────────────────────────────────────────────────────────────

@app.cls(
    gpu=_GPU, timeout=_TO, scaledown_window=_SCALE,
    image=image, volumes={VOL_PATH: volume}, secrets=[hf_secret],
)
class InternVLM:
    @modal.enter()
    def load(self):
        import os, torch
        os.environ["HF_HOME"] = HF_CACHE
        import torchvision.transforms as T
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            "OpenGVLab/InternVL2-8B", trust_remote_code=True, use_fast=False
        )
        self.model = AutoModel.from_pretrained(
            "OpenGVLab/InternVL2-8B",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().cuda()

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @modal.method()
    def run_batch(self, items: list[dict]) -> list[dict]:
        import time, torch
        gen_cfg = dict(max_new_tokens=64, do_sample=False)
        results = []
        for item in items:
            t0 = time.time()
            try:
                pil = _decode_image(item["image_bytes"])
                pixel_values = self.transform(pil).unsqueeze(0).to(torch.bfloat16).cuda()
                question = f"<image>\n{item['question']}"
                pred = self.model.chat(
                    self.tokenizer, pixel_values, question, gen_cfg
                ).strip()
                lp   = None   # InternVL2 chat() does not expose scores
                toks = 0
            except Exception as e:
                print(f"[internvl] idx={item['idx']} error: {e}")
                pred, lp, toks = "", None, 0
            results.append(_base_result(item, pred, lp, (time.time() - t0) * 1000, toks))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# MiniCPM-V-2.6
# ─────────────────────────────────────────────────────────────────────────────

@app.cls(
    gpu=_GPU, timeout=_TO, scaledown_window=_SCALE,
    image=image, volumes={VOL_PATH: volume}, secrets=[hf_secret],
)
class MiniCPMVLM:
    @modal.enter()
    def load(self):
        import os, torch
        os.environ["HF_HOME"] = HF_CACHE
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            "openbmb/MiniCPM-V-2_6", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).eval().cuda()

    @modal.method()
    def run_batch(self, items: list[dict]) -> list[dict]:
        import time
        results = []
        for item in items:
            t0 = time.time()
            try:
                pil  = _decode_image(item["image_bytes"])
                msgs = [{"role": "user", "content": [pil, item["question"]]}]
                pred = self.model.chat(
                    image=None, msgs=msgs, tokenizer=self.tokenizer,
                    sampling=False, max_new_tokens=64,
                ).strip()
                lp   = None
                toks = 0
            except Exception as e:
                print(f"[minicpm] idx={item['idx']} error: {e}")
                pred, lp, toks = "", None, 0
            results.append(_base_result(item, pred, lp, (time.time() - t0) * 1000, toks))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Qwen2.5-VL-72B  (scale-based oracle — NOT part of the 4-model ensemble)
# Uses 4-bit NF4 quantization via bitsandbytes.
# Primary:  Qwen/Qwen2.5-VL-72B-Instruct  on A100-80GB
# Fallback: Qwen/Qwen2.5-VL-32B-Instruct  on A100-40GB if 80GB unavailable
# ─────────────────────────────────────────────────────────────────────────────

_ORACLE_MODEL   = "Qwen/Qwen2-VL-72B-Instruct"   # works with transformers==4.46
_FALLBACK_MODEL = "Qwen/Qwen2-VL-7B-Instruct"    # same family, lighter fallback

@app.cls(
    gpu="A100-80GB",
    timeout=7200,
    scaledown_window=300,
    image=image,
    volumes={VOL_PATH: volume},
    secrets=[hf_secret],
)
class Qwen72BVLM:
    @modal.enter()
    def load(self):
        import os, torch
        os.environ["HF_HOME"] = HF_CACHE
        from transformers import (
            Qwen2VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        from qwen_vl_utils import process_vision_info

        self.process_vision_info = process_vision_info

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # Try 72B first; fall back to 32B if loading fails
        for model_id in [_ORACLE_MODEL, _FALLBACK_MODEL]:
            try:
                print(f"[oracle] Loading {model_id} with 4-bit quant…")
                self.processor = AutoProcessor.from_pretrained(
                    model_id,
                    min_pixels=256 * 28 * 28,
                    max_pixels=1280 * 28 * 28,
                )
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    attn_implementation="eager",
                ).eval()
                self.model_id = model_id
                print(f"[oracle] Loaded {model_id}")
                break
            except Exception as e:
                print(f"[oracle] Failed to load {model_id}: {e}")
                if model_id == _FALLBACK_MODEL:
                    raise

    @modal.method()
    def run_batch(self, items: list[dict]) -> list[dict]:
        import time, torch
        results = []
        for item in items:
            t0 = time.time()
            try:
                pil = _decode_image(item["image_bytes"])
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": pil},
                    {"type": "text",  "text":  item["question"]},
                ]}]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                img_inputs, _ = self.process_vision_info(messages)
                inputs = self.processor(
                    text=[text], images=img_inputs, return_tensors="pt"
                ).to("cuda")

                with torch.no_grad():
                    out = self.model.generate(
                        **inputs, max_new_tokens=64, do_sample=False,
                        output_scores=True, return_dict_in_generate=True,
                    )
                in_len  = inputs["input_ids"].shape[1]
                gen_ids = out.sequences[0][in_len:]
                pred    = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
                lp      = _mean_logprob(out.scores, gen_ids)
                toks    = len(gen_ids)
            except Exception as e:
                print(f"[oracle72b] idx={item['idx']} error: {e}")
                pred, lp, toks = "", None, 0
            results.append(_base_result(item, pred, lp, (time.time() - t0) * 1000, toks))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CLASSES = {
    "qwen":     QwenVLM,
    "llava":    LLaVAVLM,
    "internvl": InternVLM,
    "minicpm":  MiniCPMVLM,
    "oracle":   Qwen72BVLM,   # scale-based oracle only — not part of ensemble
}

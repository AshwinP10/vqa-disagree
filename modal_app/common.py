import modal

app = modal.App("vqa-disagree")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "ffmpeg", "libsm6", "libxext6")
    .pip_install(
        "torch==2.4.0",
        "torchvision",
        "transformers==4.46.0",
        "accelerate",
        "bitsandbytes>=0.43.0",
        "qwen-vl-utils",
        "pillow",
        "datasets",
        "huggingface_hub",
        "sentence-transformers",
        "spacy",
        "matplotlib",
        "seaborn",
        "numpy",
        "scipy",
        "pandas",
        "einops",
        "timm",
        "sentencepiece",
        "protobuf",
    )
    .run_commands("python -m spacy download en_core_web_sm")
    .add_local_python_source("modal_app", "analysis")
)

volume = modal.Volume.from_name("vqa-disagree-data", create_if_missing=True)
VOL_PATH = "/vol"
HF_CACHE = "/vol/hf_cache"   # model weights cached here; persists across container restarts

hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])

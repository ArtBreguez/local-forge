"""Upload the trained model (merged safetensors + LoRA adapter, and the Q4_K_M GGUF) from the lf-lora volume to the Hugging
Face Hub, straight from Modal (the VM has no room for the 18 GB merged model). Private repos; the owner flips visibility.

    modal run tools/modal_hf_upload.py --tag full
"""
from __future__ import annotations

import json
import pathlib

import modal

_HERE = pathlib.Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent  # inside the Modal container the file sits at /root
vol = modal.Volume.from_name("lf-lora", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub>=1.31,<3")
app = modal.App("lf-hf-upload", image=image)
REPO = "HiddenTerps/LocalForge-9B-REA-heretic"
BASE = "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic"


@app.function(volumes={"/out": vol}, secrets=[modal.Secret.from_name("hf-write")], timeout=3 * 3600, cpu=4, memory=16384)
def upload(tag: str, card: str, gguf_card: str) -> dict:
    from huggingface_hub import HfApi
    api = HfApi()
    out = {}
    api.create_repo(REPO, private=True, exist_ok=True)
    api.upload_folder(repo_id=REPO, folder_path=f"/out/{tag}/merged", commit_message="merged weights (base + REA LoRA)")
    api.upload_folder(repo_id=REPO, folder_path=f"/out/{tag}/adapter", path_in_repo="lora-adapter", commit_message="LoRA adapter")
    api.upload_file(repo_id=REPO, path_or_fileobj=card.encode(), path_in_repo="README.md", commit_message="model card")
    out["model"] = f"https://huggingface.co/{REPO}"
    g = REPO + "-GGUF"
    api.create_repo(g, private=True, exist_ok=True)
    api.upload_file(repo_id=g, path_or_fileobj=f"/out/{tag}/gguf/lora-Q4_K_M.gguf", path_in_repo="LocalForge-9B-REA-heretic-Q4_K_M.gguf", commit_message="Q4_K_M GGUF")
    api.upload_file(repo_id=g, path_or_fileobj=gguf_card.encode(), path_in_repo="README.md", commit_message="model card")
    out["gguf"] = f"https://huggingface.co/{g}"
    out["files"] = {REPO: [f.path for f in api.list_repo_tree(REPO, recursive=True)][:40], g: [f.path for f in api.list_repo_tree(g)]}
    return out


@app.local_entrypoint()
def main(tag: str = "full"):
    card = (ROOT / "results/phase6/lora/model-card.md").read_text()
    gguf_card = (ROOT / "results/phase6/lora/model-card-gguf.md").read_text()
    r = upload.remote(tag, card, gguf_card)
    print(json.dumps(r, indent=1))

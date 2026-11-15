from .config import HF_REPO, MODELS_DIR, ONNX_DIR


def ensure_models():
    if (ONNX_DIR / "model_int8.onnx").exists():
        return
    if HF_REPO.startswith("YOUR_HF_USERNAME"):
        raise SystemExit("No local models found and HF_REPO is not set in src/config.py.\n"
                         "Either train the models (see README) or set HF_REPO to the weights repository.")
    from huggingface_hub import snapshot_download
    print(f"Downloading model weights from https://huggingface.co/{HF_REPO} ...")
    snapshot_download(repo_id=HF_REPO, local_dir=str(MODELS_DIR))

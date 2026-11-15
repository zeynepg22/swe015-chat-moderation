import sys
from pathlib import Path

from huggingface_hub import HfApi

repo_id = sys.argv[1]
models = Path(__file__).resolve().parent.parent / "models"
api = HfApi()
api.create_repo(repo_id, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=str(models), repo_id=repo_id, repo_type="model", ignore_patterns=["bert/*"])
print(f"Uploaded to https://huggingface.co/{repo_id} - now set HF_REPO in src/config.py")

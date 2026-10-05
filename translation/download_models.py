
import os
import shutil
from huggingface_hub import snapshot_download

def download_model(repo_id, local_dir):
    print(f"Downloading {repo_id} to {local_dir}...")
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"Downloaded {repo_id} to {local_dir}")

if __name__ == "__main__":
    # Download NLLB model (Stage A)
    download_model("facebook/nllb-200-3.3B", "./nllb_model")
    # Download Qwen model (Stage B)
    download_model("Qwen/Qwen2.5-7B-Instruct", "./qwen_model")

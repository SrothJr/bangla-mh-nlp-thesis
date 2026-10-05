
import os
import shutil

repo_id = "gpt2"
snapshot_hash = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
blobs_dir = os.path.join(os.environ['USERPROFILE'], '.cache', 'huggingface', 'hub', f'models--{repo_id.replace("/", "--")}', 'blobs')
snapshot_dir = os.path.join(os.environ['USERPROFILE'], '.cache', 'huggingface', 'hub', f'models--{repo_id.replace("/", "--")}', 'snapshots', snapshot_hash)

print("Blobs in blobs_dir:", os.listdir(blobs_dir))

# Copy all blobs to snapshot_dir with their expected filenames? Wait no, let's download the actual file listing from the repo!
# Let's use hf_hub_download with local_dir to download directly to a local directory! That's easier!

from huggingface_hub import snapshot_download

print("Downloading gpt2 directly to local_dir='./gpt2'...")
local_dir = "./gpt2"
if os.path.exists(local_dir):
    shutil.rmtree(local_dir)
snapshot_download(repo_id="gpt2", local_dir=local_dir, local_dir_use_symlinks=False)  # Disable symlinks!

print("Files in local_dir:", os.listdir(local_dir))

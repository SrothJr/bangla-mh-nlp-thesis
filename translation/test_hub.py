
import sys
import os
import json
from huggingface_hub import hf_hub_download, try_to_load_from_cache

print("Testing hf_hub_download for gpt2 vocab.json with force_download=True...")
try:
    # Delete cache first
    import shutil
    cache_dir = os.path.join(os.environ['USERPROFILE'], '.cache', 'huggingface', 'hub')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("Deleted cache")

    vocab_path = hf_hub_download(repo_id="gpt2", filename="vocab.json", force_download=True)
    print("vocab_path found:", vocab_path)
    print("File exists:", os.path.exists(vocab_path))
    print("File size:", os.path.getsize(vocab_path))
    with open(vocab_path, 'r', encoding='utf8') as f:
        vocab = json.load(f)
    print("Loaded vocab, len(vocab):", len(vocab))
    print("First 10 entries:", list(vocab.items())[:10])
except Exception as e:
    print("ERROR in hf_hub_download:", type(e), str(e))
    import traceback
    traceback.print_exc()

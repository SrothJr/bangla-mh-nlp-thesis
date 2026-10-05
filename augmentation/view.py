from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')
cpath = Path('checkpoints/label_4_qwen3-32b_cleaned.jsonl')
if cpath.exists():
    print(f'Cleaned: {len([l for l in open(cpath, encoding="utf-8") if l.strip()])}')
else:
    print('Cleaned: Not found')
    
opath = Path('checkpoints/label_4_qwen3-32b.jsonl')
if opath.exists():
    print(f'Original: {len([l for l in open(opath, encoding="utf-8") if l.strip()])}')
else:
    print('Original: Not found')

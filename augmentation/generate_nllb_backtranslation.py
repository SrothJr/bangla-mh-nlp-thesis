import os
import sys
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# FIX 1: Explicitly disable tokenizer parallelism to prevent Windows deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import mask_utils from the tari_translation_v3 directory
TARI_DIR = r"C:\Users\user6\T2520814\tari_translation_v3"
sys.path.append(TARI_DIR)
try:
    from mask_utils import mask_text, unmask_text
except ImportError:
    print(f"ERROR: Could not import mask_utils.py from {TARI_DIR}")
    sys.exit(1)

import re

# Regex for non-BMP emojis and typical emojis
# This covers most standard emojis
EMOJI_REGEX = re.compile(r'[\U00010000-\U0010FFFF\u2600-\u27BF\uFE0F\u200D]+', flags=re.UNICODE)

def mask_emojis(text):
    """Replaces emojis with a safe placeholder so NLLB doesn't hallucinate, preserving them for later."""
    emojis_found = {}
    idx = 1
    
    def replacer(match):
        nonlocal idx
        e_str = match.group(0)
        placeholder = f" EMOJI{idx}X "
        emojis_found[idx] = e_str
        idx += 1
        return placeholder

    masked_text = EMOJI_REGEX.sub(replacer, text)
    return masked_text, emojis_found

def unmask_emojis(text, emojis_found):
    """Restores emojis from placeholders."""
    result = text
    for idx in sorted(emojis_found.keys(), reverse=True):
        placeholder = f"EMOJI{idx}X"
        # Remove surrounding spaces that might have been added
        result = re.sub(rf'\s*{placeholder}\s*', emojis_found[idx], result)
    return result


# Paths
INPUT_EXCEL = r"C:\Users\user6\T2520814\nazim_augmentation\train.xlsx"
OUTPUT_DIR = r"C:\Users\user6\T2520814\nazim_augmentation\nllb_backtranslated_output"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_backtranslated.jsonl")
MODEL_PATH = r"C:\Users\user6\T2520814\tari_translation_v3\nllb_model"

BATCH_SIZE = 4

def load_data():
    print(f"Loading {INPUT_EXCEL}...")
    df = pd.read_excel(INPUT_EXCEL)
    # df should have 'posts' and 'labels'
    return df.to_dict('records')

def get_done_indices():
    if not os.path.exists(OUTPUT_FILE):
        return set()
    done = set()
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            done.add(obj['original_index'])
    return done

def translate_batch(model, tokenizer, texts, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to('cuda')
    
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=1024,
            num_beams=3,
            repetition_penalty=1.1
        )
        
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    # Aggressive VRAM cleanup to prevent memory spilling to shared RAM
    del inputs
    del generated_tokens
    torch.cuda.empty_cache()
    
    return result

def main():
    print("Loading NLLB-200-3.3B Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
    ).to('cuda')
    model.eval()
    
    records = load_data()
    done_indices = get_done_indices()
    print(f"Total records: {len(records)}. Already done: {len(done_indices)}")
    
    pending = [(i, rec) for i, rec in enumerate(records) if i not in done_indices]
    
    if not pending:
        print("All records backtranslated!")
        return

    print(f"Processing {len(pending)} records...")
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for i in tqdm(range(0, len(pending), BATCH_SIZE)):
            batch = pending[i:i+BATCH_SIZE]
            indices = [x[0] for x in batch]
            orig_texts = [str(x[1]['posts']) for x in batch]
            labels = [x[1]['labels'] for x in batch]
            
            # Step 1: Masking (Emojis first, then clinical terms)
            masked_texts = []
            luts_clinical = []
            luts_emoji = []
            for text in orig_texts:
                # 1a. Mask Emojis
                masked_e, lut_e = mask_emojis(text)
                # 1b. Mask Clinical terms/Slang
                masked_c, lut_c = mask_text(masked_e)
                
                masked_texts.append(masked_c)
                luts_clinical.append(lut_c)
                luts_emoji.append(lut_e)
                
            # Step 2: Forward Translate (Bangla -> English)
            english_texts = translate_batch(model, tokenizer, masked_texts, src_lang="ben_Beng", tgt_lang="eng_Latn")
            
            # Step 3: Back Translate (English -> Bangla)
            back_translated = translate_batch(model, tokenizer, english_texts, src_lang="eng_Latn", tgt_lang="ben_Beng")
            
            # Step 4: Unmasking & Save (Clinical first, then Emojis)
            for j in range(len(batch)):
                # 4a. Unmask clinical terms
                unmasked_c = unmask_text(back_translated[j], luts_clinical[j])
                # 4b. Unmask emojis
                final_text = unmask_emojis(unmasked_c, luts_emoji[j])
                
                output_obj = {
                    "original_index": indices[j],
                    "original_post": orig_texts[j],
                    "backtranslated_post": final_text,
                    "label": labels[j]
                }
                f.write(json.dumps(output_obj, ensure_ascii=False) + "\n")
                
            f.flush()

    print("Backtranslation complete! Saved to train_backtranslated.jsonl")

if __name__ == "__main__":
    main()

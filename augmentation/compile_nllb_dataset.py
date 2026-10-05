import os
import json
import re
import pandas as pd

INPUT_EXCEL = r"C:\Users\user6\T2520814\nazim_augmentation\train.xlsx"
BACKTRANSLATED_FILE = r"C:\Users\user6\T2520814\nazim_augmentation\nllb_backtranslated_output\train_backtranslated.jsonl"
OUTPUT_EXCEL = r"C:\Users\user6\T2520814\nazim_augmentation\nllb_backtranslated_output\train_augmented_nllb.xlsx"

def clean_and_validate(orig_text, bt_text):
    orig_text = str(orig_text).strip()
    bt_text = str(bt_text).strip()
    
    # 1. Clean Emoji Artifacts ("ইমোজি 1 এক্স", "EMOJI1X", etc.)
    bt_text = re.sub(r'ইমোজি\s*\d+\s*এক্স', '', bt_text)
    bt_text = re.sub(r'(?i)EMOJI\s*\d+\s*X', '', bt_text)
    bt_text = re.sub(r'\s+', ' ', bt_text).strip() # Clean extra spaces
    
    # 2. Quality Filter: Minimum length (Translation failed)
    if len(bt_text) < 10:
        return None
        
    # 3. Quality Filter: Repetition Loop (NLLB hallucination where it loops)
    # If it's suddenly 2.5x longer than the original (and original isn't tiny)
    if len(orig_text) > 20 and len(bt_text) > (2.5 * len(orig_text)):
        return None
        
    # 4. Quality Filter: English leakage (NLLB gave up and wrote English)
    eng_chars = sum(1 for c in bt_text if c.isascii() and c.isalpha())
    if eng_chars > 30 and eng_chars > (0.4 * len(bt_text)):
        return None
        
    return bt_text

def main():
    print("Loading original train dataset...")
    df_orig = pd.read_excel(INPUT_EXCEL)
    df_orig['source'] = 'original'
    
    print(f"Loading backtranslated texts from {BACKTRANSLATED_FILE}...")
    if not os.path.exists(BACKTRANSLATED_FILE):
        print("Error: Backtranslated file not found.")
        return
        
    bt_records = []
    dropped_count = 0
    
    with open(BACKTRANSLATED_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            
            clean_bt = clean_and_validate(obj['original_post'], obj['backtranslated_post'])
            
            if clean_bt:
                bt_records.append({
                    'posts': clean_bt,
                    'labels': obj['label'],
                    'source': 'backtranslated_nllb'
                })
            else:
                dropped_count += 1
                
    df_bt = pd.DataFrame(bt_records)
    print(f"Loaded {len(df_bt)} high-quality backtranslated rows.")
    print(f"Dropped {dropped_count} low-quality rows (loops, artifacts, empty).")
    
    # Combine
    df_combined = pd.concat([df_orig, df_bt], ignore_index=True)
    
    print("\nFinal Combined Dataset Label Counts:")
    print(df_combined['labels'].value_counts())
    
    # Save
    print(f"\nSaving to {OUTPUT_EXCEL}...")
    df_combined.to_excel(OUTPUT_EXCEL, index=False)
    print("Done! Your data is now perfectly filtered and ready for fine-tuning.")

if __name__ == "__main__":
    main()


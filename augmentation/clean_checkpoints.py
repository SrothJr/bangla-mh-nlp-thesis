import json
import re
import time
from pathlib import Path
import ollama

# We use the fast qwen2.5:32b to evaluate coherence quickly
EVAL_MODEL = "qwen2.5:32b"

def check_banglish(text, threshold=0.3):
    """
    Returns True if more than 30% of the text's characters are English letters.
    This safely allows normal code-mixing (e.g. 'crush', 'boyfriend') but catches full Banglish.
    """
    letters = len(re.findall(r'[a-zA-Z]', text))
    ratio = letters / max(len(text), 1)
    return ratio > threshold

def check_coherence(text, model=EVAL_MODEL):
    """
    Asks the LLM to zero-shot verify if the Bengali text makes logical sense.
    """
    prompt = f"""You are an extremely strict, native Bengali linguistic evaluator.
You must identify texts that sound like poor machine translations from English to Bengali. 

Your task is to identify ANY garbled, nonsensical, unnatural, or meaningless sentences. 
A text is GARBAGE and must be rejected if it contains:
1. Words that don't belong together (e.g., "সবকিছু বিস্তারিত" used incorrectly).
2. Unnatural verb tenses or structures (e.g., "রাতে ঘুম নাই হচ্ছে").
3. Sentences that sound like they were translated from English by Google Translate.

Example of GARBAGE (Reply NO):
"ফোনে কথা বলতে ভয় হয়। সবাই আমার সাথে রাখবাসায় বসে আছি,, দরজা কেউ খুলতে চায় না। কথা বলতে আর আগ্রহ নেই। মনে হয় নিজেকে কোনো জায়গা পাওয়া যাচ্ছে না।, রাতে ঘুম নাই হচ্ছে,, সকালে ঘুমানো যাচ্ছে না। দিন শেষ হয়ে গেল কিন্তু কাজ করা যাচ্ছে না। মনে হয় সবকিছু বিস্তারিত।"

A text is GOOD (Reply YES) only if the entire story makes perfect logical sense and sounds like a real Bangladeshi person wrote it natively on Facebook.

Post: {text}

Reply with ONLY the word "YES" if the text is 100% coherent and native.
Reply with ONLY the word "NO" if it sounds like a bad translation, is garbled, or unnatural."""
    
    for attempt in range(2):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 10},
            )
            pred = response["message"]["content"].strip().upper()
            if "YES" in pred: return True
            if "NO" in pred: return False
            # Default to true if ambiguous
            return True
        except Exception as e:
            time.sleep(1)
    return True # fail open (keep data if API fails)

def main():
    cp_dir = Path("checkpoints")
    if not cp_dir.exists():
        print("No checkpoints directory found.")
        return
        
    print(f"Starting cleanup. Using {EVAL_MODEL} for coherence checking...")
    
    for cp_file in cp_dir.glob("*.jsonl"):
        # Don't clean files that are already cleaned
        if "cleaned" in cp_file.name: 
            continue
            
        print(f"\n======================================")
        print(f"Processing {cp_file.name}...")
        
        with open(cp_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            
        if not lines:
            continue
            
        kept = []
        banglish_dropped = 0
        incoherent_dropped = 0
        
        for i, line in enumerate(lines):
            try:
                data = json.loads(line)
                text = data["text"]
                label = data["label"]
            except Exception:
                continue
                
            # Filter 1: Banglish
            if check_banglish(text):
                banglish_dropped += 1
                continue
                
            # Filter 2: Coherence
            if not check_coherence(text):
                incoherent_dropped += 1
                continue
                
            kept.append(data)
            
            # Print progress every 50 lines to show it's working
            if (i+1) % 50 == 0:
                print(f"  Processed {i+1}/{len(lines)}...")
                
        # Save cleaned file alongside the original
        clean_file = cp_dir / f"{cp_file.stem}_cleaned.jsonl"
        with open(clean_file, "w", encoding="utf-8") as f:
            for data in kept:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                
        print(f"--- Results for {cp_file.name} ---")
        print(f"  Total texts       : {len(lines)}")
        print(f"  Dropped Banglish  : {banglish_dropped}")
        print(f"  Dropped Incoherent: {incoherent_dropped}")
        print(f"  Kept texts        : {len(kept)}")
        print(f"  Saved to          : {clean_file.name}")

    print("\nALL DONE! Review your cleaned files.")
    print("If you are happy with the cleaned texts, delete the original .jsonl files")
    print("and rename the '_cleaned.jsonl' files to replace them.")

if __name__ == "__main__":
    main()

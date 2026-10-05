import os
import json
import time
import random
import requests
import pandas as pd
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:32b"
OUTPUT_DIR = r"C:\Users\user6\T2520814\nazim_augmentation\tapt_qwen_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "qwen_tapt_corpus.jsonl")
TARGET_COUNT = 50000
INPUT_EXCEL = r"C:\Users\user6\T2520814\nazim_augmentation\train.xlsx"

print("Loading real dataset for Dynamic Few-Shot...")
df_real = pd.read_excel(INPUT_EXCEL)
real_posts = df_real['posts'].astype(str).tolist()

def get_existing_count(filepath):
    if not os.path.exists(filepath):
        return 0
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count

TOPICS = [
    # Academic & Career
    "severe academic pressure and fear of failing exams",
    "unemployment, career anxiety, and extreme financial debt",
    "workplace bullying, extreme burnout, or job loss",
    
    # Relationships & Family
    "romantic relationship failures, heartbreak, and betrayal",
    "family conflicts, strict parents, and lack of freedom",
    "domestic violence, abusive marriages, or physical abuse",
    "forced marriage or dowry-related severe stress",
    "feeling like a massive burden to parents or family",
    
    # Social & Lifestyle
    "extreme loneliness and having absolutely no friends",
    "social media addiction, comparison, and feeling inadequate",
    "social isolation and alienation in urban city life",
    "stigma around mental illness and parents not believing in depression",
    
    # Trauma & Medical
    "grief, bereavement, and losing a close loved one",
    "chronic insomnia, physical fatigue, and sleep deprivation",
    "health anxiety, chronic illness, or physical disability",
    "postpartum depression and struggling with new motherhood",
    "harassment, bullying, or past traumatic events"
]

FORMATS = [
    "short, fragmented emotional outburst",
    "detailed, multi-paragraph story explaining the situation",
    "rhetorical question asking others for advice",
    "deep, reflective late-night thought"
]

def generate_post():
    is_zero_shot = random.choice([True, False])
    
    # Qwen Best Practice 1: Strong System Prompt
    system_prompt = (
        "You are an expert synthetic data generator for a Bengali mental health classifier. "
        "Your job is to generate highly realistic, informal, and colloquial Bangladeshi social media posts (Facebook/Reddit). "
        "Do NOT use formal, bookish Bengali. Output strictly the Bengali text with no conversational filler, no English, and no artificial hashtags."
    )
    
    if is_zero_shot:
        topic = random.choice(TOPICS)
        fmt = random.choice(FORMATS)
        
        prompt = (
            "### Instruction:\n"
            f"Generate a {fmt} about mental health, depression, or anxiety, specifically focusing on this exact topic: '{topic}'. "
            "Ensure it sounds like a real person writing a status update. Use completely natural vocabulary related to the topic."
        )
        mode = "zero_shot"
    else:
        sample = random.choice(real_posts)
        prompt = (
            "### Instruction:\n"
            "Generate a completely NEW social media post in a similar informal Bengali style and tone as the example below. "
            "Use the example as inspiration for vocabulary and grammar structure ONLY. "
            "CRITICAL: Do NOT just reword the example. You MUST write about a completely different scenario and change the topic entirely. "
            "Ensure it sounds like a real person.\n\n"
            "### Example:\n"
            f"{sample}"
        )
        mode = "one_shot"

    payload = {
        "model": MODEL_NAME,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "top_p": 0.90  # Qwen Best Practice 3: Tighter top_p for structural stability
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip(), mode
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def main():
    print(f"Starting TAPT Corpus Generation with {MODEL_NAME}")
    
    existing = get_existing_count(OUTPUT_FILE)
    print(f"Found {existing} existing posts in {OUTPUT_FILE}")
    
    remaining = TARGET_COUNT - existing
    if remaining <= 0:
        print("Target already reached. Exiting.")
        return
        
    print(f"Generating {remaining} more posts...")
    
    # We open in append mode, so it's safe to stop and start anytime
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        pbar = tqdm(total=remaining)
        
        consecutive_errors = 0
        while remaining > 0:
            result = generate_post()
            
            if result:
                post, mode = result
                # Basic validation
                if len(post) > 10 and not post.startswith("Here is"):
                    record = {"text": post, "source": f"qwen3:32b_tapt_{mode}"}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    remaining -= 1
                    pbar.update(1)
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
            else:
                consecutive_errors += 1
                time.sleep(2) # Back off on error
                
            if consecutive_errors > 5:
                print("\nToo many consecutive errors. Is Ollama running and the model loaded? Pausing for 10s...")
                time.sleep(10)
                consecutive_errors = 0

    pbar.close()
    print("Done generating TAPT corpus.")

if __name__ == "__main__":
    main()

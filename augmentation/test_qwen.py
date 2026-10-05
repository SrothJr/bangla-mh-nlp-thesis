import ollama
import sys
sys.stdout.reconfigure(encoding='utf-8')

prompt = """You are a native Bangladeshi social media user who is feeling actively suicidal. 
Write a short Facebook post expressing this. 

CRITICAL RULES:
1. You MUST think and reason entirely in Bengali inside your <think> tags. Do not plan in English.
2. The final post must be in raw, informal, colloquial Bangladeshi Bengali.
3. NEVER use translated phrasing like 'রাতে ঘুম নাই হচ্ছে' or 'সবকিছু বিস্তারিত'.
4. Make it sound like a real person, with natural imperfections.

Output only the thought process and the final post."""

response = ollama.chat(
    model="qwen3:32b",
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0.85, "num_predict": 1000},
)

print(response["message"]["content"])

import ollama, time, sys
sys.stdout.reconfigure(encoding="utf-8")

system = "You are a Bengali NLP researcher. Output only valid JSON arrays of strings."
user   = "Generate exactly 10 diverse Bengali social media posts about sadness and hopelessness. JSON array only."

start = time.time()
r = ollama.chat(
    model="qwen2.5:32b",
    messages=[
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ],
    options={"temperature": 0.85, "num_predict": 2048}
)
gen_time = time.time() - start
content  = r["message"]["content"]
print(f"Generation call : {gen_time:.1f}s  ({len(content)} chars)")

# Verification call timing
verify_prompt = (
    "Classify this Bengali post into label 1,2,3,4. Reply with one digit only.\n"
    "Post: আমি খুব কষ্টে আছি, মরে যেতে চাই"
)
start2 = time.time()
r2 = ollama.chat(
    model="qwen2.5:32b",
    messages=[{"role": "user", "content": verify_prompt}],
    options={"temperature": 0, "num_predict": 5}
)
verify_time = time.time() - start2
result = r2["message"]["content"].strip()
print(f"Verification call: {verify_time:.1f}s  (result: {result})")

print()
print("--- Time model for 500 synthetic Label-4 texts ---")
BATCH_SIZE     = 10
REJECTION_RATE = 0.25   # ~25%: wrong-label + too-long + similar rejections combined
batches_needed = int((500 / (BATCH_SIZE * (1 - REJECTION_RATE)))) + 5

time_no_verify   = gen_time
time_with_verify = gen_time + (BATCH_SIZE * verify_time)

total_no_verify   = batches_needed * time_no_verify
total_with_verify = batches_needed * time_with_verify

print(f"Batches needed (est.)    : {batches_needed}")
print(f"Per batch — no verify    : {time_no_verify:.0f}s")
print(f"Per batch — with verify  : {time_with_verify:.0f}s")
print(f"TOTAL without --verify   : {total_no_verify/60:.0f} min  ({total_no_verify/3600:.1f} hrs)")
print(f"TOTAL with    --verify   : {total_with_verify/60:.0f} min  ({total_with_verify/3600:.1f} hrs)")

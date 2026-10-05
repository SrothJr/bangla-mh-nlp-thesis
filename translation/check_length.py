
import json

with open("merged_mental_health_dapt.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

row8 = json.loads(lines[8])
print("Row 8 text length:", len(row8["text"]))
print("Row 8 text:", row8["text"])

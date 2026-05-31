import json
from pathlib import Path
from collections import Counter

emotions = []
for f in sorted(Path("data/train-data").rglob("*.json")):
    data = json.loads(f.read_text())
    if isinstance(data, list):
        for item in data:
            e = (item.get("emotion") or "").strip()
            if e:
                emotions.append(e)

print(f"Total: {len(emotions)}, Unique: {len(set(emotions))}\n")
for e, c in Counter(emotions).most_common():
    print(f"  [{c:3d}] {e[:80]}")

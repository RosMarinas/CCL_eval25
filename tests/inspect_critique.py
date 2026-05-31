import json
with open('data/teacher/train-critique.jsonl') as f:
    lines = f.readlines()
print(f"Total lines: {len(lines)}")
for i, line in enumerate(lines):
    if i > 3: break
    rec = json.loads(line.strip())
    print(f"--- Record {i} ---")
    print(f"  record_type: {rec.get('record_type', '?')}")
    if 'error_type' in rec:
        print(f"  error_type: {rec['error_type']}")
        detail = rec.get('error_detail', '')
        print(f"  error_detail (first 300): {detail[:300]}")
    else:
        print(f"  success keys: {list(rec.keys())}")

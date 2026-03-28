#!/usr/bin/env python3
"""Print label distribution from data/table_classifications.json. Run from repo root."""
import json
import os
import sys
from collections import Counter

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
path = os.path.join(repo_root, "data", "table_classifications.json")
if not os.path.exists(path):
    print(f"File not found: {path}", file=sys.stderr)
    sys.exit(1)

with open(path) as f:
    data = json.load(f)
counts = Counter(data.values())
total = len(data)

print("Label distribution (data/table_classifications.json)")
print("Total tables:", total)
print()
for label, n in counts.most_common():
    pct = 100.0 * n / total
    print(f"  {label}: {n} ({pct:.1f}%)")
print()
print("Labels (sorted):", sorted(counts.keys()))

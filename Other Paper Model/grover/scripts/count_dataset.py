#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

p = Path('data/491_dataset.jsonl')
if not p.exists():
    print(f'File {p} not found')
    raise SystemExit(1)

cnt = Counter()
total = 0
with p.open('r', encoding='utf-8') as fh:
    for l in fh:
        total += 1
        obj = json.loads(l)
        cnt['split:'+obj.get('split','')] += 1
        cnt['label:'+obj.get('label','')] += 1

print(f'total={total}')
for k in sorted(cnt.keys()):
    print(k, cnt[k])

#!/usr/bin/env python3
"""
Prepare the provided CSV dataset into the JSONL format expected by the repo's discrimination code.

Reads: 491_Dataset.csv (expects columns: Writter, Type, Graph, Writing)
Writes: data/491_dataset.jsonl (one JSON per line) with fields used by Grover's discriminator:
  - domain (set to 'user_dataset')
  - date (empty)
  - authors (list with Writter)
  - title (empty)
  - article (the Writing column)
  - label ('human'|'machine')
  - split ('train' or 'test')
  - inst_index (integer index)

Splits deterministically with seed=123456 into 60% train and 40% test.

Usage:
  python scripts\prepare_491_dataset.py --csv 491_Dataset.csv --out data/491_dataset.jsonl

"""
import csv
import json
import argparse
import random
from pathlib import Path


def normalize_label(raw):
    if raw is None:
        return 'human'
    r = str(raw).strip().lower()
    if r in ('ai', 'machine', 'bot'):
        return 'machine'
    if r in ('human', 'writer', 'human-written'):
        return 'human'
    # fallback: treat any non-human-looking label as machine only if it contains 'ai'
    if 'ai' in r:
        return 'machine'
    return 'human'


def prepare(csv_path, out_path, seed=123456, train_frac=0.6):
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with csv_path.open('r', encoding='utf-8-sig', newline='') as fh:
        # Use csv.DictReader which handles multiline quoted fields
        reader = csv.DictReader(fh)
        for i, r in enumerate(reader):
            writer = r.get('Writter') or r.get('Writer') or r.get('writter') or ''
            raw_type = r.get('Type') or r.get('type') or ''
            text = r.get('Writing') or r.get('writing') or r.get('Text') or ''
            if text is None:
                text = ''
            label = normalize_label(raw_type)
            rows.append({'authors': [writer], 'article': text, 'label': label, 'orig_index': i})

    # shuffle deterministically
    random.Random(seed).shuffle(rows)

    n = len(rows)
    n_train = int(round(n * train_frac))

    counts = {'train': 0, 'test': 0}
    with out_path.open('w', encoding='utf-8') as out_f:
        for i, row in enumerate(rows):
            split = 'train' if i < n_train else 'test'
            counts[split] += 1
            obj = {
                'domain': 'user_dataset',
                'date': '',
                'authors': row['authors'],
                'title': '',
                'article': row['article'],
                'label': row['label'],
                'split': split,
                'inst_index': row['orig_index'],
            }
            out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    print(f'Wrote {n} examples -> train={counts["train"]}, test={counts["test"]} to {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='491_Dataset.csv')
    parser.add_argument('--out', type=str, default='data/491_dataset.jsonl')
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--train_frac', type=float, default=0.6)
    args = parser.parse_args()
    prepare(args.csv, args.out, seed=args.seed, train_frac=args.train_frac)


if __name__ == '__main__':
    main()

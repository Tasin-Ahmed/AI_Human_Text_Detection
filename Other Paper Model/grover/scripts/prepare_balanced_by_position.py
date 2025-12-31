#!/usr/bin/env python3
"""
Prepare the CSV by assigning labels in blocks of 10 alternating human/machine.

Pattern: first 10 rows -> human, next 10 -> machine, next 10 -> human, etc.
Produces: data/491_dataset_balanced.jsonl with 'split' set to train/test (60/40) and balanced across classes.

Usage:
  python scripts\prepare_balanced_by_position.py --csv 491_Dataset.csv --out data/491_dataset_balanced.jsonl
"""
import csv
import json
import argparse
import random
from pathlib import Path


def label_by_position(index):
    block = (index // 10) % 2
    return 'human' if block == 0 else 'machine'


def prepare(csv_path, out_path, seed=123456, train_frac=0.6):
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_by_label = {'human': [], 'machine': []}
    with csv_path.open('r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        for i, r in enumerate(reader):
            writer = (r.get('Writter') or r.get('Writer') or '').strip()
            text = r.get('Writing') or r.get('writing') or r.get('Text') or ''
            if text is None:
                text = ''
            label = label_by_position(i)
            rows_by_label[label].append({'authors': [writer], 'article': text, 'orig_index': i})

    # Shuffle within each label deterministically and split
    random.Random(seed).shuffle(rows_by_label['human'])
    random.Random(seed).shuffle(rows_by_label['machine'])

    n_h = len(rows_by_label['human'])
    n_m = len(rows_by_label['machine'])

    n_h_train = int(round(n_h * train_frac))
    n_m_train = int(round(n_m * train_frac))

    counts = {'train': 0, 'test': 0}
    with out_path.open('w', encoding='utf-8') as out_f:
        # write train human
        for i, ex in enumerate(rows_by_label['human']):
            split = 'train' if i < n_h_train else 'test'
            obj = {
                'domain': 'user_dataset',
                'date': '',
                'authors': ex['authors'],
                'title': '',
                'article': ex['article'],
                'label': 'human',
                'split': split,
                'inst_index': ex['orig_index'],
            }
            out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')
            counts[split] += 1

        # write machine
        for i, ex in enumerate(rows_by_label['machine']):
            split = 'train' if i < n_m_train else 'test'
            obj = {
                'domain': 'user_dataset',
                'date': '',
                'authors': ex['authors'],
                'title': '',
                'article': ex['article'],
                'label': 'machine',
                'split': split,
                'inst_index': ex['orig_index'],
            }
            out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')
            counts[split] += 1

    total = n_h + n_m
    print(f'Wrote {total} examples -> train={counts["train"]}, test={counts["test"]} (human={n_h}, machine={n_m}) to {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='491_Dataset.csv')
    parser.add_argument('--out', type=str, default='data/491_dataset_balanced.jsonl')
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--train_frac', type=float, default=0.6)
    args = parser.parse_args()
    prepare(args.csv, args.out, seed=args.seed, train_frac=args.train_frac)


if __name__ == '__main__':
    main()

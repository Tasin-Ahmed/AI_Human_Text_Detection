#!/usr/bin/env python3
"""Run the uniform random baseline multiple times and aggregate metrics.

Writes a JSON summary with per-run metrics and mean/std to the output directory, and prints a short summary.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, f1_score, balanced_accuracy_score


def find_text_column(df: pd.DataFrame):
    candidates = [c for c in df.columns if c.lower() in ("text", "writing", "content", "sentence")]
    if candidates:
        return candidates[0]
    lengths = {c: df[c].astype(str).map(len).median() for c in df.columns}
    return max(lengths.items(), key=lambda x: x[1])[0]


def find_label_column(df: pd.DataFrame):
    candidates = [c for c in df.columns if c.lower() in ("type", "label", "y")]
    if candidates:
        return candidates[0]
    for alt in ("Type", "type", "Label", "label"):
        if alt in df.columns:
            return alt
    return None


def map_labels_to_binary(raw):
    try:
        return raw.astype(int).to_numpy(), None
    except Exception:
        mapped = raw.astype(str).str.lower().str.strip().map(lambda x: 0 if x == 'human' else 1)
        return mapped.to_numpy(), mapped


def run_once(y_true, seed):
    rng = np.random.RandomState(seed)
    preds = rng.randint(0, 2, size=len(y_true)).astype(int)

    acc = float(accuracy_score(y_true, preds))
    prec, rec, f1_bin, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
    f1_macro = float(f1_score(y_true, preds, average='macro', zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, preds))

    return {
        'seed': int(seed),
        'accuracy': acc,
        'precision': float(prec),
        'recall': float(rec),
        'f1_binary': float(f1_bin),
        'f1_macro': f1_macro,
        'balanced_accuracy': bal_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--n', type=int, default=100, help='Number of repeats')
    parser.add_argument('--random_state', type=int, default=0)
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if data_path.suffix.lower() in ('.csv', '.tsv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_json(data_path, lines=True)

    label_col = find_label_column(df)
    if label_col is None:
        raise SystemExit('No label column found; cannot compute metrics')

    y_raw = df[label_col]
    y_true, _ = map_labels_to_binary(y_raw)

    results = []
    for i in range(args.n):
        seed = args.random_state + i
        res = run_once(y_true, seed)
        results.append(res)

    # aggregate
    metrics = {k: [r[k] for r in results] for k in results[0] if k != 'seed'}
    summary = {}
    for k, vals in metrics.items():
        arr = np.array(vals)
        summary[k] = {'mean': float(arr.mean()), 'std': float(arr.std()), 'min': float(arr.min()), 'max': float(arr.max())}

    out = {
        'n_runs': args.n,
        'per_run': results,
        'summary': summary,
    }

    out_path = out_dir / f'random_baseline_uniform_{args.n}_runs_summary.json'
    with open(out_path, 'w', encoding='utf8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # print compact summary
    print(f'Ran {args.n} uniform-random baseline runs. Summary (mean ± std):')
    for metric in ['accuracy', 'precision', 'recall', 'f1_binary', 'f1_macro', 'balanced_accuracy']:
        s = summary.get(metric, {})
        print(f" - {metric}: {s.get('mean'):.4f} ± {s.get('std'):.4f} (min {s.get('min'):.4f}, max {s.get('max'):.4f})")
    print(f'Wrote detailed results to: {out_path}')


if __name__ == '__main__':
    main()

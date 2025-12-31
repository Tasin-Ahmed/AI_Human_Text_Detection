#!/usr/bin/env python3
"""Random baseline predictions.

Modes:
 - distribution: sample according to the dataset class distribution (requires labels)
 - uniform: sample uniformly at random between classes (balanced random)

This script writes predictions and metrics to the specified output directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV or JSONL dataset")
    parser.add_argument("--out_dir", required=True, help="Directory to write predictions and metrics")
    parser.add_argument("--mode", choices=("distribution", "uniform"), default="uniform", help="Sampling mode for random baseline")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if data_path.suffix.lower() in (".csv", ".tsv"):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_json(data_path, lines=True)

    if df.shape[0] == 0:
        raise SystemExit("No rows in dataset")

    text_col = find_text_column(df)
    label_col = find_label_column(df)
    if label_col is None:
        raise SystemExit("No label column found in dataset; random baseline requires labels to compute metrics.")

    texts = df[text_col].astype(str).tolist()
    y_raw = df[label_col]
    y_bin, mapped_series = map_labels_to_binary(y_raw)

    rng = np.random.RandomState(args.random_state)
    if args.mode == 'uniform':
        # balanced random between two classes 0 and 1
        preds = rng.randint(0, 2, size=len(texts)).astype(int)
    else:
        # distribution: sample according to empirical class probabilities
        vals, counts = np.unique(y_bin, return_counts=True)
        probs = counts / counts.sum()
        # map probs to classes order in vals (should be 0 and 1)
        # create a full-prob vector for classes 0 and 1
        prob_map = {int(v): float(p) for v, p in zip(vals, probs)}
        p0 = prob_map.get(0, 0.0)
        p1 = prob_map.get(1, 0.0)
        preds = rng.choice([0, 1], size=len(texts), p=[p0, p1]).astype(int)

    out_df = pd.DataFrame({'text': texts, 'true_label': y_raw.astype(str).tolist(), 'pred_label': preds})
    preds_csv = out_dir / f"predictions_random_baseline_{args.mode}.csv"
    out_df.to_csv(preds_csv, index=False, encoding='utf-8')

    acc = float(accuracy_score(y_bin, preds))
    prec, rec, f1, _ = precision_recall_fscore_support(y_bin, preds, average='binary', zero_division=0)
    cm = confusion_matrix(y_bin, preds).tolist()
    creport = classification_report(y_bin, preds, output_dict=True, zero_division=0)

    metrics = {
        'mode': args.mode,
        'accuracy': acc,
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'confusion_matrix': cm,
        'classification_report': creport,
        'n': int(len(y_bin)),
    }

    metrics_path = out_dir / f"metrics_random_baseline_{args.mode}.json"
    with open(metrics_path, 'w', encoding='utf8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Wrote random predictions to: {preds_csv}")
    print(f"Wrote metrics to: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

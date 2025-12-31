#!/usr/bin/env python3
"""Majority-class baseline: predict the most frequent label for every sample.

This performs no training on your data — it only inspects the labels to determine the
majority class and then predicts that class for all samples. Writes predictions and
metrics to the specified output directory.
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
        raise SystemExit("No label column found in dataset; majority baseline requires labels to determine majority class.")

    texts = df[text_col].astype(str).tolist()
    y_raw = df[label_col]
    y_bin, mapped_series = map_labels_to_binary(y_raw)

    # determine majority class from labels
    vals, counts = np.unique(y_bin, return_counts=True)
    majority = int(vals[np.argmax(counts)])

    preds = np.full(len(texts), majority, dtype=int)

    # prepare output
    out_df = pd.DataFrame({'text': texts, 'true_label': y_raw.astype(str).tolist(), 'pred_label': preds})
    preds_csv = out_dir / f"predictions_majority_baseline.csv"
    out_df.to_csv(preds_csv, index=False, encoding='utf-8')

    acc = float(accuracy_score(y_bin, preds))
    prec, rec, f1, _ = precision_recall_fscore_support(y_bin, preds, average='binary')
    cm = confusion_matrix(y_bin, preds).tolist()
    creport = classification_report(y_bin, preds, output_dict=True)

    metrics = {
        'majority_class': int(majority),
        'accuracy': acc,
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'confusion_matrix': cm,
        'classification_report': creport,
        'n': int(len(y_bin)),
    }

    metrics_path = out_dir / "metrics_majority_baseline.json"
    with open(metrics_path, 'w', encoding='utf8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Wrote majority predictions to: {preds_csv}")
    print(f"Wrote metrics to: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

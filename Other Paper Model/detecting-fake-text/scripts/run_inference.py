#!/usr/bin/env python3
"""Load a saved pipeline (joblib) and run inference on a dataset (CSV or JSONL).

This script does NOT train. It only loads the provided model and evaluates it on the
provided dataset, writing predictions and metrics to the output directory.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


def find_text_column(df: pd.DataFrame):
    candidates = [c for c in df.columns if c.lower() in ("text", "writing", "content", "sentence")]
    if candidates:
        return candidates[0]
    lengths = {c: df[c].astype(str).map(len).median() for c in df.columns}
    best = max(lengths.items(), key=lambda x: x[1])[0]
    return best


def find_label_column(df: pd.DataFrame):
    candidates = [c for c in df.columns if c.lower() in ("type", "label", "y")]
    if candidates:
        return candidates[0]
    for alt in ("Type", "type", "Label", "label"):
        if alt in df.columns:
            return alt
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV or JSONL dataset")
    parser.add_argument("--model", required=True, help="Path to saved joblib pipeline (no training will occur)")
    parser.add_argument("--out_dir", required=True, help="Directory to write predictions and metrics")
    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    if data_path.suffix.lower() in (".csv", ".tsv"):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_json(data_path, lines=True)

    if df.shape[0] == 0:
        raise SystemExit("No rows in dataset")

    text_col = find_text_column(df)
    label_col = find_label_column(df)
    if label_col is None:
        print("No label column found; predictions will be written but metrics cannot be computed.")

    texts = df[text_col].astype(str).tolist()

    # load pipeline
    pipeline = joblib.load(model_path)

    # predict
    preds = pipeline.predict(texts)
    probs = None
    if hasattr(pipeline, 'predict_proba'):
        try:
            probs = pipeline.predict_proba(texts)
        except Exception:
            # some pipelines expose predict_proba on final estimator
            final = pipeline
            if hasattr(pipeline, 'steps'):
                final = pipeline.steps[-1][1]
            if hasattr(final, 'predict_proba'):
                probs = final.predict_proba(pipeline[:-1].transform(texts))

    # assemble predictions frame
    out_df = pd.DataFrame({
        'text': texts,
        'pred_label': preds,
    })
    if probs is not None:
        # if binary, take prob for positive class
        if probs.shape[1] == 2:
            out_df['pred_prob_pos'] = probs[:, 1]
        else:
            # include full vector as string
            out_df['pred_probs'] = probs.tolist()

    if label_col is not None:
        out_df['true_label'] = df[label_col].astype(str).tolist()

    preds_csv = out_dir / f"predictions_{model_path.stem}.csv"
    out_df.to_csv(preds_csv, index=False, encoding='utf-8')

    metrics = {}
    if label_col is not None:
        # try to map string labels to binary where possible
        y_true_raw = df[label_col]
        try:
            y_true = y_true_raw.astype(int).to_numpy()
        except Exception:
            y_true = pd.Series(y_true_raw.astype(str)).map(lambda x: 0 if x.lower().strip() == 'human' else 1).to_numpy()

        y_pred = np.array(preds).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        cm = confusion_matrix(y_true, y_pred).tolist()
        creport = classification_report(y_true, y_pred, output_dict=True)

        metrics.update({
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'confusion_matrix': cm,
            'classification_report': creport,
            'n': int(len(y_true)),
        })

    metrics_path = out_dir / f"metrics_{model_path.stem}.json"
    with open(metrics_path, 'w', encoding='utf8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions to: {preds_csv}")
    print(f"Wrote metrics to: {metrics_path}")
    if metrics:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

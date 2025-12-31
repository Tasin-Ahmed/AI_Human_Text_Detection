#!/usr/bin/env python3
"""
Train and evaluate a simple TF-IDF + Logistic Regression classifier on the prepared JSONL.
Outputs accuracy, recall, precision, and F1 (macro & per-class).

Usage:
  python scripts\train_eval_simple.py --data data/491_dataset_balanced.jsonl

"""
import json
import argparse
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score, precision_score
import numpy as np


def load_data(path):
    path = Path(path)
    X_train, y_train, X_test, y_test = [], [], [], []
    with path.open('r', encoding='utf-8') as fh:
        for l in fh:
            obj = json.loads(l)
            if obj.get('split') == 'train':
                X_train.append(obj.get('article',''))
                y_train.append(obj.get('label','human'))
            else:
                X_test.append(obj.get('article',''))
                y_test.append(obj.get('label','human'))
    return X_train, y_train, X_test, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/491_dataset_balanced.jsonl')
    parser.add_argument('--C', type=float, default=1.0)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_data(args.data)
    print(f'Train size: {len(X_train)}, Test size: {len(X_test)}')

    vect = TfidfVectorizer(max_features=30000, ngram_range=(1,2))
    Xtr = vect.fit_transform(X_train)
    Xte = vect.transform(X_test)

    # simple logistic regression
    clf = LogisticRegression(C=args.C, max_iter=1000)
    clf.fit(Xtr, y_train)

    preds = clf.predict(Xte)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, digits=4)

    print('Accuracy: {:.4f}'.format(acc))
    print('\nClassification report:\n')
    print(report)

    # also print macro recall and f1
    recall_mac = recall_score(y_test, preds, average='macro')
    f1_mac = f1_score(y_test, preds, average='macro')
    print('Macro Recall: {:.4f}, Macro F1: {:.4f}'.format(recall_mac, f1_mac))


if __name__ == '__main__':
    main()

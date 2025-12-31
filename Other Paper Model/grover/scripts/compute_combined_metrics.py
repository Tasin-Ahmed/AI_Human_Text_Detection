#!/usr/bin/env python3
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(path):
    p = Path(path)
    Xtr, ytr, Xte, yte = [], [], [], []
    with p.open('r', encoding='utf-8') as fh:
        for l in fh:
            obj = json.loads(l)
            if obj.get('split') == 'train':
                Xtr.append(obj.get('article',''))
                ytr.append(obj.get('label','human'))
            else:
                Xte.append(obj.get('article',''))
                yte.append(obj.get('label','human'))
    return Xtr, ytr, Xte, yte


def main():
    data_path = 'data/491_dataset_balanced.jsonl'
    Xtr, ytr, Xte, yte = load_data(data_path)
    vect = TfidfVectorizer(max_features=30000, ngram_range=(1,2))
    Xtr_t = vect.fit_transform(Xtr)
    Xte_t = vect.transform(Xte)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr_t, ytr)
    preds = clf.predict(Xte_t)

    acc = accuracy_score(yte, preds)
    prec_micro = precision_score(yte, preds, average='micro')
    rec_micro = recall_score(yte, preds, average='micro')
    f1_micro = f1_score(yte, preds, average='micro')

    print(f'Combined metrics (micro-averaged):')
    print(f'  Accuracy: {acc:.4f}')
    print(f'  Precision (micro): {prec_micro:.4f}')
    print(f'  Recall (micro):    {rec_micro:.4f}')
    print(f'  F1 (micro):        {f1_micro:.4f}')


if __name__ == "__main__":
    main()

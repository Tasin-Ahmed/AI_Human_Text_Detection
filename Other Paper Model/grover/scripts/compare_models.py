#!/usr/bin/env python3
"""
Run multiple quick logistic-regression experiments and print classification reports.

Experiments:
  1) baseline TF-IDF (word unigrams+bigrams) + LogisticRegression
  2) class_weight='balanced'
  3) TF-IDF with char n-grams (char 3-5)
  4) combined: char n-grams + class_weight='balanced'

Usage: python scripts\compare_models.py
"""
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


def load(path='data/491_dataset_balanced.jsonl'):
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


def run_experiment(name, vect, clf, Xtr, ytr, Xte, yte):
    print('\n' + '='*40)
    print(f'Experiment: {name}')
    Xtr_t = vect.fit_transform(Xtr)
    Xte_t = vect.transform(Xte)
    clf.fit(Xtr_t, ytr)
    preds = clf.predict(Xte_t)
    acc = accuracy_score(yte, preds)
    prec_micro = precision_score(yte, preds, average='micro')
    rec_micro = recall_score(yte, preds, average='micro')
    f1_micro = f1_score(yte, preds, average='micro')
    print(f'Accuracy: {acc:.4f}, micro P/R/F1: {prec_micro:.4f}/{rec_micro:.4f}/{f1_micro:.4f}')
    print('\nClassification report:\n')
    print(classification_report(yte, preds, digits=4))


def main():
    Xtr, ytr, Xte, yte = load()
    print(f'Train size {len(Xtr)}, Test size {len(Xte)}')

    # 1) baseline
    vect1 = TfidfVectorizer(ngram_range=(1,2), max_features=30000)
    clf1 = LogisticRegression(max_iter=1000)
    run_experiment('baseline (word ngrams 1-2)', vect1, clf1, Xtr, ytr, Xte, yte)

    # 2) balanced weights
    vect2 = TfidfVectorizer(ngram_range=(1,2), max_features=30000)
    clf2 = LogisticRegression(max_iter=1000, class_weight='balanced')
    run_experiment('class_weight=balanced', vect2, clf2, Xtr, ytr, Xte, yte)

    # 3) char ngrams
    vect3 = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=30000)
    clf3 = LogisticRegression(max_iter=1000)
    run_experiment('char n-grams (3-5)', vect3, clf3, Xtr, ytr, Xte, yte)

    # 4) char ngrams + balanced
    vect4 = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=30000)
    clf4 = LogisticRegression(max_iter=1000, class_weight='balanced')
    run_experiment('char n-grams + class_weight=balanced', vect4, clf4, Xtr, ytr, Xte, yte)


if __name__ == '__main__':
    main()

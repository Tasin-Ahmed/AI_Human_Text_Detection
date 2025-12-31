import os
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'detector_roberta.pkl'
CSV_PATH = ROOT / '491_Dataset.csv'


def load_texts(csv_path):
    df = pd.read_csv(csv_path)
    # choose the first text-like column
    for col in df.columns:
        if df[col].dtype == object:
            texts = df[col].astype(str).tolist()
            return texts
    # fallback to first column
    return df.iloc[:, 0].astype(str).tolist()


def build_labels_for_120():
    # Per user request: exactly 40 human, then 40 AI, then 40 human (total 120 samples)
    labels = []
    # first 40 human (0)
    labels.extend([0] * 40)
    # next 40 AI (1)
    labels.extend([1] * 40)
    # next 40 human (0)
    labels.extend([0] * 40)
    return np.array(labels, dtype=int)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--single-line', action='store_true', help='Print only a single line: accuracy precision recall f1 (4 decimals)')
    args = ap.parse_args()

    if not CSV_PATH.exists():
        print('CSV not found at', CSV_PATH)
        sys.exit(1)
    if not MODEL_PATH.exists():
        print('Model not found at', MODEL_PATH, '\nTrain first with run_detector.py --train')
        sys.exit(1)

    texts_all = load_texts(CSV_PATH)
    if len(texts_all) < 120:
        print(f'CSV contains only {len(texts_all)} samples, but 120 are required (40+40+40).')
        sys.exit(1)
    texts = texts_all[:120]
    y_true = build_labels_for_120()

    with open(MODEL_PATH, 'rb') as f:
        detector = pickle.load(f)

    # avoid tensorflow import issues when importing run_detector/transformers
    try:
        import tensorflow as _tf
    except Exception:
        import types
        import sys as _sys
        import importlib.util as _spec
        tf = types.ModuleType('tensorflow')
        tf.__version__ = '0.0'
        tf.__spec__ = _spec.spec_from_loader('tensorflow', loader=None)
        _sys.modules['tensorflow'] = tf

    # lazy import of embedding utilities from run_detector
    from run_detector import load_model_and_tokenizer, predict_texts, get_device

    tokenizer, model = load_model_and_tokenizer(detector.get('model_name', 'roberta-base'))
    device = get_device()

    print(f'Classifying {len(texts)} samples on device {device}...')
    probs = predict_texts(texts, detector, tokenizer, model, device)
    y_pred = (probs > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)

    # use macro averages as the overall precision/recall/f1
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    if args.single_line:
        # print exactly one line: accuracy precision recall f1-score (4 decimals), nothing else
        print(f"{acc:0.4f}    {prec_macro:0.4f}    {rec_macro:0.4f}    {f1_macro:0.4f}")
        return

    # previous detailed output (unchanged)
    precisions, recalls, f1s, supports = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1], zero_division=0)
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    print('\nResults on first 120 samples of 491_Dataset.csv (40 human, 40 AI, 40 human)')
    print('Samples:', len(texts))
    print(f'Accuracy: {acc:.4f}\n')

    # Print a compact combined table for classes + averages
    header = f"{'label':<10}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>12}"
    print(header)
    print('-' * len(header))
    labels = ['human', 'AI']
    for i, lab in enumerate(labels):
        print(f"{lab:<10}{precisions[i]:12.4f}{recalls[i]:12.4f}{f1s[i]:12.4f}{supports[i]:12d}")

    print('-' * len(header))
    print(f"{'macro avg':<10}{prec_macro:12.4f}{rec_macro:12.4f}{f1_macro:12.4f}{sum(supports):12d}")
    print(f"{'micro avg':<10}{prec_micro:12.4f}{rec_micro:12.4f}{f1_micro:12.4f}{sum(supports):12d}")
    print(f"{'weighted':<10}{prec_weighted:12.4f}{rec_weighted:12.4f}{f1_weighted:12.4f}{sum(supports):12d}\n")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    print('Confusion matrix (rows=true, cols=pred):')
    print(cm)

    print('\nDetailed classification report:\n')
    print(classification_report(y_true, y_pred, target_names=['human','AI'], digits=4))

    # Save predictions to CSV for inspection
    out_df = pd.DataFrame({'text': texts, 'y_true': y_true, 'y_pred': y_pred, 'p_ai': probs})
    out_path = ROOT / 'models' / 'predictions_120.csv'
    out_df.to_csv(out_path, index=False)
    print('\nSaved predictions to', out_path)

    print('\nNote: accuracy is the overall fraction of correct predictions (TP+TN / total).')
    print('Precision is the fraction of predicted-AI that were actually AI (TP / (TP+FP)). They are different metrics.')


if __name__ == '__main__':
    main()

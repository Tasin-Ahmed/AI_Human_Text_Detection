import os
import sys
import pickle
from pathlib import Path
import math
import argparse

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# prevent transformers from importing TensorFlow (avoids TF DLL issues on some systems)
import os
import sys
import types
import importlib.util
# prevent transformers from importing TensorFlow (avoids TF DLL issues on some systems)
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
# If tensorflow import would fail (common on some Windows setups), inject a dummy module so
# transformers can be imported without loading TF native runtime. Make sure the dummy
# module has a valid __spec__ so importlib.find_spec('tensorflow') won't raise.
if 'tensorflow' not in sys.modules:
    try:
        import tensorflow as _tf  # attempt real import
    except Exception:
        tf = types.ModuleType('tensorflow')
        tf.__version__ = '0.0'
        # create a ModuleSpec so importlib.util.find_spec won't fail
        tf.__spec__ = importlib.util.spec_from_loader('tensorflow', loader=None)
        sys.modules['tensorflow'] = tf

from transformers import RobertaTokenizer, RobertaModel


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_and_tokenizer(model_name='roberta-base'):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name, output_attentions=False)
    model.eval()
    return tokenizer, model


def embed_texts(texts, tokenizer, model, device, batch_size=16, max_length=512):
    embs = []
    model.to(device)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            input_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            last = out.last_hidden_state  # (B, T, D)
            # mean pool using attention mask
            mask = attention_mask.unsqueeze(-1)
            summed = (last * mask).sum(1)
            lens = mask.sum(1).clamp(min=1)
            mean_pooled = (summed / lens).cpu().numpy()
            embs.append(mean_pooled)
    embs = np.vstack(embs)
    return embs


def gather_gpt3d_pairs(data_dir):
    # Look for CSV files in data/gpt3d and use columns that match known patterns
    data_dir = Path(data_dir)
    rows = []
    for p in data_dir.glob('*.csv'):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        # try to find columns for human and gen completions
        if 'gold_completion' in df.columns and 'gen_completion' in df.columns:
            for _, r in df[['gold_completion', 'gen_completion']].iterrows():
                rows.append((r['gold_completion'], 0))
                rows.append((r['gen_completion'], 1))
        else:
            # fallback: try columns with 'human' or 'gen' substrings
            cols = df.columns.str.lower()
            human_cols = [c for c in df.columns if 'gold' in c or 'human' in c]
            gen_cols = [c for c in df.columns if 'gen' in c or 'model' in c]
            if human_cols and gen_cols:
                for _, r in df.iterrows():
                    rows.append((str(r[human_cols[0]]), 0))
                    rows.append((str(r[gen_cols[0]]), 1))
    if not rows:
        raise FileNotFoundError(f'No suitable GPT3D CSVs found in {data_dir}')
    texts, labels = zip(*rows)
    return list(texts), np.array(labels, dtype=int)


def train_and_save(data_dir, model_path, sample_limit=2000, random_state=42):
    print('Collecting data...')
    texts, labels = gather_gpt3d_pairs(data_dir)
    # shuffle
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(texts))
    texts = [texts[i] for i in idx]
    labels = labels[idx]

    # limit for quick runs
    if sample_limit is not None and sample_limit * 2 < len(texts):
        texts = texts[: sample_limit*2]
        labels = labels[: sample_limit*2]

    device = get_device()
    print('Using device:', device)
    tokenizer, model = load_model_and_tokenizer()

    print('Embedding texts (this will download the model if not cached) ...')
    embs = embed_texts(texts, tokenizer, model, device, batch_size=16)

    print('Training classifier...')
    X_train, X_test, y_train, y_test = train_test_split(embs, labels, test_size=0.2, random_state=random_state, stratify=labels)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(X_train_s, y_train)

    preds = clf.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    bal = balanced_accuracy_score(y_test, preds)
    print(f'Test accuracy: {acc:.3f}, balanced_acc: {bal:.3f}')

    # save tokenizer, model not needed to save (we'll re-load), save classifier + scaler
    with open(model_path, 'wb') as f:
        pickle.dump({'clf': clf, 'scaler': scaler, 'model_name': 'roberta-base'}, f)
    print('Saved detector to', model_path)
    return model_path


def load_detector(model_path):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data


def predict_texts(texts, detector, tokenizer=None, model=None, device=None):
    if device is None:
        device = get_device()
    if tokenizer is None or model is None:
        tokenizer, model = load_model_and_tokenizer(detector.get('model_name', 'roberta-base'))
    embs = embed_texts(texts, tokenizer, model, device, batch_size=8)
    Xs = detector['scaler'].transform(embs)
    probs = detector['clf'].predict_proba(Xs)[:, 1]
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true', help='Train detector from data/gpt3d CSVs')
    ap.add_argument('--data-dir', default=str(ROOT / 'data' / 'gpt3d'))
    ap.add_argument('--model-out', default=str(MODELS_DIR / 'detector_roberta.pkl'))
    ap.add_argument('--classify', nargs='+', help='Classify one or more input texts (pass as separate args)')
    args = ap.parse_args()

    if args.train:
        print('Training detector from', args.data_dir)
        train_and_save(args.data_dir, args.model_out)
        return

    if args.classify:
        model_path = args.model_out
        if not Path(model_path).exists():
            print('Model not found at', model_path, '\nTrain first with --train')
            sys.exit(1)
        detector = load_detector(model_path)
        tokenizer, model = load_model_and_tokenizer(detector.get('model_name', 'roberta-base'))
        probs = predict_texts(args.classify, detector, tokenizer, model, get_device())
        for t, p in zip(args.classify, probs):
            print('---')
            print('TEXT:', t)
            print(f'P(ai) = {p:.3f} (threshold 0.5) =>', 'AI' if p>0.5 else 'HUMAN')
        return

    ap.print_help()


if __name__ == '__main__':
    main()

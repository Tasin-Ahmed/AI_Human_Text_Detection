#!/usr/bin/env python3
"""Train and evaluate a simple text classifier (bag-of-words + logistic regression).

Usage example:
	python scripts/train_eval.py --data 491_Dataset.csv --out_dir outputs --model_name lr_bow --max_features 5000

This script accepts CSV or JSONL where the text column is named one of: 'text', 'writing', 'Writing'.
It expects a label column named one of: 'Type', 'type', 'label'. For labels, values like 'Human' and 'AI'
are mapped to 0 and 1 respectively.

The script will perform a stratified train/test split (default 60/40), train a LogisticRegression, save
the fitted vectorizer and model as joblib files in the output directory, and write metrics to JSON.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def find_text_column(df: pd.DataFrame):
	candidates = [c for c in df.columns if c.lower() in ("text", "writing", "content", "sentence")]
	if candidates:
		return candidates[0]
	# fallback: longest-text heuristic
	lengths = {c: df[c].astype(str).map(len).median() for c in df.columns}
	# choose column with largest median length
	best = max(lengths.items(), key=lambda x: x[1])[0]
	return best


def find_label_column(df: pd.DataFrame):
	candidates = [c for c in df.columns if c.lower() in ("type", "label", "y")]
	if candidates:
		return candidates[0]
	# fallback: try 'Writter' or 'Writer' unlikely, but check
	for alt in ("type", "Type", "Label", "label"):
		if alt in df.columns:
			return alt
	return None


def prepare_labels(raw):
	# map common string labels to binary 0/1
	if pd.api.types.is_numeric_dtype(raw):
		return raw.astype(int).to_numpy()
	s = raw.astype(str).str.lower().str.strip()
	mapping = {}
	# human-like -> 0, ai/gpt/robot -> 1
	human_keys = {"human", "h", "real", "author", "writer"}
	ai_keys = {"ai", "gpt", "machine", "bot", "generated", "fake"}
	out = []
	for v in s:
		if v in human_keys:
			out.append(0)
		elif any(k in v for k in ai_keys):
			out.append(1)
		else:
			# default: try to parse integers
			try:
				out.append(int(v))
			except Exception:
				# if unknown, assign nan and handle later
				out.append(np.nan)
	arr = np.array(out)
	if np.isnan(arr).any():
		# try direct equality to 'human' / 'ai' as words
		arr2 = s.map(lambda x: 0 if x == 'human' else (1 if x == 'ai' else np.nan)).to_numpy()
		mask = ~np.isnan(arr)
		arr[~mask] = arr2[~mask]
	return arr.astype(float)


def main(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", required=True, help="Path to CSV or JSONL dataset")
	parser.add_argument("--out_dir", required=True, help="Directory to save models and metrics")
	parser.add_argument("--model_name", default="lr_bow", help="Base name for saved files")
	parser.add_argument("--max_features", type=int, default=5000, help="Max features for vectorizer")
	parser.add_argument("--use_tfidf", action="store_true", help="Use TF-IDF instead of raw counts")
	parser.add_argument("--test_size", type=float, default=0.4, help="Fraction to hold out for test (0-1)")
	parser.add_argument("--random_state", type=int, default=42)
	args = parser.parse_args(argv)

	data_path = Path(args.data)
	if not data_path.exists():
		print(f"ERROR: data file not found: {data_path}")
		sys.exit(2)

	# read
	if data_path.suffix.lower() in (".csv", ".tsv"):
		df = pd.read_csv(data_path)
	else:
		# try jsonl
		try:
			df = pd.read_json(data_path, lines=True)
		except Exception as e:
			print("Failed to read data file. Supported: CSV, JSONL. Error:", e)
			raise

	if df.shape[0] == 0:
		print("No rows in dataset")
		sys.exit(1)

	text_col = find_text_column(df)
	label_col = find_label_column(df)
	if label_col is None:
		print("Could not find a label column in your dataset. Please include a column named 'Type' or 'label'.")
		print("Columns found:", list(df.columns))
		sys.exit(1)

	texts = df[text_col].astype(str).tolist()
	raw_labels = df[label_col]
	labels_pre = prepare_labels(raw_labels)
	# if mapping produced NaNs, try direct mapping: 'Human'->0, others->1
	if np.isnan(labels_pre).any():
		labels_pre = pd.Series(raw_labels.astype(str)).map(lambda x: 0 if x.lower().strip() == 'human' else 1).to_numpy()

	y = labels_pre.astype(int)

	# split train/test preserving class balance where possible.
	# For very small train fractions (e.g. 10%), stratified splitting can fail or
	# produce a training set containing only one class. We first try a stratified
	# split, and if that would yield a single-class train set, fall back to a
	# deterministic sampling that ensures at least one sample per class in train.
	n = len(texts)
	n_train_desired = int(round((1.0 - args.test_size) * n))

	def safe_stratified_split():
		return train_test_split(texts, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

	try:
		X_train, X_test, y_train, y_test = safe_stratified_split()
		unique_train = np.unique(y_train)
		if unique_train.size < np.unique(y).size:
			raise ValueError("Stratified split resulted in missing classes in the train set")
	except Exception:
		# fallback: ensure at least one sample per class in train set
		rng = np.random.RandomState(args.random_state)
		y_arr = np.array(y)
		indices = np.arange(n)
		classes, class_counts = np.unique(y_arr, return_counts=True)

		if n_train_desired < classes.size:
			# ensure at least one example per class
			n_train = int(classes.size)
			print(f"Adjusted train size from {n_train_desired} to {n_train} to include one sample per class.")
		else:
			n_train = n_train_desired

		chosen = []
		for c in classes:
			c_idx = indices[y_arr == c]
			if c_idx.size == 0:
				continue
			pick = rng.choice(c_idx, size=1, replace=False)
			chosen.append(int(pick[0]))

		remaining_pool = np.setdiff1d(indices, np.array(chosen, dtype=int))
		remaining_needed = max(0, n_train - len(chosen))
		if remaining_needed > 0:
			if remaining_needed > remaining_pool.size:
				# not enough remaining, take all
				extra = remaining_pool.tolist()
			else:
				extra = rng.choice(remaining_pool, size=remaining_needed, replace=False).tolist()
		else:
			extra = []

		train_idx = np.array(list(map(int, chosen)) + list(map(int, extra)), dtype=int)
		test_idx = np.setdiff1d(indices, train_idx)

		X_train = [texts[i] for i in train_idx]
		X_test = [texts[i] for i in test_idx]
		y_train = y_arr[train_idx]
		y_test = y_arr[test_idx]

	# vectorizer
	if args.use_tfidf:
		vec = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))
	else:
		vec = CountVectorizer(max_features=args.max_features, ngram_range=(1, 2))

	clf = LogisticRegression(max_iter=200)

	pipeline = make_pipeline(vec, clf)

	print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
	pipeline.fit(X_train, y_train)

	preds = pipeline.predict(X_test)

	acc = accuracy_score(y_test, preds)
	prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary')

	metrics = {
		'accuracy': float(acc),
		'precision': float(prec),
		'recall': float(rec),
		'f1': float(f1),
		'n_train': int(len(X_train)),
		'n_test': int(len(X_test)),
		'model': args.model_name,
	}

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	vec_path = out_dir / f"{args.model_name}_vectorizer.joblib"
	model_path = out_dir / f"{args.model_name}.joblib"
	metrics_path = out_dir / f"metrics_{args.model_name}.json"

	# save pipeline components
	# Save entire pipeline for convenience
	joblib.dump(pipeline, model_path)
	# also save vectorizer alone
	# extract vectorizer from pipeline
	joblib.dump(vec, vec_path)

	with open(metrics_path, 'w', encoding='utf8') as f:
		json.dump(metrics, f, ensure_ascii=False, indent=2)

	print("Training complete. Metrics:")
	print(json.dumps(metrics, indent=2, ensure_ascii=False))
	print(f"Saved pipeline to: {model_path}")
	print(f"Saved vectorizer to: {vec_path}")
	print(f"Saved metrics to: {metrics_path}")


if __name__ == '__main__':
	main()


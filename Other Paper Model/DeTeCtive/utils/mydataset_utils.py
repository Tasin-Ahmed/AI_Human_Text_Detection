import json
import random
import re
from torch.utils.data import Dataset

# ----------------------------
# Preprocessing
# ----------------------------
def clean_bengali_text(text):
    text = re.sub(r'[।,?!.;:"\'""\(\)\[\]—–\-…]', '', text)
    text = re.sub(r'[^\u0980-\u09FF\s]', '', text)
    return text.strip()

def simple_tokenize(text):
    return text.split()

def remove_stopwords(tokens):
    stopwords = [
        'এবং', 'ও', 'একটি', 'এর', 'এই', 'একজন', '।', 'তথা',
        'সে', 'তার', 'তিনি', 'যে', 'কে', 'কি', 'হয়', 'হয়ে',
        'করে', 'করা', 'হল', 'হলো', 'আর', 'বা', 'না', 'নি'
    ]
    return [word for word in tokens if word not in stopwords and len(word) > 1]

def preprocess_text(text):
    cleaned = clean_bengali_text(text)
    tokens = simple_tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)


# ----------------------------
# Dataset Loader
# ----------------------------
def load_mydataset(path, split_ratio=0.8, seed=42):
    with open(path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    random.seed(seed)
    random.shuffle(all_data)

    split_idx = int(len(all_data) * split_ratio)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    return {"train": train_data, "test": test_data}


# ----------------------------
# Dataset Class
# ----------------------------
class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = preprocess_text(self.data[idx]["text"])
        label = self.data[idx]["label"]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": label
        }

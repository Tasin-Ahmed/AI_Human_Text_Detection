import torch
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import os
from utils.mydataset_utils import load_mydataset, MyDataset

def test_classifier(data_path, model_path="pth/MyDataset_model", batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu"):
    dataset = load_mydataset(data_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    test_dataset = MyDataset(dataset["test"], tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    preds, labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            preds.extend(predictions.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    cm = confusion_matrix(labels, preds)

    # Print to console
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")
    print(f"✅ Confusion Matrix:\n{cm}")

    # Save results to CSV
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame([{
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TrueNeg": cm[0,0],
        "FalsePos": cm[0,1],
        "FalseNeg": cm[1,0],
        "TruePos": cm[1,1]
    }])
    results_path = "results/evaluation_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    print(f"📁 Results saved at {results_path}")

if __name__ == "__main__":
    test_classifier("data/data.json")




# import torch
# from torch.utils.data import DataLoader
# from transformers import RobertaForSequenceClassification, RobertaTokenizer
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
# import pandas as pd
# import os
# from utils.mydataset_utils import load_mydataset, MyDataset


# # ---------------------------
# # 🔍 Device Auto Detection
# # ---------------------------
# def get_device():
#     try:
#         import intel_extension_for_pytorch as ipex  # noqa: F401
#         if torch.xpu.is_available():
#             print("✅ Using Intel GPU (XPU) with IPEX")
#             return torch.device("xpu")
#     except ImportError:
#         pass

#     if torch.cuda.is_available():
#         print("✅ Using NVIDIA GPU (CUDA)")
#         return torch.device("cuda")

#     print("⚙️ Using CPU")
#     return torch.device("cpu")


# def test_classifier(data_path, model_path="pth/MyDataset_model", batch_size=32):
#     device = get_device()

#     dataset = load_mydataset(data_path)
#     tokenizer = RobertaTokenizer.from_pretrained(model_path)

#     test_dataset = MyDataset(dataset["test"], tokenizer)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     model = RobertaForSequenceClassification.from_pretrained(model_path)
#     model.to(device)
#     model.eval()

#     preds, labels = [], []

#     with torch.no_grad():
#         for batch in test_loader:
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             batch_labels = batch["labels"].to(device)

#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#             predictions = torch.argmax(outputs.logits, dim=-1)

#             preds.extend(predictions.cpu().numpy())
#             labels.extend(batch_labels.cpu().numpy())

#     # ---------------------------
#     # 📊 Metrics
#     # ---------------------------
#     acc = accuracy_score(labels, preds)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
#     cm = confusion_matrix(labels, preds)

#     print(f"✅ Accuracy: {acc:.4f}")
#     print(f"✅ Precision: {precision:.4f}")
#     print(f"✅ Recall: {recall:.4f}")
#     print(f"✅ F1 Score: {f1:.4f}")
#     print(f"✅ Confusion Matrix:\n{cm}")

#     # ---------------------------
#     # 💾 Save Results
#     # ---------------------------
#     os.makedirs("results", exist_ok=True)
#     results_df = pd.DataFrame([{
#         "Accuracy": acc,
#         "Precision": precision,
#         "Recall": recall,
#         "F1": f1,
#         "TrueNeg": cm[0, 0],
#         "FalsePos": cm[0, 1],
#         "FalseNeg": cm[1, 0],
#         "TruePos": cm[1, 1]
#     }])
#     results_path = "results/evaluation_results.csv"
#     results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
#     print(f"📁 Results saved at {results_path}")


# if __name__ == "__main__":
#     test_classifier("data/data.json")

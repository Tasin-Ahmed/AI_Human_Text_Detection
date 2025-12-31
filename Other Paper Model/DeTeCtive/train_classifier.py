import torch
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import os
from utils.mydataset_utils import load_mydataset, MyDataset

def train_classifier(data_path, model_name="roberta-base", batch_size=16, epochs=3, lr=2e-5, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load dataset
    dataset = load_mydataset(data_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    train_dataset = MyDataset(dataset["train"], tokenizer)
    test_dataset = MyDataset(dataset["test"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

    os.makedirs("pth", exist_ok=True)
    model.save_pretrained("pth/MyDataset_model")
    tokenizer.save_pretrained("pth/MyDataset_model")
    print("✅ Model saved at pth/MyDataset_model")

if __name__ == "__main__":
    train_classifier("data/data.json")



# import torch
# from torch.utils.data import DataLoader
# from transformers import RobertaForSequenceClassification, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
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


# def train_classifier(
#     data_path,
#     model_name="roberta-base",
#     batch_size=16,
#     epochs=3,
#     lr=2e-5,
# ):
#     # ---------------------------
#     # ⚙️ Device Setup
#     # ---------------------------
#     device = get_device()

#     # Load dataset
#     dataset = load_mydataset(data_path)
#     tokenizer = RobertaTokenizer.from_pretrained(model_name)

#     train_dataset = MyDataset(dataset["train"], tokenizer)
#     test_dataset = MyDataset(dataset["test"], tokenizer)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     # Model
#     model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
#     model.to(device)

#     optimizer = AdamW(model.parameters(), lr=lr)
#     total_steps = len(train_loader) * epochs
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

#     # ---------------------------
#     # 🏋️ Training Loop
#     # ---------------------------
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in train_loader:
#             optimizer.zero_grad()
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)

#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             total_loss += loss.item()
#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

#     # ---------------------------
#     # 💾 Save Model
#     # ---------------------------
#     os.makedirs("pth", exist_ok=True)
#     model.save_pretrained("pth/MyDataset_model")
#     tokenizer.save_pretrained("pth/MyDataset_model")
#     print("✅ Model saved at pth/MyDataset_model")


# if __name__ == "__main__":
#     train_classifier("data/data.json")

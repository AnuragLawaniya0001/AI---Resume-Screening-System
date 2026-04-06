"""
Step 2: train_bert.py
======================
Fine-tunes BERT for multi-output regression on resume quality scoring.

Architecture:
  BERT (bert-base-uncased)
    └── [CLS] token embedding (768-dim)
         └── Dropout(0.3)
              └── Linear(768 → 256) → ReLU
                   └── Linear(256 → 4)   ← 4 sub-scores
                        └── Sigmoid * 25  ← clamp each to 0–25

Loss  : MSE on all 4 sub-scores simultaneously
Metric: MAE per sub-score + total MAE

Output saved to: model/bert_resume_scorer/
"""

import os, math, json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────

CFG = {
    "model_name"   : "bert-base-uncased",
    "max_len"      : 512,
    "batch_size"   : 8,          # increase to 16 if you have GPU with >8GB VRAM
    "epochs"       : 5,          # 5 epochs good for fine-tune; use 8-10 for better accuracy
    "lr"           : 2e-5,
    "dropout"      : 0.3,
    "max_score"    : 25.0,       # each sub-score max
    "targets"      : ["experience_score","education_score",
                      "skills_score","structure_score"],
    "output_dir"   : "model/bert_resume_scorer",
    "data_path"    : "data/resume_dataset.csv",
    "seed"         : 42,
}

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────────

class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels          # shape (N, 4)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels"        : torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# ── Model ─────────────────────────────────────────────────────────────────────

class BertResumeScorer(nn.Module):
    """
    BERT encoder → custom regression head → 4 sub-scores (0–25 each).
    """
    def __init__(self, bert_name, dropout=0.3, max_score=25.0):
        super().__init__()
        self.bert      = BertModel.from_pretrained(bert_name)
        hidden         = self.bert.config.hidden_size   # 768
        self.max_score = max_score

        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 4),      # 4 sub-scores
            nn.Sigmoid(),           # → 0–1, then scale to 0–25
        )

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output          # [CLS] token, shape (B, 768)
        scores = self.regressor(pooled)     # (B, 4)  in [0, 1]
        return scores * self.max_score      # (B, 4)  in [0, 25]

# ── Training utils ────────────────────────────────────────────────────────────

def mae_per_target(preds, labels):
    """Returns MAE for each of the 4 sub-scores."""
    return (preds - labels).abs().mean(dim=0)   # (4,)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        lbls  = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        preds = model(ids, mask)
        loss  = criterion(preds, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, target_names):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        lbls  = batch["labels"].to(DEVICE)

        preds = model(ids, mask)
        loss  = criterion(preds, lbls)
        total_loss += loss.item()
        all_preds.append(preds.cpu())
        all_labels.append(lbls.cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    maes       = mae_per_target(all_preds, all_labels)
    total_mae  = (all_preds.sum(1) - all_labels.sum(1)).abs().mean()

    print(f"  Val Loss : {total_loss/len(loader):.4f}")
    for i, name in enumerate(target_names):
        print(f"  MAE [{name:18s}]: {maes[i]:.3f} / 25")
    print(f"  MAE [total_score        ]: {total_mae:.3f} / 100")
    return total_loss / len(loader), maes, total_mae

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load data
    print("\n[1/5] Loading dataset ...")
    df = pd.read_csv(CFG["data_path"])
    print(f"  Shape: {df.shape}")
    print(f"  Score mean: {df['total_score'].mean():.1f}")

    texts  = df["resume_text"].tolist()
    labels = df[CFG["targets"]].values.astype(np.float32)

    X_tr_txt, X_val_txt, y_tr, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=CFG["seed"])
    print(f"  Train: {len(X_tr_txt)} | Val: {len(X_val_txt)}")

    # 2. Tokenizer
    print("\n[2/5] Loading tokenizer ...")
    tokenizer = BertTokenizer.from_pretrained(CFG["model_name"])

    train_ds = ResumeDataset(X_tr_txt, y_tr,  tokenizer, CFG["max_len"])
    val_ds   = ResumeDataset(X_val_txt, y_val, tokenizer, CFG["max_len"])

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    # 3. Model
    print("\n[3/5] Building model ...")
    model = BertResumeScorer(CFG["model_name"],
                             dropout=CFG["dropout"],
                             max_score=CFG["max_score"]).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable:,}")

    # 4. Training
    print("\n[4/5] Training ...")
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG["epochs"])

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, CFG["epochs"] + 1):
        print(f"\n── Epoch {epoch}/{CFG['epochs']} ──────────────────────────")
        tr_loss = train_epoch(model, train_loader, optimizer, criterion)
        print(f"  Train Loss: {tr_loss:.4f}")
        val_loss, maes, total_mae = evaluate(model, val_loader, criterion, CFG["targets"])
        scheduler.step()

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss,
            "total_mae": float(total_mae),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(CFG["output_dir"], exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(CFG["output_dir"], "pytorch_model.bin"))
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")

    # 5. Save config + tokenizer
    print("\n[5/5] Saving artifacts ...")
    tokenizer.save_pretrained(CFG["output_dir"])

    model_cfg = {
        "bert_name"  : CFG["model_name"],
        "max_len"    : CFG["max_len"],
        "dropout"    : CFG["dropout"],
        "max_score"  : CFG["max_score"],
        "targets"    : CFG["targets"],
        "best_val_loss": best_val_loss,
        "history"    : history,
    }
    with open(os.path.join(CFG["output_dir"], "scorer_config.json"), "w") as f:
        json.dump(model_cfg, f, indent=2)

    print(f"\n  Saved to: {CFG['output_dir']}/")
    print(f"    ├── pytorch_model.bin")
    print(f"    ├── vocab.txt  (tokenizer)")
    print(f"    ├── tokenizer_config.json")
    print(f"    └── scorer_config.json")
    print(f"\n  Best Val Loss : {best_val_loss:.4f}")
    print(f"  Best Total MAE: {history[-1]['total_mae']:.2f} / 100")
    print("\nTraining complete ✓")


if __name__ == "__main__":
    main()

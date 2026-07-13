import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, matthews_corrcoef
from tqdm import tqdm
import numpy as np
import argparse

from model import Contrastive_learning_layer


# =========================
# Loss
# =========================
class TripletCosineLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor, negative)
        return torch.clamp(neg_sim - pos_sim + self.margin, min=0.0).mean()


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)


# =========================
# Dataset loader
# =========================
def get_ds(train_anchor, train_pos, train_neg,
           test_enzy, test_smiles, test_y,
           batch_size):

    train_ds = TensorDataset(
        torch.load(train_anchor),
        torch.load(train_pos),
        torch.load(train_neg)
    )

    test_ds = TensorDataset(
        torch.load(test_enzy),
        torch.load(test_smiles),
        torch.load(test_y)
    )

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    )


# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate(classifier, embedding_model, loader, device):

    classifier.eval()
    embedding_model.eval()

    y_true, y_pred, y_prob = [], [], []

    for enzy, smiles, y in loader:
        enzy, smiles = enzy.to(device), smiles.to(device)
        y = y.float().unsqueeze(1).to(device)

        e = embedding_model.encode_enzy(enzy)
        s = embedding_model.encode_smiles(smiles)

        diff = torch.abs(e - s)
        prob = classifier(diff)
        pred = (prob > 0.5).float()

        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        y_prob.append(prob.cpu().numpy())

    y_true = np.concatenate(y_true).ravel()
    y_pred = np.concatenate(y_pred).ravel()
    y_prob = np.concatenate(y_prob).ravel()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    return acc, auc, mcc, np.array([[tn, fp], [fn, tp]])


# =========================
# Fine-tuning main
# =========================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_anchor", required=True)
    parser.add_argument("--train_positive", required=True)
    parser.add_argument("--train_negative", required=True)

    parser.add_argument("--test_enzy", required=True)
    parser.add_argument("--test_smiles", required=True)
    parser.add_argument("--test_y", required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--ckpt", default="best_model_finetuned.pt")
    parser.add_argument("--save_path", default="best_finetuned_model_6.pt")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader = get_ds(
        args.train_anchor,
        args.train_positive,
        args.train_negative,
        args.test_enzy,
        args.test_smiles,
        args.test_y,
        args.batch_size
    )

    # -------- load model --------
    embedding_model = Contrastive_learning_layer().to(device)
    classifier = MLPClassifier(128).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    embedding_model.load_state_dict(ckpt["embedding"])
    classifier.load_state_dict(ckpt["classifier"])

    print("✅ Loaded pretrained model:", args.ckpt)

    optimizer = torch.optim.Adam(
        list(embedding_model.parameters()) +
        list(classifier.parameters()),
        lr=args.lr
    )

    triplet_loss = TripletCosineLoss(0.2)
    bce_loss = nn.BCELoss()

    # =========================
    # Best record
    # =========================
    best_auc = 0.0
    best_acc = 0.0
    best_mcc = 0.0
    best_epoch = 0

    # =========================
    # Fine-tuning loop
    # =========================
    for epoch in range(args.epochs):

        embedding_model.train()
        classifier.train()

        total_loss = 0

        for anchor, pos, neg in tqdm(train_loader, desc=f"FT Epoch {epoch}"):

            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

            e = embedding_model.encode_enzy(anchor)
            p = embedding_model.encode_smiles(pos)
            n = embedding_model.encode_smiles(neg)

            loss_tri = triplet_loss(e, p, n)

            prob_p = classifier(torch.abs(e - p))
            prob_n = classifier(torch.abs(e - n))

            loss_cls = (
                bce_loss(prob_p, torch.ones_like(prob_p)) +
                bce_loss(prob_n, torch.zeros_like(prob_n))
            )

            loss = loss_tri + loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # -------- evaluate --------
        acc, auc, mcc, cm = evaluate(
            classifier, embedding_model, test_loader, device
        )

        print(
            f"[FT Epoch {epoch}] "
            f"Loss={total_loss:.4f} | "
            f"ACC={acc:.4f} | "
            f"AUC={auc:.4f} | "
            f"MCC={mcc:.4f}"
        )

        # =========================
        # Save best model (by AUC)
        # =========================
        if acc > best_acc:
            best_auc = auc
            best_acc = acc
            best_mcc = mcc
            best_epoch = epoch

            torch.save({
                "epoch": epoch,
                "embedding": embedding_model.state_dict(),
                "classifier": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_auc": best_auc,
                "best_acc": best_acc,
                "best_mcc": best_mcc,
                "confusion_matrix": cm
            }, args.save_path)

            print(f"🔥 New best model saved at epoch {epoch} | AUC={auc:.4f}")

    # =========================
    # Training finished
    # =========================
    print("\n🎯 Fine-tuning finished")
    print("=================================")
    print(f"Best Epoch : {best_epoch}")
    print(f"Best ACC   : {best_acc:.4f}")
    print(f"Best AUC   : {best_auc:.4f}")
    print(f"Best MCC   : {best_mcc:.4f}")
    print("Model saved to:", args.save_path)

if __name__ == "__main__":
    main()

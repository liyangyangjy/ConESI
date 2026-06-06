import torch
import torch.nn as nn
import numpy as np
import argparse
from model import Contrastive_learning_layer

# === Define the MLP classifier (consistent with the training phase) ===

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=128):
        super(MLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)

# -------------------------- Loss function --------------------------

class TripletCosineLoss(nn.Module):
    def __init__(self, base_margin=0.2, max_epoch=300, adaptive=False):
        super().__init__()
        self.base_margin = base_margin
        self.max_epoch = max_epoch
        self.adaptive = adaptive
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, anchor, positive, negative):
        if self.adaptive:
            margin = self.base_margin * (1 - self.current_epoch / self.max_epoch)
        else:
            margin = self.base_margin

        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor, negative)
        loss = torch.clamp(neg_sim - pos_sim + margin, min=0.0)
        return loss.mean()




# === Load the model and make predictions. ===
def predict(enzy_pt, smiles_pt, model_path, output_file=None, threshold=0.5, device=None):
    # Device settings
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    print("Using device:", device)

    # === Load data ===
    enzy_embeddings = torch.load(enzy_pt, map_location=device)
    smiles_embeddings = torch.load(smiles_pt, map_location=device)
    print(f"Loaded enzyme embeddings: {enzy_embeddings.shape}")
    print(f"Loaded molecule embeddings: {smiles_embeddings.shape}")

    assert enzy_embeddings.shape[0] == smiles_embeddings.shape[0], \
        "Error: The number of samples for enzyme and molecule must be consistent!"

    # === Load model ===
    checkpoint = torch.load(model_path, map_location=device)
    embedding_model = Contrastive_learning_layer().to(device)
    classifier = MLPClassifier(input_dim=128).to(device)
    state_dict = checkpoint["classifier"]

    # Automatically infer the input dimension.
    first_weight = state_dict[list(state_dict.keys())[0]]
    input_dim = first_weight.shape[1]

    print("Detected classifier input_dim =", input_dim)

    classifier.load_state_dict(state_dict)

    embedding_model.load_state_dict(checkpoint["embedding"])
    classifier.load_state_dict(checkpoint["classifier"])
    embedding_model.eval()
    classifier.eval()

    # === Predict ===
    probs, preds = [], []

    with torch.no_grad():
        for i in range(enzy_embeddings.shape[0]):
            enzy = enzy_embeddings[i].unsqueeze(0).to(device)
            smi = smiles_embeddings[i].unsqueeze(0).to(device)

            emb_enzy, emb_smiles = embedding_model(enzy, smi)
            emb_diff = torch.abs(emb_enzy - emb_smiles)
            prob = classifier(emb_diff).item()
            pred = 1 if prob > threshold else 0

            probs.append(prob)
            preds.append(pred)

    # === Output results ===
    results = {
        "Probability": probs,
        "Prediction": preds
    }

    if output_file:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"✅ Predictions saved to {output_file}")
    else:
        print("=== Prediction Results ===")
        for i, (p, pr) in enumerate(zip(probs, preds)):
            print(f"Sample {i+1}: Prob = {p:.4f}, Pred = {pr}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict enzyme-substrate interaction using trained model.")
    parser.add_argument("--enzy_pt", type=str, required=True, help="Path to enzyme embeddings (.pt)")
    parser.add_argument("--smiles_pt", type=str, required=True, help="Path to molecule embeddings (.pt)")
    parser.add_argument("--model_path", type=str, default="best_model.pt", help="Path to trained model checkpoint")
    parser.add_argument("--output_file", type=str, default=None, help="Optional: save predictions to CSV file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for classification")
    args = parser.parse_args()

    predict(
        enzy_pt=args.enzy_pt,
        smiles_pt=args.smiles_pt,
        model_path=args.model_path,
        output_file=args.output_file,
        threshold=args.threshold
    )

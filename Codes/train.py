from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix

import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from model import Contrastive_learning_layer

import warnings
from tqdm import tqdm
import os
from pathlib import Path
import argparse

def get_ds(train_data_enzy, train_data_smiles, train_data_y, val_data_enzy, val_data_smiles, val_data_y,test_data_enzy, test_data_smiles, test_data_y, batch_size):
    # Load the saved embeddings_results
    ESP_train_df_enzy = torch.load(train_data_enzy,weights_only=False)
    ESP_val_df_enzy = torch.load(val_data_enzy,weights_only=False)
    ESP_test_df_enzy = torch.load(test_data_enzy,weights_only=False)
    print('Dataset size: protein sequence: ', ESP_train_df_enzy.shape, ESP_val_df_enzy.shape, ESP_test_df_enzy.shape)
    # Load the saved embeddings_results
    ESP_train_df_smiles = torch.load(train_data_smiles,weights_only=False)
    ESP_val_df_smiles = torch.load(val_data_smiles,weights_only=False)
    ESP_test_df_smiles = torch.load(test_data_smiles,weights_only=False)
    print('Dataset size: molecules: ', ESP_train_df_smiles.shape, ESP_val_df_smiles.shape, ESP_test_df_smiles.shape)

    y_train = torch.load(train_data_y,weights_only=False)
    y_val = torch.load(val_data_y,weights_only=False)
    y_test = torch.load(test_data_y,weights_only=False)
    print('Dataset size: label: ', y_train.shape,y_val.shape, y_test.shape)

    train_tensor_dataset = TensorDataset(ESP_train_df_enzy,ESP_train_df_smiles, y_train)
    val_tensor_dataset = TensorDataset(ESP_val_df_enzy,ESP_val_df_smiles, y_val)
    test_tensor_dataset = TensorDataset(ESP_test_df_enzy, ESP_test_df_smiles, y_test)

    # Create TensorDataset and DataLoaders
    # batch_size  # 16
    train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,  val_loader, test_loader

def get_ds_for_triplet(train_data_anchor, train_data_positive, train_data_negative, val_data_enzy, val_data_smiles, val_data_y,test_data_anchor, test_data_positive, test_data_negative, batch_size):
    # Load the saved embeddings_results
    ESP_train_df_anchor = torch.load(train_data_anchor,weights_only=False)
    ESP_val_df_enzy = torch.load(val_data_enzy,weights_only=False)
    ESP_test_df_anchor = torch.load(test_data_anchor,weights_only=False)
    print('Dataset size: protein sequence: ', ESP_train_df_anchor.shape, ESP_val_df_enzy.shape, ESP_test_df_anchor.shape)
    # Load the saved embeddings_results
    ESP_train_df_positive = torch.load(train_data_positive,weights_only=False)
    ESP_val_df_smiles = torch.load(val_data_smiles,weights_only=False)
    ESP_test_df_positive = torch.load(test_data_positive,weights_only=False)
    print('Dataset size: molecules: ', ESP_train_df_positive.shape, ESP_val_df_smiles.shape, ESP_test_df_positive.shape)

    ESP_train_df_negative = torch.load(train_data_negative,weights_only=False)
    y_val = torch.load(val_data_y,weights_only=False)
    ESP_test_df_negative = torch.load(test_data_negative,weights_only=False)
    print('Dataset size: label: ', ESP_train_df_negative.shape,y_val.shape, ESP_test_df_negative.shape)

    train_tensor_dataset = TensorDataset(ESP_train_df_anchor,ESP_train_df_positive, ESP_train_df_negative)
    val_tensor_dataset = TensorDataset(ESP_val_df_enzy,ESP_val_df_smiles, y_val)
    test_tensor_dataset = TensorDataset(ESP_test_df_anchor, ESP_test_df_positive, ESP_test_df_negative)

    # Create TensorDataset and DataLoaders
    # batch_size  # 16
    train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,  val_loader, test_loader

def run_validation(model, val_loader,loss_fn, device):
    model.eval()
    loss_sum = 0
    num_batch = len(val_loader)
    total_y_true=[]
    total_y_pred=[]
    total_y_prob=[]
    for ESP_val_df_enzy,ESP_val_df_smiles, y_val in val_loader:

        ESP_val_df_enzy = ESP_val_df_enzy.to(device)
        ESP_val_df_smiles = ESP_val_df_smiles.to(device)
        y_val = y_val.squeeze(1).to(device)

        refined_enzy_embed, refined_smiles_embed = model(ESP_val_df_enzy,ESP_val_df_smiles)
        cos_sim = torch.nn.functional.cosine_similarity(refined_enzy_embed, refined_smiles_embed, dim=1)
        #loss = loss_fn(cos_sim, y_val).detach().cpu().numpy()
        #loss_sum = loss_sum + loss # count all the loss in the training process
        y_pred = (cos_sim > 0.5).float().cpu().numpy() # if score > 0.5, assign label 1 otherwise 0, transfer to cpu as numpy
        total_y_true.append(y_val.cpu().numpy())
        total_y_pred.append(y_pred)
        total_y_prob.append(cos_sim.detach().cpu().numpy())

    #loss_sum = loss_sum/num_batch # get the overall average loss (Notice: this method is not 100% accurate)

    arrange_y_true = np.concatenate(total_y_true, axis=0)
    arrange_y_pred = np.concatenate(total_y_pred, axis=0)
    arrange_y_prob = np.concatenate(total_y_prob, axis=0)
    tn,fp,fn,tp = confusion_matrix(arrange_y_true, arrange_y_pred).ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    specificity = tn/(tn+fp)
    sensitivity = tp/(tp+fn)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    bacc = (sensitivity + specificity)/2
    MCC = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    AUC = roc_auc_score(arrange_y_true, arrange_y_prob)
    f1 = 2*precision*recall/(precision+recall)
    #print("loss_sum= ",loss_sum, "ACC= ",acc, "bacc= ",bacc, "precision= ",precision,"specificity= ",specificity, "sensitivity= ",sensitivity, "recall= ",recall, "MCC= ",MCC, "AUC= ",AUC, "f1= ",f1)
    #return loss_sum, acc, bacc   # , precision, sensitivity, recall, MCC, AUC, f1
    print("ACC= ",acc, "bacc= ",bacc, "precision= ",precision,"specificity= ",specificity, "sensitivity= ",sensitivity, "recall= ",recall, "MCC= ",MCC, "AUC= ",AUC, "f1= ",f1)
    return acc, bacc   # , precision, sensitivity, recall, MCC, AUC, f1

#================================================================================================lyy
class TripletCosineLoss(nn.Module):
    def __init__(self, margin=0.2):  
        super(TripletCosineLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Cosine similarity: the higher, the more similar (range: [-1, 1])
        pos_sim = torch.nn.functional.cosine_similarity(anchor, positive)
        neg_sim = torch.nn.functional.cosine_similarity(anchor, negative)
        
        # Loss encourages: pos_sim > neg_sim + margin
        losses = torch.clamp(neg_sim - pos_sim + self.margin, min=0.0)
        return losses.mean()

# Define a  classifier model (MLP)
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.classifier(x)

# Evaluate classifier
def evaluate_classifier(classifier, embedding_model, loader, device):
    classifier.eval()
    embedding_model.eval()
    total_y_true = []
    total_y_pred = []
    total_y_prob = []

    with torch.no_grad():
        for enzy, smiles, y in loader:
            enzy = enzy.to(device)
            smiles = smiles.to(device)
            y = y.float().unsqueeze(1).to(device)

            emb_enzy, emb_smiles = embedding_model(enzy, smiles)
            emb = torch.abs(emb_enzy - emb_smiles)
            prob = classifier(emb)
            pred = (prob > 0.5).float()

            total_y_true.append(y.cpu().numpy())
            total_y_pred.append(pred.cpu().numpy())
            total_y_prob.append(prob.cpu().numpy())

    
    y_true = np.concatenate(total_y_true).astype(int).ravel()
    y_pred = np.concatenate(total_y_pred).astype(int).ravel()
    y_prob = np.concatenate(total_y_prob).ravel()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall = sensitivity
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    bacc = (sensitivity + specificity) / 2
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) + 1e-8)
    auc = roc_auc_score(y_true, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("Test classifier results: ")
    print("ACC= ", acc, "bacc= ", bacc, "precision= ", precision,
          "specificity= ", specificity, "sensitivity= ", sensitivity,
          "recall= ", recall, "MCC= ", mcc, "AUC= ", auc, "f1= ", f1)

    return acc, np.array([[tn, fp], [fn, tp]])


@torch.no_grad()
def evaluate_valid_set(embedding_model, classifier, dataloader, device):
    embedding_model.eval()
    classifier.eval()
    total_y_true, total_y_pred, total_y_prob = [], [], []

    for enzy, smi, y_val in dataloader:
        enzy = enzy.to(device)
        smi = smi.to(device)
        y_val = y_val.float().unsqueeze(1).to(device)

        emb_enzy, emb_smiles = embedding_model(enzy, smi)
        emb_diff = torch.abs(emb_enzy - emb_smiles)
        prob = classifier(emb_diff)
        pred = (prob > 0.5).float()

        total_y_true.append(y_val.cpu().numpy())
        total_y_pred.append(pred.cpu().numpy())
        total_y_prob.append(prob.cpu().numpy())

    y_true = np.concatenate(total_y_true).astype(int).ravel()
    y_pred = np.concatenate(total_y_pred).astype(int).ravel()
    y_prob = np.concatenate(total_y_prob)

    print("DEBUG y_true unique:", np.unique(y_true))
    print("DEBUG y_pred unique:", np.unique(y_pred))
    print("DEBUG shapes:", y_true.shape, y_pred.shape)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    print("\n[Validation Summary]")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", np.array([[tn, fp], [fn, tp]]))

    return acc, np.array([[tn, fp], [fn, tp]])
#================================================================================================lyy

#================================================================================================lyy
# classifeier
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

#================================================================================================lyy

def train():
    # Define the device
    #=========================================================================================================================================lyy
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    print("Using device:", device)
    #=========================================================================================================================================lyy


    # --- argparse ---
    parser = argparse.ArgumentParser(description="Generate embeddings from a file.")
    parser.add_argument('--train_anchor', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--train_positive', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--train_negative', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--val_enzy', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--val_smiles', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--val_y', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--test_enzy', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--test_smiles', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--test_y', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size during training")
    parser.add_argument('--learning_rate', type=float, default=1e-03, help="batch size during training")
    args = parser.parse_args()

    # --- load dataset ---
    train_loader,  val_loader, test_loader = get_ds_for_triplet(
        args.train_anchor, args.train_positive, args.train_negative,
        args.val_enzy, args.val_smiles, args.val_y,
        args.test_enzy, args.test_smiles, args.test_y,
        args.batch_size
    )

    # --- model + classifier ---
    embedding_model = Contrastive_learning_layer().to(device)
    classifier = MLPClassifier(input_dim=128).to(device)

    optimizer = torch.optim.Adam(
        list(embedding_model.parameters()) + list(classifier.parameters()),
        lr=args.learning_rate
    )

    triplet_loss_fn = TripletCosineLoss(margin=0.2).to(device)
    bce_loss_fn = nn.BCELoss().to(device)
    best_epoch = -1
    best_acc = 0.5

    for epoch in range(500):
        embedding_model.train()
        classifier.train()
        epoch_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch") as tepoch:
            for anchor, pos, neg in tepoch:
                anchor = anchor.to(device)
                pos = pos.to(device)
                neg = neg.to(device)

                # Triplet embeddings
                emb_anchor, emb_pos = embedding_model(anchor, pos)
                _, emb_neg = embedding_model(anchor, neg)

                # --- Triplet Loss ---
                loss_triplet = triplet_loss_fn(emb_anchor, emb_pos, emb_neg)

                # --- Classification Loss ---
                emb_diff_pos = torch.abs(emb_anchor - emb_pos)
                prob_pos = classifier(emb_diff_pos)
                label_pos = torch.ones_like(prob_pos)
                loss_bce_pos = bce_loss_fn(prob_pos, label_pos)
                emb_diff_neg = torch.abs(emb_anchor - emb_neg)
                prob_neg = classifier(emb_diff_neg)
                label_neg = torch.zeros_like(prob_neg)
                loss_bce_neg = bce_loss_fn(prob_neg, label_neg)

                # --- 总 loss ---
                loss = loss_triplet + loss_bce_pos + loss_bce_neg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        # --- Validation ---
        acc_val, cm = evaluate_valid_set(embedding_model, classifier, val_loader, device)
        print(f"Epoch {epoch} done. Loss: {epoch_loss:.4f}")
        print(f"Val ACC: {acc_val:.4f}, Best ACC: {best_acc:.4f} (epoch {best_epoch})")

        if acc_val > best_acc:
            best_acc = acc_val
            best_epoch = epoch
            torch.save({
                "embedding_model": embedding_model.state_dict(),
                "classifier": classifier.state_dict()
            }, "best_model.pt")
    # --- Test best model ---
    print("\nTesting best model...")
    checkpoint = torch.load("best_model.pt", map_location=device)
    embedding_model.load_state_dict(checkpoint["embedding_model"])
    classifier.load_state_dict(checkpoint["classifier"])
    evaluate_classifier(classifier, embedding_model, test_loader, device)


if __name__ == '__main__':
    train()

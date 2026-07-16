import torch
import torch.nn as nn
import torch.nn.functional as F


class Contrastive_learning_layer(nn.Module):
    def __init__(self):
        super().__init__()

        self.enzy_refine_layer_1 = nn.Linear(2560, 2560)
        self.enzy_refine_layer_2 = nn.Linear(2560, 128)

        self.smiles_refine_layer_1 = nn.Linear(768, 768)
        self.smiles_refine_layer_2 = nn.Linear(768, 128)

        self.relu = nn.ReLU()
        self.batch_norm_enzy = nn.BatchNorm1d(2560)
        self.batch_norm_smiles = nn.BatchNorm1d(768)
        self.batch_norm_shared = nn.BatchNorm1d(128)

    # --------------------------------------------------------
    # Single-input mode: Triplet Loss uses (anchor / pos / neg).
    # --------------------------------------------------------
    def encode_enzy(self, x):
        x = self.enzy_refine_layer_1(x)
        x = self.batch_norm_enzy(x)
        x = self.relu(x)
        x = self.enzy_refine_layer_2(x)
        x = self.batch_norm_shared(x)
        return F.normalize(x, dim=1)

    def encode_smiles(self, x):
        x = self.smiles_refine_layer_1(x)
        x = self.batch_norm_smiles(x)
        x = self.relu(x)
        x = self.smiles_refine_layer_2(x)
        x = self.batch_norm_shared(x)
        return F.normalize(x, dim=1)

    # --------------------------------------------------------
    # Dual-input mode: used for validation and testing.
    # --------------------------------------------------------
    def encode_pair(self, enzy, smiles):
        return self.encode_enzy(enzy), self.encode_smiles(smiles)

    def forward(self, enzy, smiles):
        return self.encode_pair(enzy, smiles)

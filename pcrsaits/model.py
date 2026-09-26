from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class PCRExampleDataset(Dataset):
    """Exact tensor-boundary behavior of the legacy PCR training dataset."""

    def __init__(self, features, targets, base_errors, easy_mask):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.base_errors = torch.tensor(base_errors, dtype=torch.float32)
        self.easy_mask = torch.tensor(
            easy_mask.astype("float32"), dtype=torch.float32
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.targets[idx],
            self.base_errors[idx],
            self.easy_mask[idx],
        )


class PCRResidualNet(nn.Module):
    """Legacy v7.1 residual network.

    Important: the mask head is retained because v7.1 contains the
    `pcrsaitsv14_masked_residual` ablation. The proposed paper path uses
    `direct_residual=True`, so the learned mask head is bypassed.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(hidden_dim, 1)
        self.mask_head = nn.Linear(hidden_dim, 1)

        nn.init.xavier_uniform_(self.backbone[0].weight, gain=0.5)
        nn.init.zeros_(self.backbone[0].bias)
        nn.init.xavier_uniform_(self.backbone[2].weight, gain=0.5)
        nn.init.zeros_(self.backbone[2].bias)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.mask_head.weight)
        nn.init.constant_(self.mask_head.bias, -1.0)

    def forward(self, x, direct_residual=False):
        h = self.backbone(x)
        delta = torch.tanh(self.delta_head(h)) * 4.0
        corr_mask = (
            torch.ones_like(delta)
            if direct_residual
            else torch.sigmoid(self.mask_head(h))
        )
        return delta, corr_mask

# GNN moddel architecture
# Mainly uses the host-host and phage-phage similarity edges for message passing while the 
# labelled interaction pairs are only utilized for supervision by the decoder

from typing import Dict
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATv2Conv, to_hetero

# message-passing (GraphSAGE and GATv2 )
class GNNBackbone(nn.Module):
    def __init__(self, channels: int = 256, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.5):
        super().__init__()

        self.drop = nn.Dropout(dropout)
        self.graphsage = nn.ModuleList([
            SAGEConv((-1, -1), channels)
            for _ in range(num_layers)
        ])

        self.gat = nn.ModuleList([
            GATv2Conv(
                (-1, -1),
                channels,
                heads=num_heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for sage, gat in zip(self.graphsage, self.gat):
            # local neighborhood feature aggregation   (GraphSAGE)
            h = sage(x, edge_index)
            h = self.drop(F.gelu(h))
            x = x + h

            # atention-based neighborhood feature aggregation (GAT)
            h = gat(x, edge_index)
            h = self.drop(F.gelu(h))
            x = x + h
        return x

# The MLP decoder (uses labeled interaction pairs for prediction)
class MLPClassifier(nn.Module):
    def __init__(self, dim: int = 256, hidden_channels: int = 128, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()

        # phage, host, elementwise product and absolute difference
        in_dim = dim * 4 # 4 concatenated features
        if num_layers >= 2:
            self.mlp = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, 1),
            )
        else:
            self.mlp = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 1),
            )
    # MLP ecodes the edge probabilities from respective node embeddings
    def forward(self, phage_embeddings: torch.Tensor, host_embeddings: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        phage_idx = edge_label_index[0].long()
        host_idx = edge_label_index[1].long()

        phage_z = phage_embeddings[phage_idx]
        host_z = host_embeddings[host_idx]

        # pairwise feature vector construction
        pair_features = torch.cat([
            phage_z,
            host_z,
            phage_z * host_z,
            torch.abs(phage_z - host_z)
            ],
            dim=-1,
            )

        return self.mlp(pair_features).view(-1)

# end-to-end Heterogeneous GNN model for predicting phage-host connections
class Model(nn.Module):
    def __init__(self, data: HeteroData, hidden_channels: int = 256, output_channels: int = 256, 
                 num_gnn_layers: int = 2, num_gnn_heads: int = 2, clf_hidden_channels: int = 128, 
                 num_clf_layers: int = 2, dropout: float = 0.5):
        super().__init__()

        # Residual updates dim sanity checks (it requires equal dimensions)
        if hidden_channels != output_channels:
            raise ValueError(
                "hidden_channels must equal output_channels because the GNN uses residual additions!"
            )

        self.dropout = float(dropout)

        # INPUT PROJECTION LAYERS 
        # 1.projecting raw phage features
        self.phage_lin = nn.Sequential(
            nn.Linear(
                data["phage"].x.size(1),
                hidden_channels,
            ),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        #2.projecting raw host features
        self.host_lin = nn.Sequential(
            nn.Linear(
                data["host"].x.size(1),
                hidden_channels,
            ),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # instantiating GNN message passing backbone
        backbone = GNNBackbone(channels=output_channels, num_layers=num_gnn_layers, num_heads=num_gnn_heads, dropout=dropout,)
        self.gnn = to_hetero(backbone, metadata=data.metadata(), aggr="sum") # converts backbone for het. MP across graph metadata

        #layer normalization after message passing
        self.post_norm = nn.ModuleDict({
            "phage": nn.LayerNorm(output_channels),
            "host": nn.LayerNorm(output_channels),
        })

        # MLP edge decoder
        self.clf = MLPClassifier(
            dim=output_channels,
            hidden_channels=clf_hidden_channels,
            num_layers=num_clf_layers,
            dropout=dropout,
        )

    # projecting raw features and runs message passing
    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        # projecting raw features to hidden space
        x_dict = {
            "phage": self.phage_lin(data["phage"].x),
            "host": self.host_lin(data["host"].x),
        }

        # Message passing through: phage-phage and host-host similarity edges
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        # Layer normalization
        x_dict = {
            node_type: self.post_norm[node_type](embeddings)
            for node_type, embeddings in x_dict.items()
        }

        # final dropout before decoding
        x_dict = {
            node_type: F.dropout(
                embeddings,
                p=self.dropout,
                training=self.training,
            )
            for node_type, embeddings in x_dict.items()
        }

        return x_dict

    # predicting phage-hostinteraction probablities
    def decode(self, z_dict: Dict[str, torch.Tensor], edge_label_index: torch.Tensor) -> torch.Tensor:
        return self.clf(z_dict["phage"], z_dict["host"], edge_label_index)

     # full foward pass (from encoding the node embeddings to decoding the edge logits)
    def forward(self, data: HeteroData, edge_label_index: torch.Tensor) -> torch.Tensor:
        z_dict = self.encode(data)
        return self.decode(z_dict, edge_label_index)


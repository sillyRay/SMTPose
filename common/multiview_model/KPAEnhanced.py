import torch
import numpy as np

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np


import torch
import torch.nn as nn
def build_h36m_spatial_adj():
    """
    Manually constructs the exact same spatial adjacency matrix
    as adj_mx_from_skeleton(Human36mDataset.skeleton()) for Human3.6M (17 joints).
    """
    num_joints = 17
    # Step 1: Define the skeleton as parent-child pairs (from Human3.6M)
    skeleton_bones = [
        (0, 1), (1, 2), (2, 3),  # Right leg
        (0, 4), (4, 5), (5, 6),  # Left leg
        (0, 7), (7, 8), (8, 9), (9, 10),  # Spine to Head
        (8, 11), (11, 12), (12, 13),  # Left arm
        (8, 14), (14, 15), (15, 16)  # Right arm
    ]

    # Step 2: Build the basic binary adjacency matrix (A) and count neighbors
    A = np.zeros((num_joints, num_joints))
    neighbor_count = np.zeros(num_joints, dtype=int)  # Count of *external* neighbors

    for i, j in skeleton_bones:
        A[i, j] = 1
        A[j, i] = 1
        neighbor_count[i] += 1
        neighbor_count[j] += 1

    # Step 3: Apply the specific normalization rule observed in the output
    # - Diagonal (self-loop) is always 1.0
    # - Off-diagonal weight = 1.0 / (number of external neighbors of the source node)
    adj = np.zeros_like(A)
    np.fill_diagonal(adj, 1.0)  # Set self-loops

    for i in range(num_joints):
        for j in range(num_joints):
            if A[i, j] == 1:  # If there's a connection
                if neighbor_count[i] > 0:
                    # Normalize by the number of neighbors of node i
                    adj[i, j] = 1.0 / neighbor_count[i]
                else:
                    adj[i, j] = 1.0  # Should not happen for H36M

    return torch.from_numpy(adj).float()


class LearnableGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(LearnableGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))

        # spatial/temporal local topology
        self.adj = adj

        # simulated spatial/temporal global topology
        self.adj2 = nn.Parameter(torch.ones_like(adj))
        nn.init.constant_(self.adj2, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = self.adj.to(input.device) + self.adj2.to(input.device)

        adj = (adj.T + adj) / 2

        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device)

        output = torch.matmul(adj * E, self.M * h0) + torch.matmul(adj * (1 - E), self.M * h1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class KPA(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(KPA, self).__init__()

        self.gconv =  LearnableGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x

# --- Test the function ---
if __name__ == "__main__":
    adj_manual = build_h36m_spatial_adj()

    print("=== Manually Constructed Spatial Adjacency Matrix ===")
    print("Shape:", adj_manual.shape)
    print(adj_manual)

    # Optional: If you have the original 'adj' from your script, you can compare them
    # assert torch.allclose(adj_manual, original_adj_from_script, atol=1e-4), "Matrices do not match!"
    # print("✅ The manually constructed matrix matches the original one!")
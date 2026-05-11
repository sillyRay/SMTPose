import torch
import torch.nn as nn


class RNNBlockConsistent(nn.Module):
    """
    Bi-LSTM block that projects output back to original input dimension
    to allow easy stacking.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Project bidirectional output (hidden*2) back to input_dim
        self.projection = nn.Linear(hidden_dim * 2, input_dim)

        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        # x: [Batch, Seq, InputDim]
        lstm_out, _ = self.lstm(x)
        # lstm_out: [Batch, Seq, Hidden*2]

        x = self.projection(lstm_out)  # [Batch, Seq, InputDim]
        x = self.norm(x)
        x = self.dropout(x)

        # Residual connection
        x = x + identity
        return x


class Head(nn.Module):
    """Same as original BERT Head: extract center frame and regress to 3D pose"""

    def __init__(self, cfg, in_channels, dropout=0.25, num_joints=17):
        super().__init__()
        self.cfg = cfg
        # Use Linear instead of Conv1d as our data is [B, T, C]
        self.shrink = nn.Linear(in_channels, num_joints * 3)
        self.num_joints = num_joints

    def forward(self, x):
        B, T, C = x.shape
        # Extract center frame
        mid_idx = T // 2
        x = x[:, mid_idx:mid_idx + 1]  # [B, 1, C]

        # Apply Linear layer
        x = self.shrink(x)  # [B, 1, J*3]

        # Reshape to [B, 1, J, 3]
        x = x.view(x.shape[0], 1, self.num_joints, 3)
        return x


class RNNAsBERT(nn.Module):
    """
    Replace ModernTCN with Bi-LSTM for temporal modeling.

    INITIALIZATION MATCHES ORIGINAL:
    self.shrink = RNNAsBERT(cfg, T=cfg.NETWORK.TEMPORAL_LENGTH, dropout=0.1, num_joints=num_joints)

    Input shape:  [B, C1, C2, T, N]
    Output shape: [B, 1, J, 3, N]
    """

    def __init__(self, cfg, T=7, dropout=0.1, num_joints=17):
        super().__init__()
        self.cfg = cfg
        # Ensure we read the same config keys as the original model
        self.hidden = cfg.NETWORK.T_FORMER.NUM_CHANNELS
        self.n_layers = cfg.NETWORK.T_FORMER.NUM_LAYERS
        self.num_joints = num_joints
        self.T = T  # Store T, though logic relies on dynamic shape usually

        # 1. Project input to hidden dim
        # Input after view: [B*N, T, C_in]. C_in comes from cfg.NETWORK.NUM_CHANNELS
        self.input_proj = nn.Linear(cfg.NETWORK.NUM_CHANNELS, self.hidden)

        # 2. RNN Blocks (Stacked Bi-LSTMs)
        self.blocks = nn.ModuleList()
        for i in range(self.n_layers):
            # All layers maintain 'self.hidden' dimension thanks to internal projection
            self.blocks.append(
                RNNBlockConsistent(
                    input_dim=self.hidden,
                    hidden_dim=self.hidden,
                    dropout=dropout
                )
            )

        # 3. Head
        self.head = Head(cfg, in_channels=self.hidden, num_joints=num_joints)

    def set_bn_momentum(self, momentum):
        # RNN uses LayerNorm, but keeping this method for compatibility if called
        pass

    def forward(self, x):
        """
        x: [B, C1, C2, T, N] from FuseView
        Returns: [B, 1, J, 3, N]
        """
        # Handle 5D input
        if len(x.shape) == 5:
            B, C1, C2, T, N = x.shape
            x = x.view(B, -1, T, N)  # [B, C, T, N], C = C1*C2
            B, C, T, N = x.shape

        # Reshape to [B*N, T, C] (RNN expects Batch, Seq, Feature)
        # Original: permute(0, 3, 2, 1) -> [B, N, T, C]
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(B * N, T, C)  # [B*N, T, C]

        # Project to hidden dimension
        x = self.input_proj(x)  # [B*N, T, hidden]

        # Apply RNN blocks
        for block in self.blocks:
            x = block(x)  # [B*N, T, hidden]

        # Apply head (extracts center frame)
        # x is [B*N, T, hidden]
        x = self.head(x)  # [B*N, 1, J, 3]

        # Reshape back to [B, 1, J, 3, N]
        B_N, T_out, J, _ = x.shape
        x = x.view(B, N, T_out, J, 3)
        # Permute to match original output order: [B, T_out, J, 3, N]
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        return x
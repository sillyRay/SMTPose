import torch
import torch.nn as nn


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias),
        nn.BatchNorm1d(out_channels)
    )


class TCNBlock(nn.Module):
    """ModernTCN Block for temporal modeling (no downsampling)"""

    def __init__(self, channels, kernel_size=31, ffn_ratio=2.0, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        # Depthwise Conv over time (groups = channels)
        self.dwconv = conv_bn(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels
        )

        # FFN: expand → contract
        hidden_dim = int(channels * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, 1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, channels, 1, bias=False),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, T]
        identity = x

        # Depthwise Conv —— 仅当 T > 1 时执行
        if x.size(-1) > 1:
            x = self.dwconv(x)
            x = self.norm1(x)
            x = self.dropout(x)
            x = identity + x
        else:
            # T == 1: 跳过 dwconv，保持 x 不变（避免 padding 污染）
            pass  # x remains as identity

        # FFN (always applied, even for T=1)
        identity = x
        x = self.ffn(x)
        x = self.norm2(x)
        x = self.dropout(x)
        x = identity + x

        return x

class Head(nn.Module):
    """Same as original BERT Head: extract center frame and regress to 3D pose"""

    def __init__(self, cfg, in_channels=10, dropout=0.25, channels=2048, num_joints=17):
        super().__init__()
        self.cfg = cfg
        self.shrink = nn.Conv1d(in_channels, num_joints * 3, 1, bias=True)
        self.num_joints = num_joints

    def set_bn_momentum(self, momentum):
        pass

    def forward(self, x):
        B, T, C = x.shape
        # Extract center frame (same as original BERT)
        x = x[:, T // 2:(T // 2 + 1)]  # [B, 1, C]
        x = x.permute(0, 2, 1).contiguous()  # [B, C, 1]
        x = self.shrink(x).view(B, -1, self.num_joints, 3)  # [B, 1, J, 3]
        return x


class TCN(nn.Module):
    """
    Replace BERT with ModernTCN-style temporal modeling.

    Input shape:  [B, C1, C2, T, N]  (from FuseView output)
    Output shape: [B, T, J, 3, N]     (same as original BERT)

    Matches exactly the interface of the original BERT class in bert.py.
    """

    def __init__(self, cfg, T=7, dropout=0.1, num_joints=17):
        super().__init__()
        self.cfg = cfg
        self.hidden = cfg.NETWORK.T_FORMER.NUM_CHANNELS  # e.g., 256
        self.n_layers = cfg.NETWORK.T_FORMER.NUM_LAYERS  # e.g., 3
        self.num_joints = num_joints
        self.T = T

        # Project input to hidden dim
        # Input after view: [B*N, T, C_in] → we project C_in → hidden
        self.input_proj = nn.Linear(cfg.NETWORK.NUM_CHANNELS, self.hidden)

        # ModernTCN blocks (no positional embedding needed due to large kernel)
        self.blocks = nn.Sequential(*[
            TCNBlock(
                channels=self.hidden,
                kernel_size=cfg.get('MODERNTCN_KERNEL_SIZE', 31),  # default 31
                ffn_ratio=2.0,
                dropout=dropout
            ) for _ in range(self.n_layers)
        ])

        # Same head as original BERT
        self.shrink = Head(cfg, in_channels=self.hidden, num_joints=num_joints)

    def set_bn_momentum(self, momentum):
        # Optional: propagate to BN layers if needed
        pass

    def forward(self, x):
        """
        x: [B, C1, C2, T, N] from FuseView
        Returns: [B, T_out, J, 3, N] where T_out = 1 (center frame)
        """
        if len(x.shape) == 5:
            B, C1, C2, T, N = x.shape
            x = x.view(B, -1, T, N)  # [B, C, T, N], C = C1*C2
            B, C, T, N = x.shape

        # Reshape to [B*N, T, C]
        x = x.permute(0, 3, 2, 1).contiguous()  # [B, N, T, C]
        x = x.view(B * N, T, C)  # [B*N, T, C]

        # Project to hidden dimension
        x = self.input_proj(x)  # [B*N, T, hidden]

        # Permute to [B*N, hidden, T] for Conv1d
        x = x.permute(0, 2, 1).contiguous()  # [B*N, hidden, T]

        # Apply ModernTCN blocks
        x = self.blocks(x)  # [B*N, hidden, T]

        # Back to [B*N, T, hidden]
        x = x.permute(0, 2, 1).contiguous()  # [B*N, T, hidden]

        # Apply head (extracts center frame)
        x = self.shrink(x)  # [B*N, 1, J, 3]

        # Reshape back to [B, T_out, J, 3, N]
        B_N, T_out, J, _ = x.shape
        x = x.view(B, N, T_out, J, 3)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, T_out, J, 3, N]

        return x
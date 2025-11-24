"""
Attention blocks and modules for DFA-ELM framework
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel Attention Module (squeeze-excitation style)"""
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # Global max pooling
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Shared MLP for dimensionality reduction and expansion
        self.fc = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class TemporalAttention(nn.Module):
    """Temporal Attention Module (self-attention on time dimension)"""
    def __init__(self, channels):
        super(TemporalAttention, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, L = x.size()

        # [B, C, L] -> [B, C, L]
        query = self.conv1(x)
        key = self.conv2(x)
        value = self.conv3(x)

        # Reshape for attention
        query = query.permute(0, 2, 1)  # [B, L, C]
        key = key.permute(0, 2, 1)  # [B, L, C]
        value = value.permute(0, 2, 1)  # [B, L, C]

        # Attention score
        attention = torch.bmm(query, key.transpose(1, 2))  # [B, L, L]
        attention = self.softmax(attention)

        # Apply attention weights
        out = torch.bmm(attention, value)  # [B, L, C]
        out = out.permute(0, 2, 1)  # [B, C, L]

        return out


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention Module"""
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: [B, C, L] -> [B, L, C]
        x = x.permute(0, 2, 1)

        B, L, C = x.shape

        # [B, L, C] -> [B, L, 3*C] -> [B, L, 3, num_heads, C//num_heads]
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, C // self.num_heads)

        # [B, L, 3, num_heads, C//num_heads] -> [3, B, num_heads, L, C//num_heads]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Separate Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, L, C//num_heads]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, L, L]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Get output values
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)  # [B, L, C]
        x = self.proj(x)
        x = self.proj_drop(x)

        # [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1)
        return x


class MultiscaleAttentionFusion(nn.Module):
    """
    Multiscale Attention Feature Fusion (MAF) Module for 1D signals

    Args:
        channels (int): Number of input channels (C)
        reduction_ratio (int): Channel reduction ratio for the bottleneck (r)
    """
    def __init__(self, channels, reduction_ratio=16):
        super(MultiscaleAttentionFusion, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        # Global branch components (left side)
        self.global_gap = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling

        # Point-wise convolutions for global branch
        self.global_conv1 = nn.Conv1d(channels, channels // reduction_ratio, kernel_size=1, bias=False)
        self.global_relu = nn.ReLU(inplace=True)
        self.global_conv2 = nn.Conv1d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        self.global_sigmoid = nn.Sigmoid()

        # Point-wise convolutions for local branch (right side)
        self.local_conv1 = nn.Conv1d(channels, channels // reduction_ratio, kernel_size=1, bias=False)
        self.local_relu = nn.ReLU(inplace=True)
        self.local_conv2 = nn.Conv1d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        self.local_sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        """
        Forward pass

        Args:
            x (torch.Tensor): First input tensor of shape [B, C, L]
            y (torch.Tensor): Second input tensor of shape [B, C, L]

        Returns:
            torch.Tensor: Fused output tensor
        """
        batch_size, C, L = x.size()

        # Global branch (left side) - Extract global feature information
        x_global = x  # Original path

        # Global attention pathway
        u = self.global_gap(x)  # [B, C, 1]
        u = self.global_conv1(u)  # [B, C/r, 1]
        u = self.global_relu(u)
        u = self.global_conv2(u)  # [B, C, 1]
        k = self.global_sigmoid(u)  # Attention weights K

        # Apply global attention
        x_reweighted = x_global * k  # Element-wise multiplication for re-weighting
        y_reweighted = y * k  # Element-wise multiplication for re-weighting

        # First global output Z1 (left branch)
        z1 = x_reweighted + y_reweighted

        # Local branch (right side) - Extract local detail information
        x_local = x  # Skip connection
        y_local = y  # Skip connection

        # Local attention pathway
        m = self.local_conv1(y)  # [B, C/r, L]
        m = self.local_relu(m)
        m = self.local_conv2(m)  # [B, C, L]
        k_star = self.local_sigmoid(m)  # Spatial attention map K*

        # Apply local attention
        x_local_reweighted = x_local * k_star  # Element-wise multiplication
        y_local_reweighted = y_local * k_star  # Element-wise multiplication

        # Second local output Z2 (right branch)
        z2 = x_local_reweighted + y_local_reweighted

        # Final fusion
        z = z1 + z2

        return z

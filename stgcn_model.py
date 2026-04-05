"""
stgcn_model.py — Step 5: Spatial-Temporal Graph Convolutional Network

ST-GCN architecture for skeleton-based ISL gesture recognition.

Input shape: (batch, C_in, T, V) where:
  C_in = 3 (x, y, z coordinates)
  T = 60 (frames)
  V = 67 (joints: 25 pose + 21 left hand + 21 right hand)

Graph structure: adjacency matrix based on human body joint connections.

Architecture:
  4 ST-GCN blocks: 3→64 → 64→128 → 128→128 → 128→256
  Each block: Spatial GCN + Temporal Conv + BatchNorm + ReLU + Dropout(0.3)
  Global average pooling after last block
  Fully connected output → 9 classes

Implemented in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def build_adjacency_matrix():
    """
    Build the adjacency matrix for 67 joints based on anatomical connections.

    Joint layout:
      Pose (0–24): upper body landmarks from MediaPipe
      Left Hand (25–45): 21 hand landmarks (offset by 25)
      Right Hand (46–66): 21 hand landmarks (offset by 46)

    Returns:
        np.ndarray of shape (67, 67), the adjacency matrix (with self-loops)
    """
    num_joints = 67
    edges = []

    # --- Pose connections (MediaPipe upper body, landmarks 0–24) ---
    # Face/head
    edges += [(0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6)]  # eyes
    edges += [(3, 7), (6, 8)]  # ear connections
    edges += [(9, 10)]  # mouth
    # Torso
    edges += [(11, 12)]  # shoulders
    edges += [(11, 13), (13, 15)]  # left arm: shoulder → elbow → wrist
    edges += [(12, 14), (14, 16)]  # right arm: shoulder → elbow → wrist
    edges += [(11, 23), (12, 24)]  # shoulder → hip
    edges += [(23, 24)]  # hips
    # Wrist to pinky/index connections
    edges += [(15, 17), (15, 19), (15, 21)]  # left wrist details
    edges += [(16, 18), (16, 20), (16, 22)]  # right wrist details
    edges += [(17, 19), (18, 20)]  # thumb-index connections
    # Nose to shoulders
    edges += [(0, 11), (0, 12)]

    # --- Left hand connections (joints 25–45) ---
    # MediaPipe hand landmark connections:
    # Wrist(0) → Thumb: 1→2→3→4
    # Wrist(0) → Index: 5→6→7→8
    # Wrist(0) → Middle: 9→10→11→12
    # Wrist(0) → Ring: 13→14→15→16
    # Wrist(0) → Pinky: 17→18→19→20
    lh_offset = 25
    lh_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # index
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
        (5, 9), (9, 13), (13, 17),             # palm connections
    ]
    for (a, b) in lh_connections:
        edges.append((a + lh_offset, b + lh_offset))

    # Connect left wrist (pose 15) to left hand wrist (joint 25)
    edges.append((15, 25))

    # --- Right hand connections (joints 46–66) ---
    rh_offset = 46
    for (a, b) in lh_connections:  # Same topology as left hand
        edges.append((a + rh_offset, b + rh_offset))

    # Connect right wrist (pose 16) to right hand wrist (joint 46)
    edges.append((16, 46))

    # Build adjacency matrix with self-loops
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for (i, j) in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Add self-loops
    np.fill_diagonal(A, 1.0)

    return A


def normalize_adjacency(A):
    """
    Symmetric normalization of adjacency matrix: D^{-1/2} A D^{-1/2}
    """
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-6)))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm


class SpatialGraphConv(nn.Module):
    """
    Spatial Graph Convolution layer.
    Performs graph convolution on each frame independently.
    """

    def __init__(self, in_channels, out_channels, A):
        super(SpatialGraphConv, self).__init__()
        self.A = A  # Normalized adjacency, registered as buffer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        x: (N, C, T, V)
        A: (V, V)
        """
        # Graph convolution: x * A
        # x shape: (N, C, T, V), A shape: (V, V)
        x = torch.einsum('nctv,vw->nctw', x, self.A)
        x = self.conv(x)
        x = self.bn(x)
        return x


class TemporalConv(nn.Module):
    """
    Temporal Convolution layer.
    Applies a 1D convolution along the temporal dimension.
    """

    def __init__(self, in_channels, out_channels, kernel_size=9):
        super(TemporalConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class STGCNBlock(nn.Module):
    """
    One ST-GCN block: Spatial GCN → Temporal Conv → ReLU → Dropout

    Includes a residual connection when input/output channels match.
    """

    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.3):
        super(STGCNBlock, self).__init__()
        self.spatial = SpatialGraphConv(in_channels, out_channels, A)
        self.temporal = TemporalConv(out_channels, out_channels, kernel_size=9)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.spatial(x)
        x = self.relu(x)
        x = self.temporal(x)
        x = self.relu(x + res)
        x = self.dropout(x)
        return x


class STGCN(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network for gesture recognition.

    Architecture:
      4 ST-GCN blocks: 3→64 → 64→128 → 128→128 → 128→256
      Global average pooling
      Fully connected → 9 classes
    """

    def __init__(self, num_classes=9, in_channels=3, num_joints=67, num_frames=60, dropout=0.3):
        super(STGCN, self).__init__()

        # Build and normalize adjacency matrix
        A_raw = build_adjacency_matrix()
        A_norm = normalize_adjacency(A_raw)
        self.register_buffer('A', torch.tensor(A_norm, dtype=torch.float32))

        # Data batch normalization
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        # ST-GCN blocks with increasing channels: 64 → 128 → 128 → 256
        self.block1 = STGCNBlock(in_channels, 64, self.A, dropout=dropout)
        self.block2 = STGCNBlock(64, 128, self.A, dropout=dropout)
        self.block3 = STGCNBlock(128, 128, self.A, dropout=dropout)
        self.block4 = STGCNBlock(128, 256, self.A, dropout=dropout)

        # Classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        x: (N, C, T, V) = (batch, 3, 60, 67)
        """
        N, C, T, V = x.size()

        # Data batch normalization
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()  # (N, C, T, V)

        # ST-GCN blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Global average pooling over T and V
        x = x.mean(dim=[2, 3])  # (N, 256)

        # Classification
        x = self.fc(x)

        return x

    def _update_adjacency(self):
        """Update adjacency matrix reference in all blocks after .to(device)"""
        self.block1.spatial.A = self.A
        self.block2.spatial.A = self.A
        self.block3.spatial.A = self.A
        self.block4.spatial.A = self.A


def reshape_input(keypoints):
    """
    Reshape keypoint data from (batch, T, 201) to (batch, C, T, V).

    The 201 features per frame represent 67 joints × 3 coordinates.
    We reshape to: channels=3 (x, y, z), joints=67.

    Args:
        keypoints: np.ndarray or torch.Tensor of shape (batch, 60, 201)

    Returns:
        torch.Tensor of shape (batch, 3, 60, 67)
    """
    if isinstance(keypoints, np.ndarray):
        keypoints = torch.tensor(keypoints, dtype=torch.float32)

    N, T, F = keypoints.shape  # (batch, 60, 201)
    V = 67  # number of joints
    C = 3   # x, y, z

    # Reshape: (N, T, 67*3) → (N, T, 67, 3) → (N, 3, T, 67)
    x = keypoints.view(N, T, V, C)
    x = x.permute(0, 3, 1, 2).contiguous()  # (N, C, T, V)

    return x


def create_model(num_classes=9, device='cpu'):
    """
    Create, initialize and return the ST-GCN model.
    """
    model = STGCN(num_classes=num_classes)
    model = model.to(device)
    model._update_adjacency()
    return model


if __name__ == "__main__":
    # Quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = create_model(num_classes=9, device=device)
    print(f"\nModel created successfully on {device}")

    # Test with random input
    dummy_input = torch.randn(4, 3, 60, 67).to(device)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (softmax): {F.softmax(output[0], dim=0)}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

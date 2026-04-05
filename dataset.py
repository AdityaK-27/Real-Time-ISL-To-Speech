import torch
import numpy as np
from torch.utils.data import Dataset
import random

class ISLDataset(Dataset):
    """
    PyTorch Dataset for ISL ST-GCN.
    Applies dynamic augmentations exactly during training to prevent data leakage.
    Input shape is (60, 201). Handled and transformed to (3, 60, 67) for ST-GCN.
    """
    def __init__(self, sequences, labels, augment=False, reshape_func=None):
        """
        Args:
            sequences: float32 array of shape (N, 60, 201)
            labels: int64 array of shape (N,)
            augment: boolean, whether to apply on-the-fly augmentations
            reshape_func: function to reshape (60, 201) to (3, 60, 67)
        """
        self.sequences = sequences
        self.labels = labels
        self.augment = augment
        self.reshape_func = reshape_func

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        label = self.labels[idx]

        if self.augment:
            seq = self.apply_augmentations(seq)

        # Standardize for ST-GCN: reshape from (60, 201) to (3, 60, 67)
        if self.reshape_func:
            seq = self.reshape_func(np.expand_dims(seq, axis=0))[0]

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def apply_augmentations(self, sequence):
        """Randomly apply a combination of augmentations to the sequence."""
        # 1. Random Gaussian Noise
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.005, sequence.shape)
            sequence += noise

        # 2. Random Horizontal Flip (mirror hand)
        if random.random() < 0.3:
            # X coordinates are at indices 0, 3, 6...
            for i in range(0, sequence.shape[1], 3):
                sequence[:, i] = -sequence[:, i]

            # In MediaPipe, we must also swap left and right hands if flipping:
            # LH is 75-137, RH is 138-200
            lh = sequence[:, 75:138].copy()
            rh = sequence[:, 138:201].copy()
            sequence[:, 75:138] = rh
            sequence[:, 138:201] = lh

        # 3. Random Rotation (±15 degrees in XY plane)
        if random.random() < 0.5:
            angle = np.random.uniform(-15, 15)
            theta = np.radians(angle)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            for i in range(0, sequence.shape[1], 3):
                x = sequence[:, i].copy()
                y = sequence[:, i + 1].copy()
                sequence[:, i] = cos_t * x - sin_t * y
                sequence[:, i + 1] = sin_t * x + cos_t * y

        # 4. Random Scaling (±10%)
        if random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            sequence = sequence * scale

        # 5. Random Keypoint Dropout (simulate occlusion, drop 1-3 joints out of 67)
        if random.random() < 0.4:
            num_drop = random.randint(1, 3)
            # Choose random joints to drop (0 to 66)
            drop_joints = random.sample(range(67), num_drop)
            for j in drop_joints:
                # Joint j maps to coordinates [j*3, j*3+1, j*3+2]
                idx = j * 3
                sequence[:, idx:idx+3] = 0

        # Keep zero-padded empty frames strictly completely zeroed out
        zero_mask = np.all(self.sequences[0] == 0, axis=1) # check against original length
        # Better checking: find which rows are exactly 0 in `seq`
        zero_mask_seq = np.all(sequence == 0, axis=1)
        sequence[zero_mask_seq] = 0

        return sequence

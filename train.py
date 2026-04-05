"""
train.py — Step 5: Training Loop for ST-GCN

Trains the ST-GCN model on keypoint data for ISL gesture recognition.

Features:
  - Accepts experiment name as argument (baseline, aug_only, kalman_only, full_pipeline)
  - Loads correct data based on experiment condition
  - 70/15/15 train/val/test split, stratified, seed=42
  - Adam optimizer, CrossEntropyLoss, lr=1e-3
  - ReduceLROnPlateau, EarlyStopping (patience=15)
  - Saves best model weights and training history
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from stgcn_model import STGCN, reshape_input, create_model
from dataset import ISLDataset

# ============================================================
# Configuration
# ============================================================
SEED = 42
NUM_CLASSES = 9
BATCH_SIZE = 16
EPOCHS = 150
LEARNING_RATE = 1e-3
PATIENCE = 15  # Early stopping patience


def set_seed(seed=SEED):
    """Set random seeds everywhere for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Data Loading
# ============================================================
def load_data(data_dir):
    """
    Load all .npy keypoint files from a directory organized by gesture.

    Args:
        data_dir: path like 'data/keypoints_raw' or 'data/keypoints_augmented/...'

    Returns:
        X: np.ndarray of shape (N, 60, 201)
        y: np.ndarray of shape (N,) with integer labels
        class_names: list of gesture names (sorted)
    """
    sequences = []
    labels = []
    filenames = []
    class_names = sorted([d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d))])
    label_map = {name: idx for idx, name in enumerate(class_names)}

    print(f"Loading data from: {data_dir}")
    print(f"Classes: {class_names}")

    for gesture in class_names:
        gesture_dir = os.path.join(data_dir, gesture)
        npy_files = [f for f in os.listdir(gesture_dir) if f.endswith('.npy')]

        for npy_file in npy_files:
            data = np.load(os.path.join(gesture_dir, npy_file))
            sequences.append(data)
            labels.append(label_map[gesture])
            filenames.append(f"{gesture}/{npy_file}")

        print(f"  {gesture}: {len(npy_files)} samples")

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    filenames = np.array(filenames)

    print(f"Total: {len(X)} samples, {len(class_names)} classes")
    return X, y, filenames, class_names


def get_experiment_config(experiment_name):
    """
    Return the correct data directory, augmentation flag, and whether data is pre-split.
    """
    configs = {
        "baseline":        {"dir": os.path.join("data", "keypoints_raw"), "augment": False, "presplit": False},
        "aug_only":        {"dir": os.path.join("data", "keypoints_raw"), "augment": True, "presplit": False},
        "kalman_only":     {"dir": os.path.join("data", "keypoints_processed"), "augment": False, "presplit": False},
        "full_pipeline":   {"dir": os.path.join("data", "keypoints_processed"), "augment": True, "presplit": False},
        "offline_aug":     {"dir": os.path.join("data", "keypoints_offline_aug"), "augment": False, "presplit": True},
        "kalman_off_aug":  {"dir": os.path.join("data", "keypoints_offline_aug_kalman"), "augment": False, "presplit": True},
        "large_raw":       {"dir": os.path.join("data", "keypoints_large_raw"), "augment": False, "presplit": False},
        "large_kalman":    {"dir": os.path.join("data", "keypoints_large_processed"), "augment": False, "presplit": False},
    }
    if experiment_name not in configs:
        raise ValueError(f"Unknown experiment: {experiment_name}. "
                         f"Choose from: {list(configs.keys())}")
    return configs[experiment_name]


def load_presplit_data(base_dir, split):
    """
    Load data from a pre-split directory: base_dir/train/, base_dir/val/, base_dir/test/.
    Returns X, y, filenames, class_names for the given split.
    """
    split_dir = os.path.join(base_dir, split)
    sequences = []
    labels = []
    filenames = []
    class_names = sorted([d for d in os.listdir(split_dir)
                          if os.path.isdir(os.path.join(split_dir, d))])
    label_map = {name: idx for idx, name in enumerate(class_names)}

    for gesture in class_names:
        gesture_dir = os.path.join(split_dir, gesture)
        npy_files = sorted([f for f in os.listdir(gesture_dir) if f.endswith('.npy')])
        for npy_file in npy_files:
            data = np.load(os.path.join(gesture_dir, npy_file))
            sequences.append(data)
            labels.append(label_map[gesture])
            filenames.append(f"{gesture}/{npy_file}")

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    filenames = np.array(filenames)
    return X, y, filenames, class_names


# ============================================================
# Training
# ============================================================
def train_model(experiment_name):
    """
    Full training pipeline for one experiment.
    """
    set_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Load data
    config = get_experiment_config(experiment_name)

    if config["presplit"]:
        # --- Pre-split offline augmented data ---
        print(f"Loading pre-split data from: {config['dir']}")
        X_train, y_train, fn_train, class_names = load_presplit_data(config["dir"], "train")
        X_val, y_val, fn_val, _ = load_presplit_data(config["dir"], "val")
        X_test, y_test, fn_test, _ = load_presplit_data(config["dir"], "test")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")
    else:
        # --- Regular experiments: load + split ---
        X, y, filenames, class_names = load_data(config["dir"])
        X_train, X_temp, y_train, y_temp, fn_train, fn_temp = train_test_split(
            X, y, filenames, test_size=0.30, stratify=y, random_state=SEED
        )
        X_val, X_test, y_val, y_test, fn_val, fn_test = train_test_split(
            X_temp, y_temp, fn_temp, test_size=0.50, stratify=y_temp, random_state=SEED
        )

    print(f"\nSplit sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # ---------------------------------------------------------
    # SANITY CHECKS (ZERO OVERLAP ASSERTION)
    # ---------------------------------------------------------
    train_set = set(fn_train)
    val_set = set(fn_val)
    test_set = set(fn_test)

    assert len(train_set.intersection(val_set)) == 0, "DATA LEAKAGE: Train and Val sets overlap!"
    assert len(train_set.intersection(test_set)) == 0, "DATA LEAKAGE: Train and Test sets overlap!"
    assert len(val_set.intersection(test_set)) == 0, "DATA LEAKAGE: Val and Test sets overlap!"
    print("✓ Sanity Check Passed: Train, Val, and Test subsets have exactly ZERO file overlap.")
    # ---------------------------------------------------------

    # DataLoaders using ISLDataset
    # For presplit (offline_aug) experiments: augment=False (data already augmented on disk)
    # For dynamic experiments: augment depends on config
    def stgcn_reshape(seq_numpy):
        reshaped = reshape_input(seq_numpy)
        return reshaped

    train_dataset = ISLDataset(X_train, y_train, augment=config["augment"], reshape_func=stgcn_reshape)
    val_dataset = ISLDataset(X_val, y_val, augment=False, reshape_func=stgcn_reshape)
    test_dataset = ISLDataset(X_test, y_test, augment=False, reshape_func=stgcn_reshape)

    def custom_collate(batch):
        # batches come out as list of tuples (tensor, label)
        X_batch = torch.stack([item[0] for item in batch]).to(device)
        y_batch = torch.stack([item[1] for item in batch]).to(device)
        return X_batch, y_batch

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

    # Create model
    model = create_model(num_classes=NUM_CLASSES, device=device)
    criterion = nn.CrossEntropyLoss()
    
    # Add L2 Kernel Regularization equivalent (weight_decay=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=8, min_lr=1e-6
    )

    # Output directories
    model_save_dir = os.path.join("models", experiment_name)
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "best_stgcn.pth")

    # Training history
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "lr": []
    }

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        # --- Training phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Print progress
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'class_names': class_names,
            }, best_model_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⚠ Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # --- Final test evaluation ---
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
    
    test_correct = 0
    test_total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_X.size(0)
            
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == batch_y).sum().item()
            test_total += batch_y.size(0)

    test_acc = test_correct / test_total
    test_loss /= test_total

    print(f"\n{'='*60}")
    print(f"Final Test Results — {experiment_name}")
    print(f"{'='*60}")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # Save history
    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    history["class_names"] = class_names
    history["experiment"] = experiment_name
    history["best_epoch"] = checkpoint['epoch']

    history_path = os.path.join(model_save_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Save test data for evaluation
    np.save(os.path.join(model_save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(model_save_dir, "y_test.npy"), y_test)

    return history


if __name__ == "__main__":
    experiment = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    train_model(experiment)

"""
offline_augment.py — Offline Augmentation with Split-First Strategy

Splits raw keypoints into train/val/test FIRST (70/15/15, stratified, seed=42),
then applies 4 augmentations ONLY to the training set.
Val and test sets remain untouched originals.

Output structure:
  data/keypoints_offline_aug/train/<gesture>/   ← originals + augmented
  data/keypoints_offline_aug/val/<gesture>/     ← originals only
  data/keypoints_offline_aug/test/<gesture>/    ← originals only

  data/keypoints_offline_aug_kalman/train/<gesture>/  ← Kalman-filtered
  data/keypoints_offline_aug_kalman/val/<gesture>/    ← Kalman-filtered
  data/keypoints_offline_aug_kalman/test/<gesture>/   ← Kalman-filtered

Augmentations (4 techniques, 1 copy each):
  1. Time warping
  2. Mirror flip
  3. Frame dropout
  4. Speed variation

Per-class counts (approx):
  Train: ~15 originals + ~60 augmented = ~75 per class
  Val:   ~3 per class (untouched)
  Test:  ~3 per class (untouched)
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from augment_keypoints import AUGMENTATIONS
from kalman_filter import apply_kalman

SEED = 42
np.random.seed(SEED)


def load_all_data(data_dir):
    """Load all .npy files, return arrays + filenames."""
    sequences = []
    labels = []
    filenames = []
    class_names = sorted([d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d))])
    label_map = {name: idx for idx, name in enumerate(class_names)}

    for gesture in class_names:
        gesture_dir = os.path.join(data_dir, gesture)
        npy_files = sorted([f for f in os.listdir(gesture_dir) if f.endswith('.npy')])
        for npy_file in npy_files:
            data = np.load(os.path.join(gesture_dir, npy_file))
            sequences.append(data)
            labels.append(label_map[gesture])
            filenames.append((gesture, npy_file))

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y, filenames, class_names


def save_split(X, filenames, output_dir, split_name, augment_train=False):
    """Save samples to output_dir/split_name/<gesture>/. Optionally augment."""
    saved_count = {}

    for i, (gesture, npy_file) in enumerate(filenames):
        gesture_dir = os.path.join(output_dir, split_name, gesture)
        os.makedirs(gesture_dir, exist_ok=True)

        data = X[i]

        # Save original
        np.save(os.path.join(gesture_dir, npy_file), data)
        saved_count[gesture] = saved_count.get(gesture, 0) + 1

        # Apply augmentations only for train
        if augment_train and split_name == "train":
            base_name = npy_file.replace('.npy', '')
            for aug_name, aug_func in AUGMENTATIONS:
                augmented = aug_func(data)
                aug_filename = f"{base_name}_aug_{aug_name}.npy"
                np.save(os.path.join(gesture_dir, aug_filename), augmented)
                saved_count[gesture] = saved_count.get(gesture, 0) + 1

    return saved_count


def apply_kalman_to_split(input_base, output_base):
    """Apply Kalman filtering to all files in a pre-split directory."""
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(input_base, split)
        if not os.path.exists(split_dir):
            continue

        for gesture in sorted(os.listdir(split_dir)):
            gesture_in = os.path.join(split_dir, gesture)
            if not os.path.isdir(gesture_in):
                continue

            gesture_out = os.path.join(output_base, split, gesture)
            os.makedirs(gesture_out, exist_ok=True)

            npy_files = sorted([f for f in os.listdir(gesture_in) if f.endswith('.npy')])
            for npy_file in npy_files:
                data = np.load(os.path.join(gesture_in, npy_file))
                filtered = apply_kalman(data)
                np.save(os.path.join(gesture_out, npy_file), filtered)

        print(f"  Kalman filtered: {split}/ ({sum(len(os.listdir(os.path.join(split_dir, g))) for g in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, g)))} files)")


if __name__ == "__main__":
    RAW_DIR = os.path.join("data", "keypoints_raw")
    OFFLINE_AUG_DIR = os.path.join("data", "keypoints_offline_aug")
    OFFLINE_AUG_KALMAN_DIR = os.path.join("data", "keypoints_offline_aug_kalman")

    if not os.path.exists(RAW_DIR):
        print(f"ERROR: '{RAW_DIR}' not found. Run extract_keypoints.py first.")
        exit(1)

    # ============================================================
    # Step 1: Load all raw data
    # ============================================================
    print("=" * 60)
    print("STEP 1: Loading raw keypoints")
    print("=" * 60)
    X, y, filenames, class_names = load_all_data(RAW_DIR)
    print(f"Loaded {len(X)} samples across {len(class_names)} classes")
    for cls in class_names:
        count = sum(1 for g, _ in filenames if g == cls)
        print(f"  {cls}: {count}")

    # ============================================================
    # Step 2: Split into train/val/test (BEFORE augmentation)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: Splitting (70/15/15, stratified, seed=42)")
    print("=" * 60)

    indices = np.arange(len(X))
    idx_train, idx_temp = train_test_split(
        indices, test_size=0.30, stratify=y, random_state=SEED
    )
    y_temp = y[idx_temp]
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    X_train = X[idx_train]
    X_val = X[idx_val]
    X_test = X[idx_test]
    fn_train = [filenames[i] for i in idx_train]
    fn_val = [filenames[i] for i in idx_val]
    fn_test = [filenames[i] for i in idx_test]

    # Sanity check: zero overlap
    train_names = set(fn_train)
    val_names = set(fn_val)
    test_names = set(fn_test)
    assert len(train_names & val_names) == 0, "LEAKAGE: train/val overlap!"
    assert len(train_names & test_names) == 0, "LEAKAGE: train/test overlap!"
    assert len(val_names & test_names) == 0, "LEAKAGE: val/test overlap!"

    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print("  ✓ Zero overlap verified")

    # ============================================================
    # Step 3: Save with offline augmentation (train only)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: Saving offline augmented data")
    print("=" * 60)

    train_counts = save_split(X_train, fn_train, OFFLINE_AUG_DIR, "train", augment_train=True)
    val_counts = save_split(X_val, fn_val, OFFLINE_AUG_DIR, "val", augment_train=False)
    test_counts = save_split(X_test, fn_test, OFFLINE_AUG_DIR, "test", augment_train=False)

    print(f"\nTrain set (with augmentation):")
    for cls, cnt in sorted(train_counts.items()):
        print(f"  {cls}: {cnt} samples")
    total_train = sum(train_counts.values())

    print(f"\nVal set (no augmentation):")
    for cls, cnt in sorted(val_counts.items()):
        print(f"  {cls}: {cnt} samples")
    total_val = sum(val_counts.values())

    print(f"\nTest set (no augmentation):")
    for cls, cnt in sorted(test_counts.items()):
        print(f"  {cls}: {cnt} samples")
    total_test = sum(test_counts.values())

    print(f"\n  Total: train={total_train}, val={total_val}, test={total_test}")
    print(f"  Saved to: {OFFLINE_AUG_DIR}")

    # ============================================================
    # Step 4: Apply Kalman filtering to offline augmented data
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4: Applying Kalman filtering to offline augmented data")
    print("=" * 60)
    apply_kalman_to_split(OFFLINE_AUG_DIR, OFFLINE_AUG_KALMAN_DIR)
    print(f"  Saved to: {OFFLINE_AUG_KALMAN_DIR}")

    print("\n" + "=" * 60)
    print("✅ DONE — Offline augmentation complete")
    print("=" * 60)

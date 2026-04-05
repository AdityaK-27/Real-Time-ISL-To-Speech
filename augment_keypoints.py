"""
augment_keypoints.py — Step 4: Skeleton-Based Augmentation

This makes the original dataset more manageable.
Each technique produces 1 new sample:
  21 originals × 4 augmentations = 84 augmented + 21 original = 105 per class

Techniques:
  1. Time warping (stretch/compress then resample to 60 frames)
  2. Mirror flip (negate all X coordinates)
  3. Random frame dropout (zero out 10–15% of frames)
  4. Speed variation (duplicate/remove every nth frame, resample to 60)

Augmentation is applied separately for with-Kalman and without-Kalman conditions.
Saves to data/keypoints_augmented/without_kalman/ and with_kalman/.
"""

import os
import numpy as np
from scipy.interpolate import interp1d

# Fixed seed for reproducibility
np.random.seed(42)

TARGET_FRAMES = 60
FEATURES = 201
NUM_JOINTS = 67  # 25 pose + 21 LH + 21 RH


def augment_time_warp(sequence, sigma=0.2):
    """
    Randomly stretch or compress the temporal dimension,
    then re-interpolate back to 60 frames.
    """
    T, F = sequence.shape

    # Find actual (non-padded) frames
    non_zero_mask = ~np.all(sequence == 0, axis=1)
    actual_len = np.sum(non_zero_mask)

    if actual_len < 4:
        return sequence.copy()

    actual_seq = sequence[non_zero_mask]

    # Create random warping path
    orig_steps = np.arange(actual_len)
    warp_steps = np.sort(
        orig_steps + np.random.normal(0, sigma, actual_len).cumsum()
    )
    # Normalize warp_steps to [0, actual_len-1]
    warp_steps = (warp_steps - warp_steps[0]) / (warp_steps[-1] - warp_steps[0]) * (actual_len - 1)

    # Interpolate each feature dimension
    warped = np.zeros((TARGET_FRAMES, F))
    new_steps = np.linspace(0, actual_len - 1, TARGET_FRAMES)

    for dim in range(F):
        interp_func = interp1d(warp_steps, actual_seq[:, dim],
                               kind='linear', fill_value='extrapolate')
        warped[:, dim] = interp_func(new_steps)

    return warped


def augment_mirror_flip(sequence):
    """
    Negate all X coordinates to simulate a left-handed signer.
    X coordinates are at indices 0, 3, 6, 9, ... (every 3rd starting from 0).
    Also swaps left hand and right hand landmark sections.
    """
    flipped = sequence.copy()
    T, F = flipped.shape

    # Negate X coordinates (every 3rd value starting from index 0)
    for i in range(0, F, 3):
        flipped[:, i] = -flipped[:, i]

    # Swap left hand (indices 75–137) and right hand (indices 138–200)
    lh = flipped[:, 75:138].copy()
    rh = flipped[:, 138:201].copy()
    flipped[:, 75:138] = rh
    flipped[:, 138:201] = lh

    return flipped


def augment_frame_dropout(sequence, dropout_ratio_range=(0.10, 0.15)):
    """
    Randomly zero out 10–15% of frames to simulate occlusion.
    """
    augmented = sequence.copy()
    T = augmented.shape[0]

    # Find non-zero frames
    non_zero = np.where(~np.all(augmented == 0, axis=1))[0]
    if len(non_zero) < 5:
        return augmented

    ratio = np.random.uniform(*dropout_ratio_range)
    num_drop = max(1, int(len(non_zero) * ratio))
    drop_indices = np.random.choice(non_zero, size=num_drop, replace=False)
    augmented[drop_indices] = 0

    return augmented


def augment_speed_variation(sequence, speed_factor_range=(0.8, 1.2)):
    """
    Simulate faster/slower signing by changing the effective speed,
    then resample back to 60 frames.
    """
    T, F = sequence.shape

    non_zero_mask = ~np.all(sequence == 0, axis=1)
    actual_len = np.sum(non_zero_mask)
    if actual_len < 4:
        return sequence.copy()

    actual_seq = sequence[non_zero_mask]
    speed_factor = np.random.uniform(*speed_factor_range)
    new_len = max(4, int(actual_len * speed_factor))

    # Resample actual sequence to new_len
    orig_indices = np.linspace(0, actual_len - 1, actual_len)
    new_indices = np.linspace(0, actual_len - 1, new_len)

    resampled = np.zeros((new_len, F))
    for dim in range(F):
        interp_func = interp1d(orig_indices, actual_seq[:, dim], kind='linear')
        resampled[:, dim] = interp_func(new_indices)

    # Pad/truncate to TARGET_FRAMES
    result = np.zeros((TARGET_FRAMES, F))
    copy_len = min(new_len, TARGET_FRAMES)
    result[:copy_len] = resampled[:copy_len]

    return result


# All 4 augmentation functions with their names
AUGMENTATIONS = [
    ("time_warp", augment_time_warp),
    ("mirror_flip", augment_mirror_flip),
    ("frame_dropout", augment_frame_dropout),
    ("speed_variation", augment_speed_variation),
]


def augment_dataset(input_dir, output_dir):
    """
    Augment all .npy files in input_dir and save originals + augmented
    to output_dir. Each original produces 4 augmented copies.

    Final count per class: 21 originals + 84 augmented = 105
    """
    os.makedirs(output_dir, exist_ok=True)

    for gesture in sorted(os.listdir(input_dir)):
        gesture_in = os.path.join(input_dir, gesture)
        if not os.path.isdir(gesture_in):
            continue

        gesture_out = os.path.join(output_dir, gesture)
        os.makedirs(gesture_out, exist_ok=True)

        npy_files = [f for f in os.listdir(gesture_in) if f.endswith('.npy')]
        print(f"  Augmenting: {gesture} ({len(npy_files)} originals)")

        for npy_file in sorted(npy_files):
            data = np.load(os.path.join(gesture_in, npy_file))
            base_name = npy_file.replace('.npy', '')

            # Save original
            np.save(os.path.join(gesture_out, npy_file), data)

            # Apply each augmentation
            for aug_name, aug_func in AUGMENTATIONS:
                # Reset seed per augmentation for reproducibility
                augmented = aug_func(data)
                aug_filename = f"{base_name}_aug_{aug_name}.npy"
                np.save(os.path.join(gesture_out, aug_filename), augmented)

        # Count output
        total = len(os.listdir(gesture_out))
        print(f"    → {total} total samples")


if __name__ == "__main__":
    # Augment without-Kalman data (from keypoints_raw)
    RAW_DIR = os.path.join("data", "keypoints_raw")
    AUG_NO_KALMAN = os.path.join("data", "keypoints_augmented", "without_kalman")

    # Augment with-Kalman data (from keypoints_processed)
    PROCESSED_DIR = os.path.join("data", "keypoints_processed")
    AUG_KALMAN = os.path.join("data", "keypoints_augmented", "with_kalman")

    print("=" * 60)
    print("Augmenting WITHOUT Kalman (from keypoints_raw)")
    print("=" * 60)
    if os.path.exists(RAW_DIR):
        augment_dataset(RAW_DIR, AUG_NO_KALMAN)
    else:
        print(f"  ERROR: {RAW_DIR} not found. Run extract_keypoints.py first.")

    print()
    print("=" * 60)
    print("Augmenting WITH Kalman (from keypoints_processed)")
    print("=" * 60)
    if os.path.exists(PROCESSED_DIR):
        augment_dataset(PROCESSED_DIR, AUG_KALMAN)
    else:
        print(f"  ERROR: {PROCESSED_DIR} not found. Run kalman_filter.py first.")

"""
normalize_pose.py — Step 2: Pose Normalization

Centers the skeleton by subtracting the midpoint between left shoulder (landmark 11)
and right shoulder (landmark 12) from all coordinates.
Scales the skeleton by dividing all coordinates by the torso length (distance from
shoulder midpoint to hip midpoint — landmarks 23 & 24).

This makes the skeleton position-invariant and size-invariant, removing variation
caused by camera distance and subject position.

Keypoint layout per frame (201 values = 67 joints × 3 coords):
  - Pose landmarks 0–24 (upper body only): indices 0–74  (25 joints × 3)
  - Left hand landmarks 0–20: indices 75–137  (21 joints × 3)
  - Right hand landmarks 0–20: indices 138–200 (21 joints × 3)

Within the pose section:
  - Landmark 11 (left shoulder)  → pose index 11 → coords at [33, 34, 35]
  - Landmark 12 (right shoulder) → pose index 12 → coords at [36, 37, 38]
  - Landmark 23 (left hip)       → pose index 23 → coords at [69, 70, 71]
  - Landmark 24 (right hip)      → pose index 24 → coords at [72, 73, 74]
"""

import numpy as np


def normalize_sequence(sequence):
    """
    Normalize a full keypoint sequence of shape (T, 201).

    For each frame:
      1. Compute shoulder midpoint from landmarks 11 and 12
      2. Compute hip midpoint from landmarks 23 and 24
      3. Compute torso length = distance(shoulder_mid, hip_mid)
      4. Subtract shoulder midpoint from all (x, y, z) triplets
      5. Divide all coordinates by torso length

    Args:
        sequence: np.ndarray of shape (T, 201)

    Returns:
        np.ndarray of shape (T, 201), normalized
    """
    normalized = sequence.copy().astype(np.float64)
    num_frames, num_features = normalized.shape
    num_joints = num_features // 3  # 67 joints

    for t in range(num_frames):
        frame = normalized[t]

        # Skip all-zero frames (padding)
        if np.all(frame == 0):
            continue

        # Reshape to (67, 3) for easier manipulation
        joints = frame.reshape(num_joints, 3)

        # Pose landmarks are indices 0–24 in the joint array
        # Landmark 11 (left shoulder) = joint index 11
        # Landmark 12 (right shoulder) = joint index 12
        left_shoulder = joints[11]
        right_shoulder = joints[12]
        shoulder_mid = (left_shoulder + right_shoulder) / 2.0

        # Landmark 23 (left hip) = joint index 23
        # Landmark 24 (right hip) = joint index 24
        left_hip = joints[23]
        right_hip = joints[24]
        hip_mid = (left_hip + right_hip) / 2.0

        # Torso length = distance from shoulder midpoint to hip midpoint
        torso_length = np.linalg.norm(shoulder_mid - hip_mid)

        # Avoid division by zero (if torso length is too small, skip normalization)
        if torso_length < 1e-6:
            continue

        # Step 1: Center — subtract shoulder midpoint from all joints
        joints -= shoulder_mid

        # Step 2: Scale — divide all coordinates by torso length
        joints /= torso_length

        normalized[t] = joints.flatten()

    return normalized


if __name__ == "__main__":
    import os

    # Test on a sample file
    sample_path = os.path.join("data", "keypoints_raw")
    if os.path.exists(sample_path):
        for gesture in os.listdir(sample_path):
            gesture_dir = os.path.join(sample_path, gesture)
            if os.path.isdir(gesture_dir):
                for f in os.listdir(gesture_dir)[:1]:
                    data = np.load(os.path.join(gesture_dir, f))
                    print(f"Before normalization — mean: {data.mean():.6f}, std: {data.std():.6f}")
                    normed = normalize_sequence(data)
                    print(f"After normalization  — mean: {normed.mean():.6f}, std: {normed.std():.6f}")
                    print(f"Shape: {normed.shape}")
                    break
                break
    else:
        print("No keypoints_raw data found. Run extract_keypoints.py first.")

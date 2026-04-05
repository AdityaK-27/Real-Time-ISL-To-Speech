"""
extract_keypoints.py — Step 1: MediaPipe Holistic Keypoint Extraction

Extracts per frame:
  - Left hand (21 landmarks × 3 coords = 63 values)
  - Right hand (21 landmarks × 3 coords = 63 values)
  - Pose landmarks 0–24 only (upper body, drop legs 25–32) = 25 × 3 = 75 values
  Total per frame: 201 values

All sequences are padded/truncated to 60 frames.
Missing landmarks (e.g., hand not detected) are filled with zeros.
Logs any video where MediaPipe fails to detect hands in >30% of frames.

After extraction, applies pose normalization (Step 2) before saving.
Saves to data/keypoints_raw/<gesture>/<video>.npy with shape (60, 201).
"""

import os
import argparse
import cv2
import numpy as np
import mediapipe as mp
from normalize_pose import normalize_sequence

mp_holistic = mp.solutions.holistic

# --- Configuration ---
MAX_FRAMES = 60
NUM_POSE_LANDMARKS = 25   # landmarks 0–24 (upper body only)
NUM_HAND_LANDMARKS = 21
FEATURES_PER_FRAME = (NUM_POSE_LANDMARKS + NUM_HAND_LANDMARKS * 2) * 3  # 201

# Mapping from greetings_data folder names to clean gesture names
FOLDER_NAME_MAP = {
    "48. Hello": "hello",
    "49. How are you": "how_are_you",
    "50. Alright": "alright",
    "51. Good Morning": "good_morning",
    "52. Good afternoon": "good_afternoon",
    "53. Good evening": "good_evening",
    "54. Good night": "good_night",
    "55. Thank you": "thank_you",
    "56. Pleased": "pleased",
}


def extract_landmarks(results):
    """
    Extract keypoints from a single MediaPipe Holistic result.

    Returns a 1D array of 201 values:
      [pose_0_x, pose_0_y, pose_0_z, ..., pose_24_x, y, z,
       lh_0_x, y, z, ..., lh_20_x, y, z,
       rh_0_x, y, z, ..., rh_20_x, y, z]
    """
    # Pose: only landmarks 0–24 (upper body), 3D (x, y, z)
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark[:NUM_POSE_LANDMARKS]]
        ).flatten()
    else:
        pose = np.zeros(NUM_POSE_LANDMARKS * 3)

    # Left hand: all 21 landmarks, 3D
    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        ).flatten()
    else:
        lh = np.zeros(NUM_HAND_LANDMARKS * 3)

    # Right hand: all 21 landmarks, 3D
    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        ).flatten()
    else:
        rh = np.zeros(NUM_HAND_LANDMARKS * 3)

    return np.concatenate([pose, lh, rh])


def pad_or_truncate(sequence, target_length=MAX_FRAMES):
    """
    Pad (with zeros) or truncate a sequence to exactly target_length frames.
    """
    current_length = len(sequence)
    if current_length == target_length:
        return np.array(sequence)
    elif current_length > target_length:
        return np.array(sequence[:target_length])
    else:
        padding = [np.zeros(FEATURES_PER_FRAME) for _ in range(target_length - current_length)]
        return np.array(sequence + padding)


def extract_video_keypoints(video_path):
    """
    Extract keypoints from a single video file.

    Returns:
        tuple: (keypoints array of shape (60, 201), hand_missing_ratio float)
    """
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    total_frames = 0
    hand_missing_count = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            # Convert BGR to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            # Track hand detection failures
            if not results.left_hand_landmarks and not results.right_hand_landmarks:
                hand_missing_count += 1

            keypoints_list.append(extract_landmarks(results))

    cap.release()

    # Calculate hand missing ratio
    hand_missing_ratio = hand_missing_count / max(total_frames, 1)

    # Pad or truncate to 60 frames
    keypoints_array = pad_or_truncate(keypoints_list, MAX_FRAMES)

    return keypoints_array, hand_missing_ratio


def process_all_videos(import_path, export_path):
    """
    Process all videos from greetings_data/ and save to data/keypoints_raw/.

    Applies pose normalization after extraction (Step 2).
    """
    problematic_videos = []

    for folder_name in sorted(os.listdir(import_path)):
        folder_path = os.path.join(import_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Map numbered folder names to clean gesture names
        gesture_name = FOLDER_NAME_MAP.get(folder_name, folder_name.lower().replace(" ", "_"))
        export_gesture_dir = os.path.join(export_path, gesture_name)
        os.makedirs(export_gesture_dir, exist_ok=True)

        video_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.mp4', '.mov', '.avi'))]

        print(f"\n--- Processing gesture: {gesture_name} ({len(video_files)} videos) ---")

        for video_file in sorted(video_files):
            video_path = os.path.join(folder_path, video_file)
            keypoints, hand_missing_ratio = extract_video_keypoints(video_path)

            # Apply pose normalization (Step 2)
            keypoints = normalize_sequence(keypoints)

            # Save as .npy
            npy_filename = os.path.splitext(video_file)[0] + ".npy"
            npy_path = os.path.join(export_gesture_dir, npy_filename)
            np.save(npy_path, keypoints)

            status = "✓"
            if hand_missing_ratio > 0.3:
                status = "⚠ HANDS MISSING"
                problematic_videos.append((video_path, hand_missing_ratio))

            print(f"  {status} {video_file} → {keypoints.shape} "
                  f"(hand missing: {hand_missing_ratio:.1%})")

    # Log problematic videos
    if problematic_videos:
        print(f"\n{'='*60}")
        print(f"⚠ WARNING: {len(problematic_videos)} videos with >30% missing hands:")
        for path, ratio in problematic_videos:
            print(f"  - {path} ({ratio:.1%})")
        print(f"{'='*60}")

    print(f"\n✅ Extraction complete. Saved to {export_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keypoints from videos")
    parser.add_argument('--source', type=str, default="greetings_data", help="Folder containing raw videos")
    parser.add_argument('--dest', type=str, default=os.path.join("data", "keypoints_raw"), help="Output folder")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"ERROR: '{args.source}' not found. Place raw videos there first.")
    else:
        process_all_videos(args.source, args.dest)

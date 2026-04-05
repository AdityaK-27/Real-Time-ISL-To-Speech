"""
utils.py — Shared utility functions

Contains helpers for MediaPipe detection, landmark drawing, and visualization
that are used by both extract_keypoints.py and main.py.
"""

import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Number of landmarks per body part
NUM_POSE_LANDMARKS = 25   # Upper body only (0–24)
NUM_HAND_LANDMARKS = 21
FEATURES_PER_FRAME = (NUM_POSE_LANDMARKS + NUM_HAND_LANDMARKS * 2) * 3  # 201


def mediapipe_detection(image, model):
    """
    Run MediaPipe Holistic detection on a single frame.

    Args:
        image: BGR image from OpenCV
        model: MediaPipe Holistic model instance

    Returns:
        tuple: (annotated image, MediaPipe results)
    """
    if image is None:
        return None, None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, results


def extract_landmarks(results):
    """
    Extract keypoints from MediaPipe results.
    Returns 201 values: 25 pose (3D) + 21 LH (3D) + 21 RH (3D).
    """
    # Pose: landmarks 0–24, 3D
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark[:NUM_POSE_LANDMARKS]]
        ).flatten()
    else:
        pose = np.zeros(NUM_POSE_LANDMARKS * 3)

    # Left hand
    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        ).flatten()
    else:
        lh = np.zeros(NUM_HAND_LANDMARKS * 3)

    # Right hand
    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        ).flatten()
    else:
        rh = np.zeros(NUM_HAND_LANDMARKS * 3)

    return np.concatenate([pose, lh, rh])


def draw_styled_landmarks(image, results):
    """Draw pose and hand landmarks on the image."""
    if image is None:
        return

    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )

    # Draw left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )

    # Draw right hand connections
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )


def prob_viz(res, actions, input_frame):
    """
    Draw prediction probabilities on the frame.

    Args:
        res: prediction probability array (softmax output)
        actions: list of gesture names
        input_frame: BGR image to draw on
    """
    output_frame = input_frame.copy()
    if res is not None:
        for num, prob in enumerate(res):
            cv2.putText(
                output_frame,
                f"{actions[num]} : {int(prob * 100)}%",
                (10, 85 + num * 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA
            )
    else:
        for num in range(len(actions)):
            cv2.putText(
                output_frame,
                f"{actions[num]} : 0%",
                (10, 85 + num * 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA
            )
    return output_frame

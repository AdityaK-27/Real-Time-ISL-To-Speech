"""
main.py — Real-Time ISL Gesture Detection using ST-GCN (PyTorch)

Captures webcam feed, extracts keypoints using MediaPipe Holistic,
runs ST-GCN inference in real time, and provides text-to-speech output.

Controls:
  'd' — Start/stop detection
  's' — Stop camera, prepare for speech
  'r' — Read aloud detected gestures and exit
  'q' — Quit

Usage:
  python main.py                          # uses best model from full_pipeline
  python main.py --model baseline         # uses a specific experiment's model
"""

import os
import sys
import warnings
import argparse
import time
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn.functional as F

from utils import mediapipe_detection, extract_landmarks, draw_styled_landmarks, prob_viz
from normalize_pose import normalize_sequence
from stgcn_model import STGCN, reshape_input, create_model

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# --- Configuration ---
SEQUENCE_LENGTH = 60     # Number of frames for one prediction
CONFIDENCE_THRESHOLD = 0.80
NUM_CLASSES = 9


def load_stgcn_model(experiment_name='full_pipeline', device='cpu'):
    """
    Load a trained ST-GCN model from the models/ directory.

    Args:
        experiment_name: which experiment's model to load
        device: 'cpu' or 'cuda'

    Returns:
        tuple: (model, class_names)
    """
    model_path = os.path.join("models", experiment_name, "best_stgcn.pth")

    if not os.path.exists(model_path):
        # Fallback: try other experiments
        for fallback in ['full_pipeline', 'aug_only', 'kalman_only', 'baseline']:
            fallback_path = os.path.join("models", fallback, "best_stgcn.pth")
            if os.path.exists(fallback_path):
                model_path = fallback_path
                experiment_name = fallback
                print(f"Model '{experiment_name}' not found. Using '{fallback}' instead.")
                break
        else:
            raise FileNotFoundError(
                "No trained ST-GCN model found. Run train.py or run_experiments.py first."
            )

    print(f"Loading ST-GCN model from: {model_path}")

    model = create_model(num_classes=NUM_CLASSES, device=device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    class_names = checkpoint.get('class_names', [
        'alright', 'good_afternoon', 'good_evening', 'good_morning',
        'good_night', 'hello', 'how_are_you', 'pleased', 'thank_you'
    ])

    # Make class names human-readable for display
    display_names = [name.replace('_', ' ').title() for name in class_names]

    print(f"Loaded model with {len(class_names)} classes: {display_names}")
    return model, class_names, display_names


def predict_gesture(model, sequence, device):
    """
    Run inference on a keypoint sequence.

    Args:
        model: trained ST-GCN model
        sequence: np.ndarray of shape (60, 201)
        device: torch device

    Returns:
        tuple: (probabilities array, predicted class index, confidence)
    """
    # Normalize the sequence
    normalized = normalize_sequence(sequence)

    # Reshape: (1, 60, 201) → (1, 3, 60, 67)
    x = reshape_input(normalized[np.newaxis, ...]).to(device)

    with torch.no_grad():
        output = model(x)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]

    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]

    return probs, pred_idx, confidence


def main():
    parser = argparse.ArgumentParser(description="Real-Time ISL Gesture Detection")
    parser.add_argument('--model', default='full_pipeline',
                        help='Experiment name to load model from')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    try:
        model, class_names, display_names = load_stgcn_model(args.model, device)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # State variables
    sequence = []
    last_detected = ""
    detected_gestures = []
    res = None
    detecting = False
    camera_active = True

    mp_holistic = mp.solutions.holistic

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print("\n" + "=" * 50)
    print("Real-Time ISL Gesture Detection (ST-GCN)")
    print("=" * 50)
    print("Controls:")
    print("  'd' — Start/stop detection")
    print("  's' — Stop camera, prepare for speech")
    print("  'r' — Read aloud detected gestures")
    print("  'q' — Quit")
    print("=" * 50 + "\n")

    with mp_holistic.Holistic(
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        model_complexity=1
    ) as holistic:

        while camera_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            key = cv2.waitKey(5) & 0xFF

            # Toggle detection
            if key == ord('d'):
                detecting = not detecting
                if detecting:
                    sequence = []  # Reset sequence when starting detection
                    print("Detection: ON")
                else:
                    print("Detection: OFF")

            # Stop camera
            elif key == ord('s'):
                cap.release()
                cv2.destroyAllWindows()
                camera_active = False
                print("\nCamera stopped. Press 'r' to hear detected gestures.")
                break

            # Quit
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                camera_active = False
                break

            if detecting:
                frame = cv2.resize(frame, (640, 480))
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # Extract keypoints (201 features, 3D)
                keypoints = extract_landmarks(results)
                sequence.append(keypoints)

                # Keep only last SEQUENCE_LENGTH frames
                if len(sequence) > SEQUENCE_LENGTH:
                    sequence = sequence[-SEQUENCE_LENGTH:]

                # Run prediction when we have enough frames
                if len(sequence) == SEQUENCE_LENGTH:
                    seq_array = np.array(sequence)
                    probs, pred_idx, confidence = predict_gesture(
                        model, seq_array, device
                    )
                    res = probs

                    pred_display = display_names[pred_idx]

                    if confidence > CONFIDENCE_THRESHOLD:
                        if not detected_gestures or pred_display != detected_gestures[-1]:
                            detected_gestures.append(pred_display)
                            last_detected = pred_display
                            print(f"  Detected: {pred_display} ({confidence:.2f})")

                # Draw probability visualization
                image = prob_viz(res, display_names, image)

                # Draw detection bar
                cv2.rectangle(image, (0, 0), (width, 60), (0, 0, 0), -1)

                if res is not None and last_detected:
                    cv2.putText(
                        image,
                        f"{last_detected} ({res[np.argmax(res)]:.2f})",
                        (3, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA
                    )

                cv2.imshow('ISL Gesture Detection (ST-GCN)', image)

            else:
                cv2.putText(
                    frame, "Press 'd' to start detection", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                cv2.imshow('ISL Gesture Detection (ST-GCN)', frame)

    # === SPEECH OUTPUT ===
    while True:
        key = input("Press 'r' to read detected gestures aloud, 'q' to quit: ").strip().lower()
        if key == 'r' and detected_gestures:
            try:
                import pyttsx3
                engine = pyttsx3.init('sapi5')
                voices = engine.getProperty('voices')
                engine.setProperty('voice', voices[0].id)
                engine.setProperty('rate', 170)
                engine.setProperty('volume', 1.0)

                sentence = "The gestures detected were: " + ", ".join(detected_gestures) + "."
                time.sleep(1.0)
                print(f"\nSpeaking: {sentence}")
                engine.say(sentence)
                engine.runAndWait()
                time.sleep(1.0)
                print("Done speaking. Exiting...\n")
            except ImportError:
                print("pyttsx3 not installed. Detected gestures:")
                print(", ".join(detected_gestures))
            break
        elif key == 'q':
            print("Exiting without speaking.\n")
            break
        else:
            print("No gestures detected or invalid key. Try again.")


if __name__ == "__main__":
    main()

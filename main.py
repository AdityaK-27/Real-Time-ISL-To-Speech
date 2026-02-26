import os
import warnings
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import time
from utils import mediapipe_detection, landmarks_data, prob_viz
from models import load_model

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    sequence = []
    last_detected = ""
    detected_gestures = []  # store detected gestures
    res = None
    thresh = 0.85
    detecting = False
    camera_active = True

    mp_holistic = mp.solutions.holistic

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'models', 'lstm_v3')
    model = load_model(model_dir, pretrained=True, training=False)

    # Clean up gesture names (remove numbers/dots)
    actions = [a.strip() for a in os.listdir('greetings_data')]
    actions = [a.split('.', 1)[-1].strip() if '.' in a else a for a in actions]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print("\nControls:")
    print("Press 'd' to start/stop detection")
    print("Press 's' to stop camera and prepare for speech")
    print("Press 'r' to read aloud detected gestures and exit")
    print("Press 'q' to quit\n")

    with mp_holistic.Holistic(
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        model_complexity=0
    ) as holistic:

        while camera_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            key = cv2.waitKey(5) & 0xFF

            # Toggle detection
            if key == ord('d'):
                detecting = not detecting

            # Stop camera when 's' pressed
            elif key == ord('s'):
                cap.release()
                cv2.destroyAllWindows()
                camera_active = False
                print("\nCamera stopped. Press 'r' to hear detected gestures.\n")
                break

            # Quit manually
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                camera_active = False
                break

            if detecting:
                frame = cv2.resize(frame, (640, 480))
                image, results = mediapipe_detection(frame, holistic)
                keypoints = landmarks_data(results)
                sequence.append(keypoints)

                if len(sequence) > 30:
                    sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    pred_action = actions[np.argmax(res)]
                    conf = res[np.argmax(res)]

                    if conf > thresh and (not detected_gestures or pred_action != detected_gestures[-1]):
                        detected_gestures.append(pred_action)
                        last_detected = pred_action
                        print(pred_action)

                image = prob_viz(res, actions, frame)
                cv2.rectangle(image, (0, 0), (width, 60), (0, 0, 0), -1)

                if res is not None and last_detected:
                    cv2.putText(image,
                                f"{last_detected} ({res[np.argmax(res)]:.2f})",
                                (3, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA)

                cv2.imshow('OpenCV Feed', image)

            else:
                cv2.putText(frame, "Press 'd' to start detection", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('OpenCV Feed', frame)

    # === SPEECH CONTROL (after camera closed) ===
    while True:
        key = input("Press 'r' to read detected gestures aloud and exit: ").strip().lower()
        if key == 'r' and detected_gestures:
            # Initialize pyttsx3 after camera is closed
            engine = pyttsx3.init('sapi5')
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[0].id)
            engine.setProperty('rate', 170)
            engine.setProperty('volume', 1.0)

            # Build one complete sentence in the same detection order
            sentence = "The gestures detected were: " + ", ".join(detected_gestures) + "."

            # Give engine time to initialize
            time.sleep(1.0)

            print("\nSpeaking:\n", sentence)
            engine.say(sentence)
            engine.runAndWait()

            # Wait briefly so the audio finishes before exit
            time.sleep(1.0)
            print("\nDone speaking. Exiting program...\n")
            break

        elif key == 'q':
            print("\nExiting without speaking.\n")
            break

        else:
            print("No gestures detected or invalid key. Try again.")

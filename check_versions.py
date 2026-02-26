import sys

# Core Libraries
import tensorflow as tf
import mediapipe as mp
import cv2
import numpy as np
import pyttsx3

# Print versions in a clear format
print("=" * 60)
print("SYSTEM AND LIBRARY VERSIONS")
print("=" * 60)
print(f"Python Version     : {sys.version.split()[0]}")
print(f"TensorFlow Version : {tf.__version__}")
print(f"MediaPipe Version  : {mp.__version__}")
print(f"OpenCV Version     : {cv2.__version__}")
print(f"NumPy Version      : {np.__version__}")

# pyttsx3 doesn’t have a built-in __version__ attribute, so handle safely
try:
    import pkg_resources
    pyttsx3_version = pkg_resources.get_distribution('pyttsx3').version
except Exception:
    pyttsx3_version = "Unknown"
print(f"pyttsx3 Version    : {pyttsx3_version}")

print("=" * 60)

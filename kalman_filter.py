"""
kalman_filter.py — Step 3: Kalman Filtering

Applies a 1D Kalman filter independently on each of the 201 coordinate
dimensions across the 60 frames.

This smooths jitter in keypoint trajectories caused by MediaPipe detection noise.

Kalman parameters (tuned for MediaPipe keypoints):
  - Process noise (Q): 1e-2  (allows natural gesture motion through)
  - Measurement noise (R): 1e-2  (moderate trust in measurements)
  - Initial estimate error (P): 1.0

Reads from data/keypoints_raw/, saves to data/keypoints_processed/.
Keeps keypoints_raw/ intact for the comparison experiments.
"""

import os
import argparse
import numpy as np


def kalman_filter_1d(measurements, Q=1e-2, R=1e-2, P=1.0):
    """
    Apply a simple 1D Kalman filter to a sequence of measurements.

    The state model is a constant-position model:
      x_k = x_{k-1} + w_k   (w_k ~ N(0, Q))
      z_k = x_k + v_k        (v_k ~ N(0, R))

    Args:
        measurements: 1D array of length T
        Q: Process noise variance
        R: Measurement noise variance
        P: Initial estimate error variance

    Returns:
        Filtered 1D array of length T
    """
    n = len(measurements)
    filtered = np.zeros(n)

    # Initialize with first measurement
    x_hat = measurements[0]  # State estimate
    p = P                     # Error covariance

    for k in range(n):
        # --- Prediction step ---
        x_hat_minus = x_hat       # Predicted state (constant model)
        p_minus = p + Q           # Predicted error covariance

        # --- Update step ---
        K = p_minus / (p_minus + R)               # Kalman gain
        x_hat = x_hat_minus + K * (measurements[k] - x_hat_minus)  # Updated estimate
        p = (1 - K) * p_minus                      # Updated error covariance

        filtered[k] = x_hat

    return filtered


def apply_kalman(sequence, Q=1e-2, R=1e-2, P=1.0):
    """
    Apply Kalman filter to a full keypoint sequence.

    Filters each of the 201 coordinate dimensions independently
    across the 60 time frames.

    Args:
        sequence: np.ndarray of shape (T, 201)
        Q, R, P: Kalman filter parameters

    Returns:
        np.ndarray of shape (T, 201), smoothed
    """
    smoothed = sequence.copy().astype(np.float64)
    num_frames, num_features = smoothed.shape

    for dim in range(num_features):
        signal = smoothed[:, dim]

        # Skip all-zero dimensions (e.g., missing hand data)
        if np.all(signal == 0):
            continue

        smoothed[:, dim] = kalman_filter_1d(signal, Q=Q, R=R, P=P)

    return smoothed


def process_all_keypoints(input_dir, output_dir, Q=1e-2, R=1e-2, P=1.0):
    """
    Read all .npy files from input_dir, apply Kalman filter,
    and save to output_dir with the same folder structure.
    """
    for gesture in sorted(os.listdir(input_dir)):
        gesture_in = os.path.join(input_dir, gesture)
        if not os.path.isdir(gesture_in):
            continue

        gesture_out = os.path.join(output_dir, gesture)
        os.makedirs(gesture_out, exist_ok=True)

        npy_files = [f for f in os.listdir(gesture_in) if f.endswith('.npy')]
        print(f"  Kalman filtering: {gesture} ({len(npy_files)} files)")

        for npy_file in sorted(npy_files):
            data = np.load(os.path.join(gesture_in, npy_file))
            filtered = apply_kalman(data, Q=Q, R=R, P=P)
            np.save(os.path.join(gesture_out, npy_file), filtered)

    print(f"✅ Kalman filtering complete. Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Kalman filter to keypoints")
    parser.add_argument('--input', type=str, default=os.path.join("data", "keypoints_raw"), help="Input folder")
    parser.add_argument('--output', type=str, default=os.path.join("data", "keypoints_processed"), help="Output folder")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: '{args.input}' not found.")
    else:
        os.makedirs(args.output, exist_ok=True)
        process_all_keypoints(args.input, args.output)

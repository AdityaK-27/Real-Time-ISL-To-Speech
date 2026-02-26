# Real-Time Indian Sign Language (ISL) Translation

A deep learning-based system for real-time recognition and translation of Indian Sign Language (ISL) gestures using computer vision and neural networks.

## Overview

This project implements an end-to-end pipeline for detecting and recognizing Indian Sign Language gestures in real-time using a webcam. The system extracts hand and body keypoints from video frames and uses LSTM neural networks to classify the gestures into different sign categories.

## Features

- **Real-time Detection**: Recognize ISL gestures from live webcam feed
- **Multiple Model Architectures**: Support for various LSTM variants including:
  - LSTM v1, v2, v3
  - BiLSTM with Attention mechanism
  - Transformer-based models
- **MediaPipe Integration**: Efficient pose and hand keypoint extraction
- **Data Augmentation**: Gaussian noise addition for robust model training
- **Text-to-Speech**: Convert detected gestures to speech
- **High Accuracy**: Trained on multiple sign language categories

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV (cv2)
- MediaPipe
- NumPy
- Scikit-learn
- Matplotlib
- pyttsx3 (for text-to-speech)
- SciPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AdityaK-27/Real-Time-ISL-To-Speech
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Real-Time-ISL-Translation/
├── main.py                    # Real-time gesture detection from webcam
├── train.py                   # Model training script
├── evaluate.py                # Model evaluation script
├── keypoint_extraction.py     # Extract keypoints from video files
├── models.py                  # Neural network architectures
├── utils.py                   # Utility functions
├── check_versions.py          # Check library versions
├── greetings_data/            # Sample video data for gestures
├── keypoint_data/             # Extracted keypoint .npy files (organized by gesture)
├── models/                    # Pre-trained model weights
│   ├── lstm_v1/
│   ├── lstm_v2/
│   ├── lstm_v3/
│   ├── bilstm_attention/
│   └── transformer/
└── results/                   # Evaluation results and metrics
```

## Gesture Categories

The system recognizes the following ISL gestures:

- Hello
- How are you
- Alright
- Good Morning
- Good Afternoon
- Good Evening
- Good Night
- Thank you
- Pleased

## Usage

### 1. Extract Keypoints from Videos

Convert raw video files to keypoint data:

```bash
python keypoint_extraction.py
```

This extracts hand, pose, and face landmarks using MediaPipe and saves them as NumPy arrays.

### 2. Train a Model

Train a model on the extracted keypoint data:

```bash
python train.py
```

The training script:
- Loads keypoint data from `keypoint_data/` directory
- Splits data into training (80%) and testing (20%) sets
- Applies Gaussian noise for data augmentation
- Saves the best model checkpoint based on validation loss
- Uses device acceleration (GPU if available)

### 3. Evaluate Model Performance

Evaluate a trained model:

```bash
python evaluate.py
```

This generates accuracy metrics and confusion matrices.

### 4. Real-Time Gesture Detection

Run real-time detection from your webcam:

```bash
python main.py
```

**Controls:**
- **'d'**: Start/stop detection
- **'s'**: Stop camera and prepare for speech
- **'r'**: Read aloud detected gestures and exit
- **'q'**: Quit the application

The system will display:
- Live video feed with pose landmarks
- Detected gesture with confidence probability
- Prediction threshold: 0.85 (only shows predictions above this confidence)

## Model Architecture Details

### LSTM Variants

The project includes multiple LSTM-based architectures:

- **LSTM v1**: Sequential LSTM layers with increasing/decreasing units
- **LSTM v2**: Enhanced architecture with optimized dimensionality
- **LSTM v3**: Further improvements with better feature extraction
- **BiLSTM with Attention**: Bidirectional LSTM with attention mechanism for focus on important frames
- **Transformer**: Transformer-based architecture for sequence modeling

### Input Shape

- Sequences: 30 frames (temporal dimension)
- Keypoints per frame: 150 values (combination of hand, pose, and face landmarks)
- Input shape: `(batch_size, 30, 150)`

## Training Details

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Data Augmentation**: Gaussian noise (μ=0, σ=0.01)
- **Regularization**: Dropout layers to prevent overfitting
- **Validation Monitoring**: Early stopping based on validation loss
- **Frame Skipping**: Skip frames during video processing (default: every 2nd frame)

## Dataset Structure

```
keypoint_data/
├── hello/
│   ├── MVI_0001_skip_2.npy
│   ├── MVI_0002_skip_2.npy
│   └── ...
├── good_morning/
│   ├── MVI_0046_skip_2.npy
│   └── ...
└── [other gestures]/
```

Each `.npy` file contains keypoint sequences with shape `(num_frames, 150)`.

## Configuration

Key parameters can be modified in the scripts:

- **Confidence Threshold** (`main.py`): `thresh = 0.85` - Minimum confidence for gesture recognition
- **Max Frame Length** (`keypoint_extraction.py`): `max_frame_length=30` - Number of frames per sequence
- **Frame Skip** (`keypoint_extraction.py`): `skip_frame=2` - Process every nth frame
- **MediaPipe Confidence**: Adjustable in holistic model initialization

## Performance Metrics

- The system achieves high accuracy on recognition tasks
- Evaluation results are saved in `evaluation_results.txt`
- Confusion matrices help identify misclassified gestures

## Troubleshooting

1. **Webcam not detected**: Ensure your webcam is properly connected and accessible
2. **Poor detection accuracy**: Ensure good lighting and clear visibility of hands/body
3. **GPU not detected**: Install compatible CUDA/cuDNN for TensorFlow GPU support
4. **Out of memory**: Reduce batch size in training or use a lighter model version

## Future Enhancements

- Support for more ISL gestures
- Real-time translation to multiple languages
- Mobile app deployment
- Improved pose estimation accuracy
- Support for continuous sentence recognition

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

**Note**: This system is designed to recognize isolated gestures. For continuous sign language translation, additional processing and context modeling would be required.

import os
import json
import time
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logs

# Define standard variables
NUM_CLASSES = 9
NUM_FRAMES = 60
NUM_FEATURES = 201
SEED = 42

def get_lstm_experiment_config(experiment_name):
    """Configuration mapping for LSTM experiments."""
    configs = {
        "lstm_baseline": {"dir": os.path.join("data", "keypoints_raw")},
        "lstm_large_raw": {"dir": os.path.join("data", "keypoints_large_raw")}
    }
    if experiment_name not in configs:
        raise ValueError(f"Unknown experiment: {experiment_name}. Available: {list(configs.keys())}")
    return configs[experiment_name]

def load_data_as_numpy(data_dir):
    """Load all .npy files into a numpy array (X) and label array (y)."""
    sequences, labels = [], []
    classes = sorted(os.listdir(data_dir))
    
    for label, action in enumerate(classes):
        action_dir = os.path.join(data_dir, action)
        if not os.path.isdir(action_dir): continue
            
        for seq_file in os.listdir(action_dir):
            if not seq_file.endswith('.npy'): continue
            res = np.load(os.path.join(action_dir, seq_file))
            # If shape is (60, 201), keep it. Keras LSTM expects (batch, time_steps, features)
            if res.shape == (NUM_FRAMES, NUM_FEATURES):
                sequences.append(res)
                labels.append(label)
    
    return np.array(sequences), np.array(labels), classes

def build_lstm_v3(input_shape=(NUM_FRAMES, NUM_FEATURES), num_classes=NUM_CLASSES):
    """The LSTMv3 architecture from archive/models.py, modified for 60x201 inputs."""
    model = Sequential()
    # Updated input shape to match new pipeline
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128, return_sequences=False))

    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4) # Standardized learning rate
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_lstm_model(experiment_name, num_epochs=150, batch_size=8):
    """Main training loop mimicking the PyTorch train.py workflow."""
    print(f"\n{'='*60}")
    print(f"Starting LSTM Training: {experiment_name}")
    print(f"{'='*60}")
    
    config = get_lstm_experiment_config(experiment_name)
    data_dir = config["dir"]
    model_dir = os.path.join("models", experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Load Data
    X, y, classes = load_data_as_numpy(data_dir)
    print(f"Loaded {len(X)} total sequences from {data_dir}.")
    
    # 2. Train/Val/Test Split (70/15/15) exactly like PyTorch
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)
    
    print(f"Train/Val/Test distribution: {len(X_train)} / {len(X_val)} / {len(X_test)}")
    
    # Save test data for evaluation step
    np.save(os.path.join(model_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(model_dir, 'y_test.npy'), y_test)
    
    # 3. Build Model
    model = build_lstm_v3(num_classes=len(classes))
    model.summary()
    
    # 4. Callbacks (Early Stopping & Model Checkpoint)
    checkpoint_path = os.path.join(model_dir, 'best_lstm.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=False, verbose=1
    )
    
    # 5. Train Model
    start_time = time.time()
    keras_history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    training_time = time.time() - start_time
    
    # 6. Save standard format history for unified evaluation
    history = {
        'train_loss': [float(x) for x in keras_history.history['loss']],
        'train_acc': [float(x) for x in keras_history.history['accuracy']],
        'val_loss': [float(x) for x in keras_history.history['val_loss']],
        'val_acc': [float(x) for x in keras_history.history['val_accuracy']],
        'best_epoch': int(np.argmin(keras_history.history['val_loss']) + 1),
        'training_time_s': float(training_time),
        'class_names': classes
    }
    
    history_file = os.path.join(model_dir, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
        
    print(f"\n{chr(10004)} Training complete in {training_time/60:.2f} mins.")
    print(f"Saved best model to {checkpoint_path}")
    print(f"Saved history to {history_file}")
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the LSTMv3 Model")
    parser.add_argument("experiment", type=str, choices=["lstm_baseline", "lstm_large_raw"], help="Experiment configuration")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    
    train_lstm_model(args.experiment, args.epochs)

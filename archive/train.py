import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from models import load_model

device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
IMPORT_PATH = os.path.join('keypoint_data')
actions = os.listdir(IMPORT_PATH)
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    for videofile in os.listdir(os.path.join(IMPORT_PATH, action)):
        data = np.load(os.path.join(IMPORT_PATH, action, videofile))
        sequences.append(data)
        labels.append(label_map[action])

X = np.array(sequences, dtype = np.float32)
y = to_categorical(labels)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,  
    stratify=y,
    random_state=42
)

# Gussian Noise Addition for Data Augmentation [Key-Point shift]
noise = np.random.normal(0, 0.01, X_train.shape)
X_train = X_train + noise

model = load_model('bilstm_attention', pretrained=False, device=device)

model_save_path = os.path.join("models", "bilstm_attention")
os.makedirs(model_save_path, exist_ok=True)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_save_path, "isl_model.h5"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=8,
        min_lr=1e-6,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
]

with tf.device(device):
    history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)
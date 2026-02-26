import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from models import AttentionLayer


# -----------------------------------------------------------
# 🔹 1. Load Dataset
# -----------------------------------------------------------
IMPORT_PATH = os.path.join("keypoint_data")
actions = sorted(os.listdir(IMPORT_PATH))
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    for videofile in os.listdir(os.path.join(IMPORT_PATH, action)):
        data = np.load(os.path.join(IMPORT_PATH, action, videofile))
        sequences.append(data)
        labels.append(label_map[action])

X = np.array(sequences, dtype=np.float32)
y = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

y_true = np.argmax(y_test, axis=1)


# -----------------------------------------------------------
# 🔹 2. Function to Evaluate Model
# -----------------------------------------------------------
def evaluate_model(model_name):

    print("\n==============================")
    print(f"Evaluating Model: {model_name}")
    print("==============================")

    model_dir = os.path.join("models", model_name)
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]
    model_path = os.path.join(model_dir, model_files[0])

    if model_name == "bilstm_attention":
        model = keras_load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
    else:
        model = keras_load_model(model_path)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nAccuracy: {acc * 100:.2f}%")
    print(f"Loss: {loss:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=actions))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=actions,
                yticklabels=actions,
                cmap="Blues")

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Save image
    os.makedirs("results", exist_ok=True)
    image_path = os.path.join("results", f"confusion_matrix_{model_name}.png")
    plt.savefig(image_path, dpi=300, bbox_inches='tight')

    print(f"\nConfusion matrix saved at: {image_path}")

    return acc


# -----------------------------------------------------------
# 🔹 3. Compare Both Models
# -----------------------------------------------------------
lstm_acc = evaluate_model("lstm_v3")
bilstm_acc = evaluate_model("bilstm_attention")

print("\n==============================")
print("Final Comparison")
print("==============================")
print(f"LSTM Accuracy: {lstm_acc * 100:.2f}%")
print(f"BiLSTM + Attention Accuracy: {bilstm_acc * 100:.2f}%")
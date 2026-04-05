import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Need to set this before loading keras models occasionally
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_lstm_experiment(experiment_name):
    """Load the trained LSTM model, test data, and training history."""
    model_dir = os.path.join("models", experiment_name)

    # Load model
    model_path = os.path.join(model_dir, "best_lstm.h5")
    model = tf.keras.models.load_model(model_path)

    # Load test data
    X_test = np.load(os.path.join(model_dir, "X_test.npy"))
    y_test = np.load(os.path.join(model_dir, "y_test.npy"))

    # Load history
    with open(os.path.join(model_dir, "training_history.json"), 'r') as f:
        history = json.load(f)

    class_names = history.get('class_names', [])
    return model, X_test, y_test, history, class_names


def plot_training_curves(history, save_dir, experiment_name):
    """Plot training vs validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training vs Validation Loss\n({experiment_name})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Training vs Validation Accuracy\n({experiment_name})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir, experiment_name):
    """Plot custom confusion matrix showing raw counts."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Counts)\n({experiment_name})', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()


def plot_roc_curves(y_true, y_probs, class_names, save_dir, experiment_name):
    """Plot ROC curves and compute mean AUC."""
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    plt.figure(figsize=(10, 8))
    auc_scores = []
    
    for i, class_name in enumerate(class_names):
        if y_bin.shape[1] > i:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC={roc_auc:.3f})')
            
    mean_auc = np.mean(auc_scores) if auc_scores else 0
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves\n({experiment_name}) — Mean AUC: {mean_auc:.3f}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    return mean_auc


def measure_inference_time(model, X_sample, num_runs=100):
    """Measure inference time (milliseconds) per iteration for TensorFlow model."""
    x = X_sample[:1]
    
    # Warmup
    for _ in range(10): model.predict(x, verbose=0)
        
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model.predict(x, verbose=0)
        times.append((time.perf_counter() - start) * 1000)
        
    return np.mean(times), np.std(times)


def evaluate_lstm_experiment(experiment_name):
    """Full evaluation generation."""
    print(f"\n{'='*60}")
    print(f"Evaluating LSTM: {experiment_name}")
    print(f"{'='*60}")

    save_dir = os.path.join("results", experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    model, X_test, y_test, history, class_names = load_lstm_experiment(experiment_name)

    # 1. Prediction mapping
    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    # 2. Training curves
    plot_training_curves(history, save_dir, experiment_name)

    # 3. Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, class_names, save_dir, experiment_name)

    # 4. Classification Report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    
    print(f"\n{report_text}")
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Experiment: {experiment_name}\n{'='*60}\n\n{report_text}")

    # 5. ROC Curves
    mean_auc = plot_roc_curves(y_test, y_probs, class_names, save_dir, experiment_name)

    # 6. Inference time
    avg_t, std_t = measure_inference_time(model, X_test)
    print(f"  ✓ Inference time: {avg_t:.2f} \N{PLUS-MINUS SIGN} {std_t:.2f} ms")

    # 7. Standardized summary JSON
    summary = {
        "experiment": experiment_name,
        "test_accuracy": float(report.get('accuracy', 0)),
        "macro_f1": float(report.get('macro avg', {}).get('f1-score', 0)),
        "mean_auc": float(mean_auc),
        "inference_time_ms": float(avg_t),
        "inference_time_std_ms": float(std_t),
        "best_epoch": int(history.get('best_epoch', 0)),
        "num_test_samples": len(y_test)
    }

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"  ✓ Summary saved to {save_dir}/summary.json")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_lstm.py <experiment_name>")
    else:
        evaluate_lstm_experiment(sys.argv[1])

"""
evaluate.py — Step 6: Comprehensive Evaluation

For each experiment generates:
  - Training vs validation loss curve
  - Training vs validation accuracy curve
  - Confusion matrix (normalized)
  - Classification report (precision, recall, F1 per class + macro average)
  - ROC curves (one-vs-rest) + mean AUC
  - Inference time per sample in milliseconds

All plots saved to results/<experiment_name>/
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from stgcn_model import STGCN, reshape_input, create_model


NUM_CLASSES = 9


def load_experiment(experiment_name, device):
    """
    Load the trained model, test data, and training history for an experiment.
    """
    model_dir = os.path.join("models", experiment_name)

    # Load model
    model = create_model(num_classes=NUM_CLASSES, device=device)
    checkpoint = torch.load(
        os.path.join(model_dir, "best_stgcn.pth"),
        map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test data
    X_test = np.load(os.path.join(model_dir, "X_test.npy"))
    y_test = np.load(os.path.join(model_dir, "y_test.npy"))

    # Load history
    with open(os.path.join(model_dir, "training_history.json"), 'r') as f:
        history = json.load(f)

    class_names = checkpoint.get('class_names', history.get('class_names', []))

    return model, X_test, y_test, history, class_names


def plot_training_curves(history, save_dir, experiment_name):
    """Plot training vs validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curve
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Training vs Validation Loss\n({experiment_name})', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Training vs Validation Accuracy\n({experiment_name})', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Training curves saved")


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir, experiment_name):
    """Plot raw count confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    # Plot exact numbers (integer formatting)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Counts)\n({experiment_name})', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix saved")


def plot_normalized_confusion_matrix(y_true, y_pred, class_names, save_dir, experiment_name):
    """Plot normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(10, 8))
    # Plot percentage formatting
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Normalized Confusion Matrix\n({experiment_name})', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Normalized confusion matrix saved")


def plot_roc_curves(y_true, y_probs, class_names, save_dir, experiment_name):
    """Plot one-vs-rest ROC curves + mean AUC."""
    # Binarize labels
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
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves (One-vs-Rest)\n({experiment_name}) — Mean AUC: {mean_auc:.3f}',
              fontsize=13)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ ROC curves saved (Mean AUC: {mean_auc:.3f})")

    return mean_auc


def measure_inference_time(model, X_sample, device, num_runs=100):
    """Measure average inference time per sample in milliseconds."""
    model.eval()
    x = reshape_input(X_sample[:1]).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    # Time it
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time


def evaluate_experiment(experiment_name):
    """
    Full evaluation for one experiment.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"Evaluating: {experiment_name}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Output directory
    save_dir = os.path.join("results", experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    # Load experiment
    model, X_test, y_test, history, class_names = load_experiment(
        experiment_name, device
    )

    # Get predictions
    X_test_t = reshape_input(X_test).to(device)
    with torch.no_grad():
        outputs = model(X_test_t)
        y_probs = F.softmax(outputs, dim=1).cpu().numpy()
        y_pred = np.argmax(y_probs, axis=1)

    y_true = y_test

    # 1. Training curves
    plot_training_curves(history, save_dir, experiment_name)

    # 2. Confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, save_dir, experiment_name)
    plot_normalized_confusion_matrix(y_true, y_pred, class_names, save_dir, experiment_name)

    # 3. Classification report
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=class_names,
                                        zero_division=0)

    print(f"\n{report_text}")

    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(report_text)
    print(f"  ✓ Classification report saved")

    # 4. ROC curves
    mean_auc = plot_roc_curves(y_true, y_probs, class_names, save_dir, experiment_name)

    # 5. Inference time
    avg_time, std_time = measure_inference_time(model, X_test, device)
    print(f"  ✓ Avg inference time: {avg_time:.2f} ± {std_time:.2f} ms/sample")

    # 6. Save summary
    summary = {
        "experiment": experiment_name,
        "test_accuracy": float(report.get('accuracy', 0)),
        "macro_f1": float(report.get('macro avg', {}).get('f1-score', 0)),
        "mean_auc": float(mean_auc),
        "inference_time_ms": float(avg_time),
        "inference_time_std_ms": float(std_time),
        "best_epoch": history.get('best_epoch', 'N/A'),
        "num_test_samples": len(y_test),
    }

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary saved to {save_dir}/summary.json")

    return summary


if __name__ == "__main__":
    experiment = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    evaluate_experiment(experiment)

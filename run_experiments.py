"""
run_experiments.py — Master Script

Runs the full ISL gesture recognition pipeline for all 4 comparison experiments.
Everything else is kept identical: same model architecture, same seed=42,
same train/val/test split.

Experiments:
  1. baseline      — raw keypoints, no augmentation, no Kalman
  2. aug_only      — raw keypoints + augmentation, no Kalman
  3. kalman_only   — processed keypoints (Kalman), no augmentation
  4. full_pipeline — processed keypoints (Kalman) + augmentation

Usage:
  python run_experiments.py              # run all 4 experiments
  python run_experiments.py baseline     # run a single experiment
  python run_experiments.py --extract    # re-extract keypoints before training
"""

import os
import sys
import json
import argparse

# Pipeline step imports
from extract_keypoints import process_all_videos
from kalman_filter import process_all_keypoints
from train import train_model
from evaluate import evaluate_experiment


EXPERIMENTS = ["baseline", "aug_only", "kalman_only", "full_pipeline", "offline_aug", "kalman_off_aug", "large_raw", "large_kalman"]


def run_extraction(import_path="greetings_data", export_path=os.path.join("data", "keypoints_raw")):
    """Step 1+2: Extract keypoints from raw videos and apply normalization."""
    print("\n" + "=" * 60)
    print(f"STEP 1+2: Keypoint Extraction + Pose Normalization")
    print(f"Source: {import_path} -> Dest: {export_path}")
    print("=" * 60)

    if not os.path.exists(import_path):
        print(f"ERROR: '{import_path}' folder not found!")
        return False

    os.makedirs(export_path, exist_ok=True)
    process_all_videos(import_path, export_path)
    return True


def run_kalman(input_dir=os.path.join("data", "keypoints_raw"), output_dir=os.path.join("data", "keypoints_processed")):
    """Step 3: Apply Kalman filtering."""
    print("\n" + "=" * 60)
    print(f"STEP 3: Kalman Filtering")
    print(f"Source: {input_dir} -> Dest: {output_dir}")
    print("=" * 60)

    if not os.path.exists(input_dir):
        print(f"ERROR: '{input_dir}' not found. Run extraction first.")
        return False

    os.makedirs(output_dir, exist_ok=True)
    process_all_keypoints(input_dir, output_dir)
    return True





def run_pipeline(experiments=None, extract=False):
    """
    Run the full pipeline.

    Args:
        experiments: list of experiment names to run (default: all 4)
        extract: if True, re-extract keypoints from videos
    """
    if experiments is None:
        experiments = EXPERIMENTS

    # --- Data preparation ---
    if extract:
        if not run_extraction():
            print("Extraction failed. Aborting.")
            return

    # Check if raw data exists
    raw_dir = os.path.join("data", "keypoints_raw")
    if not os.path.exists(raw_dir) or len(os.listdir(raw_dir)) == 0:
        print(f"No data in {raw_dir}. Running extraction...")
        if not run_extraction():
            print("Extraction failed. Aborting.")
            return

    # Run Kalman if needed
    if any(exp in experiments for exp in ["kalman_only", "full_pipeline"]):
        processed_dir = os.path.join("data", "keypoints_processed")
        if not os.path.exists(processed_dir) or len(os.listdir(processed_dir)) == 0:
            run_kalman()
            
    # --- LARGE DATASET PREPARATION ---
    # Check if large raw data exists
    if any(exp in experiments for exp in ["large_raw", "large_kalman"]):
        large_raw_dir = os.path.join("data", "keypoints_large_raw")
        if not os.path.exists(large_raw_dir) or len(os.listdir(large_raw_dir)) == 0:
            print(f"No data in {large_raw_dir}. Running extraction on greetings_data_large...")
            if not run_extraction(import_path="greetings_data_large", export_path=large_raw_dir):
                print("Large extraction failed. Aborting large dataset experiments.")
                
        # Run large Kalman if needed
        if "large_kalman" in experiments:
            large_processed_dir = os.path.join("data", "keypoints_large_processed")
            if not os.path.exists(large_processed_dir) or len(os.listdir(large_processed_dir)) == 0:
                run_kalman(input_dir=large_raw_dir, output_dir=large_processed_dir)

    # Augmentation is now handled dynamically in dataset.py (on-the-fly, train-only)
    # No pre-generated augmented data needed on disk

    # --- Training and evaluation ---
    all_results = {}

    for exp in experiments:
        print(f"\n{'#'*60}")
        print(f"# EXPERIMENT: {exp}")
        print(f"{'#'*60}")

        try:
            # Train
            history = train_model(exp)

            # Evaluate
            summary = evaluate_experiment(exp)
            all_results[exp] = summary

        except Exception as e:
            print(f"\n❌ ERROR in experiment '{exp}': {e}")
            import traceback
            traceback.print_exc()
            all_results[exp] = {"error": str(e)}

    # --- Final comparison ---
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Experiment':<20} {'Test Acc':>10} {'Macro F1':>10} {'Mean AUC':>10} {'Infer (ms)':>12}")
    print("-" * 70)

    for exp, result in all_results.items():
        if "error" in result:
            print(f"{exp:<20} {'ERROR':>10}")
        else:
            print(f"{exp:<20} "
                  f"{result.get('test_accuracy', 0)*100:>9.2f}% "
                  f"{result.get('macro_f1', 0):>10.4f} "
                  f"{result.get('mean_auc', 0):>10.4f} "
                  f"{result.get('inference_time_ms', 0):>11.2f}")

    # Save comparison
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "comparison.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nComparison saved to results/comparison.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ISL ST-GCN experiments")
    parser.add_argument('experiments', nargs='*', default=None,
                        help='Experiments to run (default: all 4)')
    parser.add_argument('--extract', action='store_true',
                        help='Re-extract keypoints from videos')
    args = parser.parse_args()

    experiments = args.experiments if args.experiments else None
    run_pipeline(experiments=experiments, extract=args.extract)

# main.py - Simple sequential pipeline for SWRD YOLO project
# This script runs each step of the pipeline in order, reproducing the SWRD paper's weld defect detection workflow.
# It uses PyTorch-based YOLO models (focus on YOLOX for free commercial licensing under Apache 2.0; Ultralytics YOLOv8 is AGPL-3.0, which may require open-sourcing modifications if distributed).
# Each step is controlled by a boolean flag (e.g., run_analyze = True). To skip a step, set it to False or comment out the if block.
# Steps use subprocess to run separate scripts, capturing output with UTF-8 encoding to handle Chinese labels from JSON.
# OpenCV (cv2) is used in preprocessing/enhancement/inference for image handling (e.g., reading TIFFs, CLAHE enhancement, patch extraction).
# To customize: Edit flags, arguments in the lists (e.g., ["--arg", "value"]), or paths. Run from project root: python main.py
# Logs errors to main_log.txt for debugging.
# Reproducibility: Assumes data in C:\AWrk\SWRD_YOLO_Project\data; outputs to processed_balanced/, processed_balanced_final/, models/, evaluation_results/ etc. Matches SWRD: 6 defect classes, sliding window (50% overlap, min area 100), balanced patches (defect/background), 9:1 train-val split in original paper but this emphasizes using a 0.8 / 0.1 / 0/1 train val test split instead.

import subprocess
import logging
import traceback
import os
import sys

logging.basicConfig(filename='main_log.txt', level=logging.ERROR, format='%(asctime)s - %(message)s')

    # run_script Helper function to run a Python script via subprocess.
    # Args:
    # - script_name: Name of the script in scripts/ folder (e.g., "analyze.py").
    # - args: Optional list of command-line arguments (e.g., ["--source", "processed_balanced", "--output", "processed_balanced_final"]).
    # Uses UTF-8 encoding to handle non-ASCII (e.g., Chinese defect labels in JSON).
    # Captures stdout/stderr, prints them, and returns True if successful (returncode 0).
    # Logs exceptions to main_log.txt.
# def run_script(script_name, args=None):

#     try:
#         script_path = os.path.join(os.path.dirname(__file__), "scripts", script_name)
#         cmd = [sys.executable, script_path]
#         if args:
#             cmd += args  # Append arguments as list elements (subprocess handles them separately).
#         result = subprocess.run(
#             cmd, 
#             capture_output=True, 
#             text=True, 
#             encoding='utf-8',
#             errors='replace'  # Replace invalid bytes to avoid decoding errors.
#         )
#         print(result.stdout)
#         if result.stderr:
#             print(f"Warnings/Errors: {result.stderr}")
#         return result.returncode == 0
#     except Exception as e:
#         error_msg = f"Unexpected error running {script_name}: {str(e)}\n{traceback.format_exc()}"
#         logging.error(error_msg)
#         print(error_msg)
#         return False

def run_script(script_name, args=None):
    try:
        script_path = os.path.join(os.path.dirname(__file__), "scripts", script_name)
        cmd = [sys.executable, script_path]
        if args:
            cmd += args
        
        # Use Popen for streaming output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1  # Line-buffered for real-time
        )
        
        # Stream stdout in real-time
        for line in process.stdout:
            print(line, end='')  # Print without extra newline
        
        # Wait for completion and get stderr
        stderr = process.communicate()[1]
        if stderr:
            print(f"Warnings/Errors: {stderr}")
        
        return process.returncode == 0
    except Exception as e:
        error_msg = f"Unexpected error running {script_name}: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        print(error_msg)
        return False

if __name__ == "__main__":
    # Step flags: Set to False to skip, or comment out the entire if block for that step.
    # To run only specific steps, set others to False.
    run_analyze = False  # Flag for dataset analysis.
    run_preprocess = False  # Flag for preprocessing with sliding window.
    run_balance = False  # Flag for dataset balancing.
    run_train = True  # Flag for training (choose YOLOv8 or YOLOX below).
    run_evaluate = False  # Flag for model evaluation.
    run_inference = False  # Flag for inference on full images.

    # Step 1: Analyze dataset
    # Runs scripts/analyze.py with no arguments.
    # What it does: Loads JSON annotations, counts defects per class (porosity, inclusion, etc.), generates stats (e.g., totals, per-class distribution), saves to dataset_totals.json, and creates visualizations (e.g., bar charts of class imbalance).
    # Reproduces SWRD: Identifies class imbalance (e.g., porosity dominant) to inform balancing.
    # Arguments: None. To customize, edit analyze.py directly (e.g., add plots).
    # Output: dataset_totals.json, class_distribution.png, etc.
    # Skip if: Already analyzed or not needed.
    if run_analyze:
        print("1. Running dataset analysis...")
        if run_script("analyze.py"):
            print("Analysis complete. Check dataset_totals.json and visualizations.")
        else:
            print("Analysis failed. Check main_log.txt for details.")

    #or CMD line use: python analyze.py

    # Step 2: Preprocess with sliding window and tracking
    # Runs scripts/preprocess_with_tracking.py with no arguments (defaults to train_val split).
    # What it does: Applies sliding window (window_size = min(h,w)//2 ~320-640, 50% overlap), filters to 6 valid classes, enhances images (contrast stretch + CLAHE via OpenCV), balances defect/background patches, splits 9:1 train-val (or 8:1:1 with --split train_val_test), saves YOLO-format (images/labels, dataset.yaml).
    # Reproduces SWRD: Patch-based preprocessing, min defect area 100, only 6 classes, enhancement for better detection.
    # Arguments: Optional --split (train_val or train_val_test). Add ["--split", "train_val_test"] if needed for test set.
    # Output: processed_balanced/ with images/labels/train/val, preprocessing_stats.json.
    # Skip if: Data already preprocessed.
    if run_preprocess:
        print("\n2. Running preprocessing with sliding window...")
        # Example: To use train_val_test split, add: args=["--split", "train_val_test"]
        if run_script("preprocess_with_tracking.py"):  # Add args=... if customizing.
            print("Preprocessing complete. Check processed_balanced/ folder.")
        else:
            print("Preprocessing failed. Check main_log.txt for details.")

    # Step 3: Balance dataset
    # Runs scripts/balance_dataset.py with --source and --output.
    # What it does: Balances classes via undersampling (e.g., porosity to ~5000), augmentation (e.g., undercut to ~2000 via rotate/flip/brightness via OpenCV), keeps others as-is or slightly augmented. Mixed strategy recommended.
    # Reproduces SWRD: Addresses imbalance for better mAP on minority classes (e.g., undercut AP50 0.775 post-balance).
    # Arguments: --source (input dir, default processed_balanced), --output (output dir, default processed_balanced_final), --strategy (undersample/mixed/oversample, default mixed).
    # To customize: Change args list (e.g., add "--strategy", "oversample").
    # Output: processed_balanced_final/ with balanced images/labels, balancing_stats.json.
    # Skip if: No balancing needed (but recommended for SWRD reproduction).
    if run_balance:
        print("\n3. Running dataset balancing...")
        if run_script("balance_dataset.py", args=["--source", "processed_balanced", "--output", "processed_balanced_final"]):
            print("Balancing complete. Check processed_balanced_final/ folder.")
        else:
            print("Balancing failed. Check main_log.txt for details.")

    # Step 4: Train model
    # Runs scripts/train_models.py with --data, --model, --size, --epochs.
    # What it does: Trains YOLO on balanced data using PyTorch. For YOLOv8 (Ultralytics): loads yolov8.pt, trains on dataset.yaml, uses class weights [0.5,1.0,1.0,3.0,1.5,1.5] for imbalance. For YOLOX (Apache 2.0, free for companies): similar, but more permissive license.
    # Reproduces SWRD: Object detection (not just classification) with bounding boxes, ~100 epochs, medium size for efficiency.
    # Arguments: --data (dataset dir), --model (yolov8 or yolox), --size (n/s/m/l/x), --epochs (int), --batch (optional, default 16).
    # Choose model: Use yolox for commercial freedom. Uncomment the desired train call.
    # Output: models/ with weights (best.pt), training_info.json, results.csv/plots.
    # Skip if: Model already trained.
    if run_train:
        print("\n4. Running model training...")
        # Option 1: Train YOLOv8m (comment out if using YOLOX)
        if run_script("train_models_optimized.py", args=["--data", "processed_balanced_final", "--size", "n", "--epochs", "10"]):
            print("Training complete. Check models/ folder for weights and logs.")
        
        # Option 2: Train YOLOX-m (recommended for free licensing; install via pip install yolox if needed)
        # if run_script("train_models.py", args=["--data", "processed_balanced_final", "--model", "yolox", "--size", "m", "--epochs", "100"]):
            # print("Training complete. Check models/ folder for weights and logs.")

        else:
            print("Training failed. Check main_log.txt for details.")
        
    # or CMD line use:  python scripts/train_models_optimized.py --size n --epochs 10
    # CMD Line Resume for YoloV8: python scripts\train_models_optimized.py --size n --epochs 10 --resume "C:\AWrk\SWRD_YOLO_Project\models\yolov8n_20250908_164110\train\weights\last.pt"

    # Step 5: Evaluate model
    # Runs scripts/evaluate_model.py with --weights and --split.
    # What it does: Evaluates trained model on splits (train/val/test), computes mAP50/ mAP50-95, precision/recall, per-class AP, generates plots (training curves, per-class bars) via Plotly, saves JSON/HTML.
    # Reproduces SWRD: Comprehensive metrics (e.g., mAP50 ~0.552 overall, undercut 0.775), visualizations for analysis.
    # Arguments: --weights (path to best.pt; update with your actual training run's path, e.g., from models/yolov8n_20250907_233859/train/weights/best.pt), --split (train/val/test/all).
    # To customize: Update weights path if training timestamp changes.
    # Output: evaluation_results/ with evaluation_metrics.json, training_plots.html, class_performance.html.
    # Skip if: No evaluation needed.
    if run_evaluate:
        print("\n5. Running model evaluation...")
        # Update --weights with your actual best.pt path (from training output).
        if run_script("evaluate_model.py", args=["--weights", "models/yolov8n_20250907_233859/train/weights/best.pt", "--split", "all"]):
            print("Evaluation complete. Check evaluation_results/ for metrics and plots.")
        else:
            print("Evaluation failed. Check main_log.txt for details.")

    # or CMD line use: python scripts/evaluate_model.py --weights "models/yolov8n_20250908_164110/train/weights/best.pt" --split all

    # Step 6: Run inference on full images
    # Runs scripts/inference_full_image.py with no arguments (hardcoded paths in script).
    # What it does: Applies sliding window (320x320, 50% overlap) to full weld TIFFs via OpenCV, runs detection with trained model, adjusts bboxes, applies NMS (IoU 0.45), outputs detections JSON.
    # Reproduces SWRD: Inference on originals (not patches), merging overlaps for accurate defect localization/classification.
    # Arguments: None (edit script for model_path/test_image). To add: Modify script or extend here.
    # Output: Printed detections; extend script for JSON/save as shown in prior advice.
    # Skip if: No inference needed.
    if run_inference:
        print("\n6. Running inference on full images...")
        if run_script("inference_full_image.py"):
            print("Inference complete. Check console or extend script for saved results.")
        else:
            print("Inference failed. Check main_log.txt for details.")

    print("\nPipeline complete! Customize flags/args as needed for reruns.")

    # or CMD line use: python scripts/inference_full_image_fixed.py --image "C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif" --output "C:\AWrk\SWRD_YOLO_Project\inference_results"

    
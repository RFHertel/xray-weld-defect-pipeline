# main.py - Simple sequential pipeline for SWRD YOLO project
# This script runs each step of the pipeline in order, reproducing the SWRD paper's weld defect detection workflow.
# It uses PyTorch-based YOLO models (focus on YOLOv8 for performance, with YOLOv5 as alternative; Ultralytics YOLOv8 is AGPL-3.0, which may require open-sourcing modifications if distributed, but YOLOv5 is MIT-licensed for free company use).
# Each step is controlled by a boolean flag (e.g., run_analyze = True). To skip a step, set it to False or comment out the if block.
# Steps use subprocess to run separate scripts, capturing output with UTF-8 encoding to handle Chinese labels from JSON.
# OpenCV (cv2) is used in preprocessing/enhancement/inference for image handling (e.g., reading TIFFs, CLAHE enhancement, patch extraction).
# To customize: Edit flags, arguments in the lists (e.g., ["--arg", "value"]), or paths. Run from project root: python main.py
# Logs errors to main_log.txt for debugging.
# Reproducibility: Assumes data in C:\AWrk\SWRD_YOLO_Project\data; outputs to processed_balanced/, processed_balanced_final/, models/, evaluation_results/ etc. Matches SWRD: 6 defect classes, sliding window (50% overlap, min area 100), balanced patches (defect/background), 9:1 train-val split in original paper but this emphasizes using a 0.8 / 0.1 / 0.1 train val test split instead.

import subprocess
import logging
import traceback
import os
import sys

logging.basicConfig(filename='main_log.txt', level=logging.ERROR, format='%(asctime)s - %(message)s')

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
    run_analyze = False  # Flag for Step 1: Dataset analysis.
    run_preprocess = False  # Flag for Step 2a: Preprocessing with sliding window.
    run_preprocess_verify = False  # Flag for Step 2b: Preprocess verification visualization.
    run_balance = False  # Flag for Step 3a: Dataset balancing.
    run_balance_check = False  # Flag for Step 3b: Check balanced dataset.
    run_analyze_undersample = False  # Flag for Step 3c: Analyze and undersample porosity.
    run_check_gpus = False  # Flag for Step 3d: Check GPU status.
    run_train = True  # Flag for Step 4: Model training (YOLOv8 or YOLOv5).
    run_evaluate = False  # Flag for Step 5: Model evaluation.
    run_inference = False  # Flag for Step 6: Inference on full images.

    # Step 1: Analyze dataset
    # Runs scripts/analyze.py with no arguments.
    # What it does: Loads JSON annotations, counts defects per class (porosity, inclusion, etc.), generates stats (e.g., totals, per-class distribution), saves to dataset_totals.json, and creates visualizations (e.g., bar charts of class imbalance).
    # CMD line: python analyze.py
    # Output: dataset_totals.json, class_distribution.png, etc.
    # Skip if: Already analyzed or not needed.
    if run_analyze:
        print("Step 1: Running dataset analysis...")
        if run_script("analyze.py"):
            print("Analysis complete. Check dataset_totals.json and visualizations.")
        else:
            print("Analysis failed. Check main_log.txt for details.")

    # Step 2a: Preprocess with sliding window and tracking
    # Runs scripts/preprocess_with_tracking.py with no arguments (defaults to train_val split).
    # What it does: Applies sliding window (window_size = min(h,w)//2 ~320-640, 50% overlap), filters to 6 valid classes, enhances images (contrast stretch + CLAHE via OpenCV), balances defect/background patches, splits 9:1 train-val (or 8:1:1 with --split train_val_test), saves YOLO-format (images/labels, dataset.yaml).
    # Reproduces SWRD: Patch-based preprocessing, min defect area 100, only 6 classes, enhancement for better detection.
    # CMD line: python scripts/preprocess_with_tracking.py (old) or python scripts/preprocess_with_tracking_overlap_fix.py (new where some annotations were not included in some images)
    # Output: processed_balanced/ with images/labels/train/val, preprocessing_stats.json.
    # Skip if: Data already preprocessed.
    # Note: Use "preprocess_with_tracking_overlap_fix.py" for the updated version.
    if run_preprocess:
        print("\nStep 2a: Running preprocessing with sliding window...")
        # Use the new fixed script; add ["--split", "train_val_test"] if needed for test set.
        if run_script("preprocess_with_tracking_overlap_fix.py"):  # Or "preprocess_with_tracking.py" for old version.
            print("Preprocessing complete. Check processed_balanced/ folder.")
        else:
            print("Preprocessing failed. Check main_log.txt for details.")

    # Step 2b: Preprocess Verification Visualization step
    # Run scripts/preprocess_with_tracking_viz_verify.py but need to alter the name inside the script of the image for now.
    # This script allows you to visualize all the patches of an original .tif X-ray image. If this is not run following the earlier preprocessing step certain images may be missing annotations.
    # We need to verify the earlier preprocessing script is working. A 2 stage user interface will appear that allows you to flip through all the images and compare with the original long X-ray weld.
    # In stage 2, on the keyboard, 'a' is move to next patch image, 'd' is move to last patch image, 'q' is to exit the user interface.
    # CMD line: python scripts/preprocess_with_tracking_viz_verify.py
    # Output: Visualizations in processed_balanced/visualizations/ with overview.jpg, interactive UI.
    # Skip if: Verification not needed.
    if run_preprocess_verify:
        print("\nStep 2b: Running preprocess verification visualization...")
        if run_script("preprocess_with_tracking_viz_verify.py"):
            print("Verification complete. Check visualizations folder and UI.")
        else:
            print("Verification failed. Check main_log.txt for details.")

    # Step 3a: Balance dataset
    # Run scripts/balance_dataset_with_diversity.py with --source, --output, --strategy, --seed.
    # What it does: Balances classes via undersampling (e.g., porosity to ~5000), augmentation (e.g., undercut to ~2000 via rotate/flip/brightness via OpenCV), keeps others as-is or slightly augmented. Mixed strategy recommended.
    # Note on Augmentation strategy in mixed strategy when augmenting minority classes: Physics preserving augmentations (detailed in README).
    # CMD line: python scripts/balance_dataset_with_diversity.py --source processed_balanced --output processed_balanced_final --strategy mixed --seed 42
    # All arguments: --source processed_balanced: Points to your preprocessing output, --output processed_balanced_final: Where the balanced dataset will go, --strategy mixed: Uses the 75th percentile strategy (good balance between under/oversampling), --seed 42: For reproducibility.
    # Output: processed_balanced_final/ with balanced images/labels, balancing_stats.json.
    # Skip if: No balancing needed (but recommended for SWRD reproduction).
    if run_balance:
        print("\nStep 3a: Running dataset balancing...")
        if run_script("balance_dataset_with_diversity.py", args=["--source", "processed_balanced", "--output", "processed_balanced_final", "--strategy", "mixed", "--seed", "42"]):
            print("Balancing complete. Check processed_balanced_final/ folder.")
        else:
            print("Balancing failed. Check main_log.txt for details.")

    # Step 3b: Check Balanced Dataset
    # Check the actual balance in processed_balanced_final.
    # CMD line: python scripts/balanced_dataset_with_diversity_check_amounts.py (with progress bar).
    # Output: Printed stats on class distribution post-balancing.
    # Skip if: Balance check not needed.
    if run_balance_check:
        print("\nStep 3b: Checking balanced dataset...")
        if run_script("balanced_dataset_with_diversity_check_amounts.py"):
            print("Balance check complete. Check console output.")
        else:
            print("Balance check failed. Check main_log.txt for details.")

    # Step 3c: Check Balanced Dataset (Analyze and Undersample Porosity)
    # Run scripts/analyze_and_undersample_porosity.py.
    # Check to see if the script really made an even split of patches. If the instances in the patches are imbalanced either reweight them in the training of the model or use the script to remove some of the class or classes that there are too many of.
    # CMD line: python scripts/analyze_and_undersample_porosity.py
    # Output: Analyzed stats, optionally undersampled dataset.
    # Skip if: No further adjustment needed.
    if run_analyze_undersample:
        print("\nStep 3c: Analyzing and undersampling porosity...")
        if run_script("analyze_and_undersample_porosity.py"):
            print("Analysis and undersampling complete. Check output.")
        else:
            print("Analysis and undersampling failed. Check main_log.txt for details.")

    # Step 3d: Script to check status and information about GPU
    # Run scripts/check_gpus.py to check on GPU specific information and status.
    # CMD line: python scripts/check_gpus.py
    # Output: Printed GPU info (e.g., memory, utilization).
    # Skip if: GPU check not needed.
    if run_check_gpus:
        print("\nStep 3d: Checking GPU status...")
        if run_script("check_gpus.py"):
            print("GPU check complete. Check console output.")
        else:
            print("GPU check failed. Check main_log.txt for details.")

    # Step 4: Train model - The focus of the project was on working with YoloV8 but YoloV5 is a training option
    # The following scripts are for a YoloV8: Run python scripts/train_models_balanced.py --model --epochs --batch --workers or scripts/train_models_optimized.py with --data, --size, --epochs.
    # A few training scripts to run Yolov8 models. The last one is for running a custom pytorch coded v5 model with downloaded weights.
    # Latest CMD line: python scripts/train_models_balanced.py --model n --epochs 100 --batch 64
    # CMD line: python scripts/train_models_optimized.py --data processed_balanced_final --size n --epochs 40 (Older Version that worked with processed_balanced_final_underrep3classes)
    # In Colab use: train_with_colab.py (although the script is old and needs updates - removal of a few augmentations - it can be easily modified with the same format and tried again)
    # The following script is for a yoloV5. It's a pytorch implemented model (downloaded weights) and was made to run on an NVIDIA RTX 3060 with 12GB VRAM: Run python scripts/train_yolov5_working.py
    # It runs with the following arguments: '--data', default='processed_balanced_final', '--size', default='n', choices=['n', 's', 'm', 'l', 'x'], '--epochs', type=int, default=100, '--lr', type=float, default=0.01, '--batch-size'
    # What it does: Trains YOLO on balanced data using PyTorch. For YOLOv8: loads yolov8.pt, trains on dataset.yaml, uses class weights for imbalance. For YOLOv5: similar, MIT-licensed for companies.
    # Reproduces SWRD: Object detection with bounding boxes, ~100 epochs, medium size for efficiency.
    # Arguments: Vary by script; update as needed.
    # Choose script: Use train_models_balanced.py for latest YOLOv8 with balancing.
    # Output: models/ with weights (best.pt), training_info.json, results.csv/plots.
    # Skip if: Model already trained.
    if run_train:
        print("\nStep 4: Running model training...")
        # Use latest YOLOv8 script; adjust args as needed.
        if run_script("train_models_balanced.py", args=["--model", "n", "--epochs", "100", "--batch", "64"]):
            print("Training complete. Check models/ folder for weights and logs.")
        # Alternative: Older optimized script
        # if run_script("train_models_optimized.py", args=["--data", "processed_balanced_final", "--size", "n", "--epochs", "40"]):
        #     print("Training complete.")
        # Alternative: YOLOv5 script
        # if run_script("train_yolov5_working.py", args=["--data", "processed_balanced_final", "--size", "n", "--epochs", "100", "--lr", "0.01", "--batch-size", "16"]):
        #     print("YOLOv5 training complete.")
        else:
            print("Training failed. Check main_log.txt for details.")

    # Step 5: Evaluate model
    # Run scripts/evaluate_model_enhanced.py with --weights and --split.
    # What it does: Evaluates trained model on splits (train/val/test), computes mAP50/ mAP50-95, precision/recall, per-class AP, generates plots (training curves, per-class bars) via Plotly, saves JSON/HTML.
    # CMD line: python scripts/evaluate_model_enhanced.py --weights "models/yolov8n_20250908_225351/train/weights/best.pt" --split all
    # Output: evaluation_results/ with evaluation_metrics.json, training_plots.html, class_performance.html.
    # Skip if: No evaluation needed.
    if run_evaluate:
        print("\nStep 5: Running model evaluation...")
        # Update --weights with your actual best.pt path (from training output).
        if run_script("evaluate_model_enhanced.py", args=["--weights", "models/yolov8n_20250908_225351/train/weights/best.pt", "--split", "all"]):
            print("Evaluation complete. Check evaluation_results/ for metrics and plots.")
        else:
            print("Evaluation failed. Check main_log.txt for details.")

    # Step 6: Run inference on full images
    # Run scripts/inference_full_image_fixed.py 
    # What it does: This script takes a YoloV8 model and runs inferences on all the patches of a full .tif image and shows bounding boxes around all the weld defects discovered with classification weights. It allows a user to choose different NMS and confidence settings.
    # Arguments: '--model', default=r"C:\AWrk\SWRD_YOLO_Project\models\yolov8n_20250910_025458\train\weights\best.pt", '--image', required=True, '--output', default=None, '--conf', type=float, default=0.25, '--nms', type=float, default=0.45, '--no-viz', action='store_true'
    # CMD line example: python scripts/inference_full_image_fixed.py --image "C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif" --output "C:\AWrk\SWRD_YOLO_Project\inference_results"
    # Output: Printed detections, optional saved images/JSON.
    # Skip if: No inference needed.
    if run_inference:
        print("\nStep 6: Running inference on full images...")
        # Update paths/args as needed; example with one image.
        if run_script("inference_full_image_fixed.py", args=["--image", r"C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif", "--output", r"C:\AWrk\SWRD_YOLO_Project\inference_results"]):
            print("Inference complete. Check console or output folder for results.")
        else:
            print("Inference failed. Check main_log.txt for details.")

    print("\nPipeline complete! Customize flags/args as needed for reruns.")
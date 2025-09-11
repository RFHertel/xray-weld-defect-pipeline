# X-Ray Weld Defect Detection Pipeline Implementation:

# Step 1: Analyze dataset:

Runs scripts/analyze.py with no arguments.
What it does: Loads JSON annotations, counts defects per class (porosity, inclusion, etc.), generates stats (e.g., totals, per-class distribution), saves to dataset_totals.json, and creates visualizations (e.g., bar charts of class imbalance)

CMD line use: python analyze.py

# Step 2: Preprocess with sliding window and tracking

Runs scripts/preprocess_with_tracking.py with no arguments (defaults to train_val split).
What it does: Applies sliding window, filters to 6 valid classes, enhances images (contrast stretch + CLAHE via OpenCV), balances defect/background patches, splits 9:1 train-val (or 8:1:1 with --split train_val_test), saves YOLO-format (images/labels, dataset.yaml).
Reproduces SWRD: Patch-based preprocessing, min defect area 100, only 6 classes, enhancement for better detection.

python scripts/preprocess_with_tracking.py

# Step 3: Balance dataset
Runs scripts/balance_dataset.py with --source and --output.
What it does: Balances classes via undersampling (e.g., porosity to ~5000), augmentation (undercut to ~2000 via rotate/flip/brightness via OpenCV), keeps others as-is or slightly augmented. Mixed strategy recommended.

python scripts/balance_dataset.py --source processed_balanced --output processed_balanced_final --strategy mixed  

# Step 4: Train model

Runs scripts/train_models_optimized.py with --data, --size, --epochs.

or CMD line use: python scripts/train_models_optimized.py --data processed_balanced_final --size n --epochs 40

# Step 5: Evaluate model

Runs scripts/evaluate_model.py with --weights and --split.
What it does: Evaluates trained model on splits (train/val/test), computes mAP50/ mAP50-95, precision/recall, per-class AP, generates plots (training curves, per-class bars) via Plotly, saves JSON/HTML.

CMD line use: python scripts/evaluate_model_enhanced.py --weights "models/yolov8n_20250908_225351/train/weights/best.pt" --split all

# Step 6: Run inference on full images
Runs scripts/inference_full_image.py 

CMD line change directories in args and use: python scripts/inference_full_image_fixed.py --image "C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif" --output "C:\AWrk\SWRD_YOLO_Project\inference_results"

or "python scripts/inference_full_image_fixed.py --image "C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220708-3.tif" --output "C:\AWrk\SWRD_YOLO_Project\inference_results""



# Comprehensive Project Review: SWRD Xray-defect Implementation

Looking at the complete pipeline against the original paper, here's a detailed analysis of what was built and why:

1. Data Analysis Phase (scripts/analyze.py)

Purpose: Understanding the raw dataset structure and class distribution
Key outputs: master_index.json, class distribution statistics
Paper alignment: The paper mentions 6 classes, your analysis confirmed these plus found additional classes (weld_tumor, pseudo_defect) that you correctly excluded

2. Data Preprocessing Pipeline (scripts/preprocess_with_tracking.py)

Patch generation:

Window size: Adaptive (min(h,w)//2, minimum 320px)
Overlap: 50% (matching paper's approach to avoid boundary defects)
Enhancement: CLAHE with clipLimit=2.0, tileGridSize=(8,8)

Critical fix implemented: Image-level splitting BEFORE patching to prevent data leakage
Paper alignment: Matches the sliding window approach, though paper used fixed 320x320

3. Dataset Balancing (scripts/balance_dataset.py)

Strategy: Mixed approach (undersampling majority + augmentation for minority)
Augmentations: Rotation, flips, brightness/contrast, noise, gamma correction
Paper gap: Paper doesn't detail their balancing strategy, but mentions class imbalance as a challenge

4. Model Training (scripts/train_models_optimized.py)

Architecture: YOLOv8n (nano variant for GPU constraints)
Key optimizations:

Disabled memory-heavy augmentations (mosaic, mixup)
Batch size adaptation based on GPU memory
Mixed precision training (AMP)

Training parameters: 50-100 epochs, SGD optimizer, patience=20
Paper alignment: Paper used YOLOv5, you upgraded to v8 for better performance

5. Evaluation Pipeline (scripts/evaluate_model.py)

Metrics tracked: mAP50, mAP50-95, per-class AP, precision, recall
Results: mAP50: 0.552, mAP50-95: 0.249
Paper comparison: Paper reported mAP@0.5 of 0.92, but likely had data leakage

6. Inference System (scripts/inference_full_image_fixed.py)

Sliding window: 320px with 50% overlap
Preprocessing: Exact match to training (grayscale→CLAHE→BGR)
Post-processing: NMS with threshold 0.45
Coordinate mapping: Proper scaling from 640x640 back to original dimensions

Critical Design Decisions & Rationale:
Why Patches Instead of Full Images?

Memory constraints: Full weld images (7904x708) too large for GPU
Detail preservation: Small defects visible at patch level
Data augmentation: More training samples from limited images
Paper validation: Following established methodology

## Why CLAHE Enhancement?

X-ray images have poor contrast
CLAHE provides local contrast enhancement
Consistent preprocessing crucial for model performance

## Why Image-Level Splitting?

Initial error: Patch-level splitting caused data leakage
Fix: Split images first, then generate patches
Impact: More realistic performance metrics

## Key Differences from Paper:

YOLOv8 vs YOLOv5: Better architecture, improved performance
Adaptive window sizing: Paper used fixed 320x320, but we adapt based on image size
Comprehensive balancing: Paper doesn't detail their approach
Proper train/test splitting: Addresses potential data leakage in paper

## Missing Components from Paper:

Weld type classification: Paper mentions longitudinal (L) and transverse (T) welds - not explicitly used in your model
Multi-scale detection: Paper hints at this but not implemented
Real-time performance metrics: FPS/inference time not tracked
Cross-validation: Paper mentions k-fold, you used single split

## Project Strengths:

Robust preprocessing: Handles various image formats gracefully
Memory optimization: Works on limited GPU (RTX 3050 4GB)
Comprehensive tracking: Metadata preserved throughout pipeline
Proper evaluation: No data leakage and realistic metrics. The data leakage fix is a critical methodological improvement
Production-ready inference: Handles full images with sliding window
Document the adaptive window sizing: Innovation over fixed-size approach
Include ablation studies: Effect of CLAHE, overlap percentage, NMS threshold
Report inference time: Still needed here
Compare memory usage: It took up close to 16 GB on RAM and 1-2 GB of NVIDIA GPU during training run

The pipeline is architected for hard constraints (limited GPU, class imbalance) and improves upon the paper's methodology in several key areas, particularly in preventing data leakage and handling class imbalance systematically. 

## Patching: An Innovation in Weld Defect Detection

## Why the patching was initially needed:

The paper introduces the SWRD dataset, which consists of high-resolution X-ray images of seam welds (long, narrow structures like pipelines or pressure vessels). These images are first cropped to focus only on the weld regions (removing irrelevant non-weld areas), resulting in 4,930 images. However, even after this initial cropping, the images still have a high aspect ratio (typically very long horizontally and narrow vertically, resembling strips due to the linear nature of seam welds). To prepare these for deep learning tasks like object detection with YOLOv8, the authors apply a "sliding window cropping" technique to break them into smaller patches (sub-images). This is detailed in Section 3.2 of the paper.

1. The Problem with High Aspect Ratio Images

Image Dimensions and Resizing Issues: The original (and weld-cropped) X-ray images are 16-bit grayscale scans with high resolution but extreme aspect ratios (very elongated, sometimes 5000+ pixels wide but only a few hundred pixels tall). Deep learning models like YOLOv8 typically expect square or near-square inputs, such as 640x640 pixels, for efficient training and inference.

Real-World Analogy from the Paper: The authors draw inspiration from remote sensing imagery (satellite photos), where images are often too large or irregularly shaped to feed directly into neural networks. In those fields, resizing large images leads to similar losses, so patching is a standard workaround.

2. The Logic Behind Sliding Window Cropping (Patching)

Core Technique: They use a "sliding window" to crop the weld images into smaller patches:

Window Size: Set to half the length of the image's shorter side (likely the vertical height, since welds are horizontally long). This creates roughly square or near-square patches that are more suitable for YOLOv8's input requirements.

Sliding Mechanism: The window moves from left to right (and top to bottom if needed) with a 50% overlap between adjacent patches. This overlap ensures continuity—no defects are split or lost at the edges of patches.

3. Key Benefits:

Preserves Critical Information: By cropping into smaller patches without aggressive resizing, high-resolution details are retained. Defects like cracks (jagged lines) or porosities (small dots) remain sharp and identifiable.

Avoids Data Loss at Boundaries: The 50% overlap means every part of the image is covered multiple times. If a defect straddles two patches, it's fully captured in at least one (and partially in others), reducing the risk of missing subtle flaws.

Increases Dataset Size and Diversity: Patching multiplies the number of training samples (from thousands to hundreds of thousands). This provides more data for the model to learn from, improving robustness. It also creates varied views of the same defects (a long crack might appear in multiple overlapping patches from different angles or contexts).

Better Suitability for Deep Learning: Patches are easier to process in models like YOLOv8, which perform best on consistent, moderate-sized inputs. This aligns with how CNNs (convolutional neural networks) extract features—smaller, focused regions allow better localization of defects.

Handles Class Imbalance and Negative Samples: Many patches are defect-free, which are useful as "negative samples" to teach the model what a normal weld looks like, reducing false positives.

Inspiration and Precedent: As noted, this is borrowed from remote sensing [references 20–22 in the paper], where large aerial images are patched to avoid resizing artifacts. In weld inspection, similar challenges arise because X-ray films of seams are inherently linear and high-res (scanned at 1200 dpi, up to 355.6 × 5000 mm).

# Why used Mixed Approach over weighted class loss in loss function

## Comparison: Weighted Loss vs. Mixed Approach

### Weighted Loss Function:

How it Works: Assign higher weights to minority class losses in the classification head, making the model pay more attention to rare classes during backpropagation. In PyTorch, this modifies the cross-entropy (CE) loss by scaling per-class terms (e.g., weight minority higher inversely proportional to frequency).

### Pros for High Imbalance:

Simple and computationally efficient—no extra data generation
Faster training than augmentation (no I/O overhead) 
Effective for moderate imbalances; can be combined with focal loss in YOLO 


### Cons for Really High Imbalance:

If minority samples are too few (our original undercut=564), weights amplify noise/overfit to those samples without adding diversity
Doesn't address data scarcity—model and sees the same minority examples repeatedly
In Object Detection, it may not handle multi-label images well if backgrounds dominate
Benchmarks show it underperforms sampling/augmentation for ratios >20:1 (YOLOv5 study)

When Better: For mild imbalances or when dataset size is fixed (no augmentation possible). In SWRD reproduction, it could boost minority recall without changing data.

### Mixed Approach (Undersampling Majority + Augmentation for Minority):

How it Works: Undersample majority (random drop porosity patches) to reduce dominance; augment minority (rotate/flip undercut patch, synthetic data, etc) to increase count/diversity. We implemented 6x augmentation for undercut.
Pros for Really High Imbalance:

Increases minority data volume and variety, reducing overfitting and improving generalization.
Better for Object Detection as augmentation preserves bounding boxes (with adjustments).
Hybrid data-level fixes root cause (scarcity) vs. just compensating in loss.
Surveys recommend it for extreme cases: augmentation reveals more data distribution and undersampling prevents majority bias.

Cons:

Computationally heavier (augmentation generates data)
Risk of artifacts if augmentation not domain-appropriate (excessive rotation on welds)
Undersampling discards data, potentially losing majority nuances

When Better: For "really high" imbalances like our SWRD datasaet (>>10:1), where minority data is insufficient. 

Which is Better Overall?: For really high imbalances like in SWRD (undercut much smaller than porosity), the mixed approach is better as a standalone method. It directly tackles data scarcity by adding diverse minority examples, which is critical when minority instances are <1k (as in our undercut). Weighted loss alone often insufficient for such extremes.
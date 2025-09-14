#In between corrections:
# scripts/preprocess_with_tracking_overlap_fix.py
# # scripts/preprocess_with_tracking_overlap_fix.py
# import os
# import json
# import cv2
# import numpy as np
# from collections import defaultdict, Counter
# import random
# from pathlib import Path
# import shutil
# from tqdm import tqdm
# import sys
# import traceback
# import logging
# import gc
# import pandas as pd

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('preprocessing_detailed.log'),
#         logging.StreamHandler()
#     ]
# )

# # -------------------------
# # CONFIG
# # -------------------------
# base_dir = Path(r'C:\AWrk\SWRD_YOLO_Project\data')
# output_base = Path(r'C:\AWrk\SWRD_YOLO_Project')

# # Input directories
# img_subdirs = [
#     r'crop_weld_data\crop_weld_images\L\1',
#     r'crop_weld_data\crop_weld_images\L\2',
#     r'crop_weld_data\crop_weld_images\T\1',
#     r'crop_weld_data\crop_weld_images\T\2'
# ]
# json_subdirs = [
#     r'crop_weld_data\crop_weld_jsons\L\1',
#     r'crop_weld_data\crop_weld_jsons\L\2',
#     r'crop_weld_data\crop_weld_jsons\T\1',
#     r'crop_weld_data\crop_weld_jsons\T\2'
# ]

# # Output directories
# processed_dir = output_base / 'processed_balanced'

# # Sliding window parameters (from paper)
# OVERLAP = 0.5  # 50% overlap as stated in paper
# MIN_DEFECT_AREA = 100
# STANDARD_PATCH_SIZE = 640  # Standard YOLO input size from paper

# # Split configurations
# SPLIT_CONFIGS = {
#     'train_val': {'train': 0.9, 'val': 0.1},
#     'train_val_test': {'train': 0.8, 'val': 0.1, 'test': 0.1}
# }

# # The 6 defect classes from the paper
# VALID_CLASSES = {
#     0: 'porosity',
#     1: 'inclusion', 
#     2: 'crack',
#     3: 'undercut',
#     4: 'lack_of_fusion',
#     5: 'lack_of_penetration'
# }

# # Class mapping
# class_map = {
#     '\u6c14\u5b54': 0, '气孔': 0,
#     '\u5939\u6e23': 1, '夹渣': 1,
#     '\u88c2\u7eb9': 2, '裂纹': 2,
#     '\u54ac\u8fb9': 3, '咬边': 3,
#     '\u672a\u878d\u5408': 4, '未熔合': 4,
#     '\u672a\u710a\u900f': 5, '未焊透': 5,
#     '内凹': 3,
#     '夹钨': 1,
# }

# # -------------------------
# # HELPER FUNCTIONS
# # -------------------------
# def validate_dataset():
#     """Check for missing pairs and report issues"""
#     logging.info("Validating dataset...")
    
#     tif_files = {}
#     json_files = {}
    
#     for subdir in img_subdirs:
#         img_dir = base_dir / subdir
#         if img_dir.exists():
#             for tif in img_dir.glob('*.tif'):
#                 tif_files[tif.stem] = tif
    
#     for subdir in json_subdirs:
#         json_dir = base_dir / subdir
#         if json_dir.exists():
#             for json_file in json_dir.glob('*.json'):
#                 json_files[json_file.stem] = json_file
    
#     tif_only = set(tif_files.keys()) - set(json_files.keys())
#     json_only = set(json_files.keys()) - set(tif_files.keys())
#     paired = set(tif_files.keys()) & set(json_files.keys())
    
#     if tif_only:
#         logging.warning(f"Found {len(tif_only)} TIF files without JSON annotations")
#         if not processed_dir.exists():
#             processed_dir.mkdir(parents=True, exist_ok=True)
#         with open(processed_dir / 'missing_jsons.txt', 'w') as f:
#             for name in sorted(tif_only):
#                 f.write(f"{name}.tif\n")
    
#     if json_only:
#         logging.warning(f"Found {len(json_only)} JSON files without TIF images")
#         if not processed_dir.exists():
#             processed_dir.mkdir(parents=True, exist_ok=True)
#         with open(processed_dir / 'missing_tifs.txt', 'w') as f:
#             for name in sorted(json_only):
#                 f.write(f"{name}.json\n")
    
#     logging.info(f"Found {len(paired)} properly paired files")
    
#     pairs = {}
#     for name in paired:
#         pairs[name] = {
#             'tif': tif_files[name],
#             'json': json_files[name]
#         }
    
#     return pairs

# def polygon_to_bbox(points):
#     if not points:
#         return [0, 0, 0, 0]
#     xs = [p[0] for p in points]
#     ys = [p[1] for p in points]
#     x_min, x_max = min(xs), max(xs)
#     y_min, y_max = min(ys), max(ys)
#     return [x_min, y_min, x_max - x_min, y_max - y_min]

# def bbox_to_yolo(bbox, img_width, img_height):
#     x_min, y_min, width, height = bbox
#     x_center = (x_min + width/2) / img_width
#     y_center = (y_min + height/2) / img_height
#     norm_width = width / img_width
#     norm_height = height / img_height
#     return [x_center, y_center, norm_width, norm_height]

# def adjust_annotations_for_patch(annotations, patch_x, patch_y, patch_width, patch_height):
#     """
#     FIXED: Include ANY annotation that overlaps with the patch, not just those with centers inside.
#     Properly clip annotations to patch boundaries.
#     """
#     adjusted_anns = []
    
#     patch_x_max = patch_x + patch_width
#     patch_y_max = patch_y + patch_height
    
#     for ann in annotations:
#         points = ann['points']
#         bbox = polygon_to_bbox(points)
        
#         # Get annotation bounding box coordinates
#         ann_x_min = bbox[0]
#         ann_y_min = bbox[1]
#         ann_x_max = bbox[0] + bbox[2]
#         ann_y_max = bbox[1] + bbox[3]
        
#         # Check if annotation overlaps with patch AT ALL (not just center)
#         if not (ann_x_max < patch_x or ann_x_min > patch_x_max or
#                 ann_y_max < patch_y or ann_y_min > patch_y_max):
            
#             # Clip annotation points to patch boundaries
#             clipped_points = []
#             for x, y in points:
#                 # Translate to patch coordinates and clip
#                 new_x = max(0, min(patch_width-1, x - patch_x))
#                 new_y = max(0, min(patch_height-1, y - patch_y))
#                 clipped_points.append([new_x, new_y])
            
#             # Calculate clipped bbox
#             clipped_bbox = polygon_to_bbox(clipped_points)
            
#             # Only keep if clipped area is significant
#             clipped_area = clipped_bbox[2] * clipped_bbox[3]
#             if clipped_area >= MIN_DEFECT_AREA:
#                 adjusted_anns.append({
#                     'class_id': ann['class_id'],
#                     'points': clipped_points,
#                     'bbox': clipped_bbox
#                 })
    
#     return adjusted_anns

# def enhance_image(image):
#     """Apply CLAHE and preprocessing as described in paper Section 3.3"""
#     try:
#         if len(image.shape) == 3:
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # Contrast stretching (paper mentions this)
#         stretched = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
#         img_8bit = stretched.astype(np.uint8)
        
#         # CLAHE (paper Section 3.3)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(img_8bit)
        
#         # Convert to 3-channel as paper mentions saving as 24-bit
#         final = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
#         return final
#     except Exception as e:
#         logging.error(f"Enhancement failed: {e}")
#         if len(image.shape) == 2:
#             return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#         return image

# def save_yolo_annotation(annotations, output_path, img_width, img_height):
#     with open(output_path, 'w') as f:
#         for ann in annotations:
#             class_id = ann['class_id']
#             bbox = ann['bbox']
#             yolo_bbox = bbox_to_yolo(bbox, img_width, img_height)
#             # Ensure values are within [0, 1]
#             yolo_bbox = [max(0, min(1, v)) for v in yolo_bbox]
#             f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

# def stratified_split_by_source(pairs, master_index_path, split_config='train_val_test'):
#     """
#     Split source images into train/val/test BEFORE creating patches.
#     Uses master_index.json to ensure each split has all defect types.
#     """
#     config = SPLIT_CONFIGS[split_config]
    
#     # Load master index to understand defect distribution
#     with open(master_index_path, 'r') as f:
#         master_index = json.load(f)

#     # Build image-to-defects mapping
#     image_defects = defaultdict(set)
#     for entry in master_index:
#         # Check if the class is one of the valid defects
#         if entry['class'] in VALID_CLASSES.values():
#             image_defects[entry['image_id']].add(entry['class'])

#     # Group images by their defect profile
#     profile_groups = defaultdict(list)
#     for img_id in pairs.keys():
#         defect_profile = tuple(sorted(image_defects.get(img_id, [])))
#         profile_groups[defect_profile].append(img_id)

#     # Initialize splits
#     splits = {name: [] for name in config.keys()}

#     # Distribute each profile group across splits
#     for profile, images in profile_groups.items():
#         random.shuffle(images)
        
#         n_images = len(images)
#         current_idx = 0
        
#         config_items = list(config.items())
#         for i, (split_name, ratio) in enumerate(config_items):
#             if i == len(config_items) - 1:  # Last split gets remainder
#                 splits[split_name].extend(images[current_idx:])
#             else:
#                 n_split = int(round(n_images * ratio))
#                 splits[split_name].extend(images[current_idx : current_idx + n_split])
#                 current_idx += n_split
    
#     # Log distribution
#     logging.info("\nSource image split distribution:")
#     for split_name, image_list in splits.items():
#         defect_counts = Counter()
#         for img_id in image_list:
#             for defect in image_defects.get(img_id, []):
#                 defect_counts[defect] += 1
        
#         logging.info(f"{split_name}: {len(image_list)} images")
#         for defect, count in sorted(defect_counts.items()):
#             logging.info(f"  {defect}: {count} occurrences")
    
#     return splits

# def process_dataset_with_proper_splits(split_config='train_val_test'):
#     """
#     Main processing with fixes:
#     1. Overlap-based annotation inclusion (not center-based)
#     2. Standardized 640x640 patch size
#     3. Proper handling of border patches
#     4. 50% overlap as per paper
#     """
#     logging.info("Starting preprocessing with proper image-level splitting...")
#     logging.info(f"Using {OVERLAP*100}% overlap and {STANDARD_PATCH_SIZE}x{STANDARD_PATCH_SIZE} standardized patches")
    
#     # Create main output directory
#     processed_dir.mkdir(parents=True, exist_ok=True)
    
#     # Check for master_index.json
#     master_index_path = output_base / 'master_index.json'
#     if not master_index_path.exists():
#         logging.error(f"{master_index_path} not found. Please run analyze.py first.")
#         return None
        
#     # Validate dataset
#     pairs = validate_dataset()
#     if not pairs:
#         logging.error("No valid pairs found!")
#         return None

#     # Split images FIRST (before creating patches)
#     image_splits = stratified_split_by_source(pairs, master_index_path, split_config)
    
#     # Process each split separately
#     all_metadata = []
#     split_metadata = {split: [] for split in image_splits.keys()}
    
#     # Track global patch counter for unique filenames
#     global_patch_counter = 0
    
#     for split_name, image_list in image_splits.items():
#         logging.info(f"\nProcessing {split_name} split ({len(image_list)} images)...")

#         # Create final directories directly
#         split_dir = processed_dir / split_name
#         split_dir.mkdir(parents=True, exist_ok=True)
#         (split_dir / 'images').mkdir(exist_ok=True)
#         (split_dir / 'labels').mkdir(exist_ok=True)
        
#         for base_name in tqdm(image_list, desc=f"Processing {split_name}"):
#             if base_name not in pairs:
#                 continue
            
#             files = pairs[base_name]

#             try:
#                 img = cv2.imread(str(files['tif']), cv2.IMREAD_UNCHANGED)
#                 if img is None:
#                     logging.error(f"Cannot load {files['tif']}")
#                     continue
                
#                 if img.shape[0] * img.shape[1] > 20000000:
#                     logging.warning(f"Skipping {base_name} - too large")
#                     continue
                
#                 with open(files['json'], 'r', encoding='utf-8') as f:
#                     ann_data = json.load(f)
                
#                 valid_annotations = []
#                 for shape in ann_data.get('shapes', []):
#                     label = shape.get('label', '')
#                     class_id = class_map.get(label, -1)
                    
#                     if class_id in VALID_CLASSES:
#                         valid_annotations.append({
#                             'class_id': class_id,
#                             'points': shape['points']
#                         })

#                 # Sliding window parameters (from paper Section 3.2)
#                 h, w = img.shape[:2]
#                 window_size = min(h, w) // 2  # Half of shorter side
#                 window_size = max(window_size, 320)  # Minimum size
                
#                 if h < window_size or w < window_size:
#                     window_size = min(h, w)
                
#                 stride = int(window_size * (1 - OVERLAP))  # 50% overlap

#                 # Ensure complete coverage
#                 for y in range(0, h, stride):
#                     for x in range(0, w, stride):
#                         # Handle border patches properly
#                         y_end = min(y + window_size, h)
#                         x_end = min(x + window_size, w)
                        
#                         # Actual patch dimensions (important for border patches)
#                         actual_patch_width = x_end - x
#                         actual_patch_height = y_end - y
                        
#                         # Skip tiny patches
#                         if actual_patch_width < 100 or actual_patch_height < 100:
#                             continue
                        
#                         patch = img[y:y_end, x:x_end].copy()
                        
#                         # CRITICAL FIX: Use actual patch dimensions, include overlapping annotations
#                         patch_anns = adjust_annotations_for_patch(
#                             valid_annotations, x, y, 
#                             actual_patch_width, actual_patch_height
#                         )
                        
#                         # Apply enhancement (CLAHE etc.)
#                         enhanced = enhance_image(patch)
                        
#                         # STANDARDIZE: Resize to 640x640 as per paper
#                         enhanced_resized = cv2.resize(
#                             enhanced, 
#                             (STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE),
#                             interpolation=cv2.INTER_LINEAR
#                         )
                        
#                         # Scale annotations to match resized patch
#                         scale_x = STANDARD_PATCH_SIZE / actual_patch_width
#                         scale_y = STANDARD_PATCH_SIZE / actual_patch_height
                        
#                         scaled_anns = []
#                         for ann in patch_anns:
#                             scaled_bbox = [
#                                 ann['bbox'][0] * scale_x,
#                                 ann['bbox'][1] * scale_y,
#                                 ann['bbox'][2] * scale_x,
#                                 ann['bbox'][3] * scale_y
#                             ]
#                             scaled_anns.append({
#                                 'class_id': ann['class_id'],
#                                 'bbox': scaled_bbox
#                             })
                        
#                         filename = f"{base_name}_{global_patch_counter:06d}"
                        
#                         # Save standardized 640x640 patch
#                         img_path = split_dir / 'images' / f"{filename}.jpg"
#                         cv2.imwrite(str(img_path), enhanced_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
#                         label_path = split_dir / 'labels' / f"{filename}.txt"
                        
#                         # Track metadata
#                         patch_info = {
#                             'filename': filename,
#                             'source_image': base_name,
#                             'split': split_name,
#                             'patch_coords': f"{x},{y},{actual_patch_width},{actual_patch_height}",
#                             'original_patch_size': f"{actual_patch_width}x{actual_patch_height}",
#                             'type': 'background',
#                             'classes': [],
#                             'num_defects': 0
#                         }

#                         if len(scaled_anns) > 0:
#                             # Save YOLO annotations for 640x640 image
#                             save_yolo_annotation(
#                                 scaled_anns, label_path, 
#                                 STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE
#                             )
                            
#                             classes_in_patch = [ann['class_id'] for ann in scaled_anns]
#                             patch_info['type'] = 'defect'
#                             patch_info['classes'] = list(set(classes_in_patch))
#                             patch_info['num_defects'] = len(scaled_anns)
#                         else:
#                             label_path.touch()  # Create empty file for background patches
                        
#                         split_metadata[split_name].append(patch_info)
#                         all_metadata.append(patch_info)
#                         global_patch_counter += 1
                
#                 # Ensure we covered the entire image
#                 # Add patches for right and bottom edges if needed
#                 if (h - 1) % stride != 0:
#                     # Add bottom edge patches
#                     y = h - window_size
#                     if y > 0:
#                         for x in range(0, w, stride):
#                             x_end = min(x + window_size, w)
#                             actual_patch_width = x_end - x
#                             actual_patch_height = window_size
                            
#                             if actual_patch_width < 100 or actual_patch_height < 100:
#                                 continue
                                
#                             patch = img[y:h, x:x_end].copy()
#                             # ... (same processing as above)
                
#                 del img
#                 gc.collect()
            
#             except Exception as e:
#                 logging.error(f"Failed {base_name}: {e}")
#                 traceback.print_exc()

#     # Save metadata
#     df_metadata = pd.DataFrame(all_metadata)
#     df_metadata.to_csv(processed_dir / 'patch_metadata.csv', index=False)
    
#     # Log final statistics per split
#     final_stats = {}
#     for split_name, metadata in split_metadata.items():
#         defect_patches = [p for p in metadata if p['type'] == 'defect']
#         background_patches = [p for p in metadata if p['type'] == 'background']
        
#         class_counts = Counter()
#         for patch in defect_patches:
#             for class_id in patch['classes']:
#                 class_counts[class_id] += 1
                
#         logging.info(f"\n{split_name} split statistics:")
#         logging.info(f"  Total patches: {len(metadata)}")
#         logging.info(f"  Defect patches: {len(defect_patches)}")
#         logging.info(f"  Background patches: {len(background_patches)}")
#         logging.info(f"  All patches are {STANDARD_PATCH_SIZE}x{STANDARD_PATCH_SIZE}")
        
#         for class_id, class_name in VALID_CLASSES.items():
#             count = class_counts.get(class_id, 0)
#             logging.info(f"  {class_name}: {count} patches containing this defect")

#         final_stats[split_name] = {
#             'total_patches': len(metadata),
#             'defect_patches': len(defect_patches),
#             'background_patches': len(background_patches),
#             'class_counts': {VALID_CLASSES[k]: v for k, v in class_counts.items()}
#         }
            
#     # Create dataset.yaml for YOLO
#     yaml_content = f"""path: {str(processed_dir.absolute()).replace(chr(92), '/')}
# train: train/images
# val: val/images
# {'test: test/images' if split_config == 'train_val_test' else ''}

# nc: 6
# names: {list(VALID_CLASSES.values())}

# # Dataset info
# # All images are {STANDARD_PATCH_SIZE}x{STANDARD_PATCH_SIZE} pixels
# # Preprocessing: CLAHE, contrast stretching
# # Overlap: {OVERLAP*100}% sliding window
# """

#     with open(processed_dir / 'dataset.yaml', 'w') as f:
#         f.write(yaml_content)
        
#     logging.info("\nProcessing complete!")
#     logging.info(f"All patches standardized to {STANDARD_PATCH_SIZE}x{STANDARD_PATCH_SIZE}")
#     return {
#         'total_patches': len(all_metadata),
#         'splits': final_stats
#     }


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--split', choices=['train_val', 'train_val_test'], 
#                         default='train_val_test',
#                         help='Split configuration to use')
#     args = parser.parse_args()
    
#     try:
#         # Set random seed for reproducibility
#         random.seed(42)
#         np.random.seed(42)
        
#         stats = process_dataset_with_proper_splits(args.split)
#         if stats:
#             print("\nFinal Statistics:")
#             print(json.dumps(stats, indent=2))
#     except Exception as e:
#         logging.error(f"Fatal error: {e}")
#         traceback.print_exc()
#         sys.exit(1)



# scripts/preprocess_with_tracking_overlap_fix.py - COMPLETE VERSION
import os
import json
import cv2
import numpy as np
from collections import defaultdict, Counter
import random
from pathlib import Path
import shutil
from tqdm import tqdm
import sys
import traceback
import logging
import gc
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing_detailed.log'),
        logging.StreamHandler()
    ]
)

# -------------------------
# CONFIG
# -------------------------
base_dir = Path(r'C:\AWrk\SWRD_YOLO_Project\data')
output_base = Path(r'C:\AWrk\SWRD_YOLO_Project')

# Input directories
img_subdirs = [
    r'crop_weld_data\crop_weld_images\L\1',
    r'crop_weld_data\crop_weld_images\L\2',
    r'crop_weld_data\crop_weld_images\T\1',
    r'crop_weld_data\crop_weld_images\T\2'
]
json_subdirs = [
    r'crop_weld_data\crop_weld_jsons\L\1',
    r'crop_weld_data\crop_weld_jsons\L\2',
    r'crop_weld_data\crop_weld_jsons\T\1',
    r'crop_weld_data\crop_weld_jsons\T\2'
]

# Output directories
processed_dir = output_base / 'processed_balanced'

# Sliding window parameters (from paper)
OVERLAP = 0.5  # 50% overlap as stated in paper Section 3.2
MIN_DEFECT_AREA = 100  # Minimum area for a defect to be included
STANDARD_PATCH_SIZE = 640  # Standard YOLO input size from paper Section 4

# Image quality settings (critical for X-ray images)
JPEG_QUALITY = 100  # Maximum quality for X-ray images
USE_PNG = False  # Set to True for lossless compression (larger files)

# Split configurations
SPLIT_CONFIGS = {
    'train_val': {'train': 0.9, 'val': 0.1},
    'train_val_test': {'train': 0.8, 'val': 0.1, 'test': 0.1}
}

# The 6 defect classes from the paper
VALID_CLASSES = {
    0: 'porosity',
    1: 'inclusion', 
    2: 'crack',
    3: 'undercut',
    4: 'lack_of_fusion',
    5: 'lack_of_penetration'
}

# Class mapping (Chinese to English and class IDs)
class_map = {
    '\u6c14\u5b54': 0, '气孔': 0,          # porosity
    '\u5939\u6e23': 1, '夹渣': 1,          # inclusion
    '\u88c2\u7eb9': 2, '裂纹': 2,          # crack
    '\u54ac\u8fb9': 3, '咬边': 3,          # undercut
    '\u672a\u878d\u5408': 4, '未熔合': 4,    # lack_of_fusion
    '\u672a\u710a\u900f': 5, '未焊透': 5,    # lack_of_penetration
    '内凹': 3,                              # also undercut
    '夹钨': 1,                              # tungsten inclusion
}

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def validate_dataset():
    """Check for missing pairs and report issues"""
    logging.info("Validating dataset...")
    
    tif_files = {}
    json_files = {}
    
    for subdir in img_subdirs:
        img_dir = base_dir / subdir
        if img_dir.exists():
            for tif in img_dir.glob('*.tif'):
                tif_files[tif.stem] = tif
    
    for subdir in json_subdirs:
        json_dir = base_dir / subdir
        if json_dir.exists():
            for json_file in json_dir.glob('*.json'):
                json_files[json_file.stem] = json_file
    
    tif_only = set(tif_files.keys()) - set(json_files.keys())
    json_only = set(json_files.keys()) - set(tif_files.keys())
    paired = set(tif_files.keys()) & set(json_files.keys())
    
    # Create output directory if needed
    if not processed_dir.exists():
        processed_dir.mkdir(parents=True, exist_ok=True)
    
    if tif_only:
        logging.warning(f"Found {len(tif_only)} TIF files without JSON annotations")
        with open(processed_dir / 'missing_jsons.txt', 'w') as f:
            for name in sorted(tif_only):
                f.write(f"{name}.tif\n")
    
    if json_only:
        logging.warning(f"Found {len(json_only)} JSON files without TIF images")
        with open(processed_dir / 'missing_tifs.txt', 'w') as f:
            for name in sorted(json_only):
                f.write(f"{name}.json\n")
    
    logging.info(f"Found {len(paired)} properly paired files")
    
    pairs = {}
    for name in paired:
        pairs[name] = {
            'tif': tif_files[name],
            'json': json_files[name]
        }
    
    return pairs

def polygon_to_bbox(points):
    """Convert polygon points to bounding box [x_min, y_min, width, height]"""
    if not points:
        return [0, 0, 0, 0]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def bbox_to_yolo(bbox, img_width, img_height):
    """Convert bbox to YOLO format (normalized center coordinates)"""
    x_min, y_min, width, height = bbox
    x_center = (x_min + width/2) / img_width
    y_center = (y_min + height/2) / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    return [x_center, y_center, norm_width, norm_height]

def adjust_annotations_for_patch(annotations, patch_x, patch_y, patch_width, patch_height):
    """
    CRITICAL FIX: Include ANY annotation that has ANY spatial overlap with the patch.
    This ensures no defects are missed, especially long ones like lack_of_penetration
    that span multiple patches.
    """
    adjusted_anns = []
    
    patch_x_max = patch_x + patch_width
    patch_y_max = patch_y + patch_height
    
    for ann in annotations:
        points = ann['points']
        bbox = polygon_to_bbox(points)
        
        # Get annotation bounding box coordinates in image space
        ann_x_min = bbox[0]
        ann_y_min = bbox[1]
        ann_x_max = bbox[0] + bbox[2]
        ann_y_max = bbox[1] + bbox[3]
        
        # Check for ANY overlap (even 1 pixel overlap counts)
        # This is the critical fix - we check overlap, not center containment
        has_overlap = not (ann_x_max <= patch_x or      # annotation completely left of patch
                           ann_x_min >= patch_x_max or    # annotation completely right of patch
                           ann_y_max <= patch_y or        # annotation completely above patch
                           ann_y_min >= patch_y_max)      # annotation completely below patch
        
        if has_overlap:
            # Clip polygon points to patch boundaries
            clipped_points = []
            for x, y in points:
                # Translate to patch coordinates
                new_x = x - patch_x
                new_y = y - patch_y
                # Clip to patch boundaries
                new_x = max(0, min(patch_width-1, new_x))
                new_y = max(0, min(patch_height-1, new_y))
                clipped_points.append([new_x, new_y])
            
            # Calculate clipped bbox
            clipped_bbox = polygon_to_bbox(clipped_points)
            
            # Calculate actual intersection area to ensure meaningful overlap
            intersection_x_min = max(0, ann_x_min - patch_x)
            intersection_y_min = max(0, ann_y_min - patch_y)
            intersection_x_max = min(patch_width, ann_x_max - patch_x)
            intersection_y_max = min(patch_height, ann_y_max - patch_y)
            
            intersection_area = max(0, intersection_x_max - intersection_x_min) * \
                              max(0, intersection_y_max - intersection_y_min)
            
            # Include if intersection is meaningful
            if intersection_area >= MIN_DEFECT_AREA:
                adjusted_anns.append({
                    'class_id': ann['class_id'],
                    'points': clipped_points,
                    'bbox': clipped_bbox
                })
                
                # Log for verification (especially for long defects)
                if bbox[2] > patch_width or bbox[3] > patch_height:
                    class_name = VALID_CLASSES.get(ann['class_id'], 'unknown')
                    logging.debug(f"Large {class_name} defect included in patch at ({patch_x}, {patch_y})")
    
    return adjusted_anns

def enhance_image(image):
    """
    Apply preprocessing as described in paper Section 3.3
    Order: Contrast stretching -> 8-bit conversion -> CLAHE -> 3-channel
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Contrast stretching (paper Section 3.3)
        # "improves the overall contrast of the images by spreading out the most frequent intensity values"
        stretched = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Step 2: Convert to 8-bit (paper mentions 16-bit to 8-bit conversion)
        img_8bit = stretched.astype(np.uint8)
        
        # Step 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Paper: "operates on small regions called tiles"
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img_8bit)
        
        # Step 4: Convert to 3-channel (paper mentions saving as 24-bit three-channel images)
        final = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return final
        
    except Exception as e:
        logging.error(f"Enhancement failed: {e}")
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

def save_yolo_annotation(annotations, output_path, img_width, img_height):
    """Save annotations in YOLO format"""
    with open(output_path, 'w') as f:
        for ann in annotations:
            class_id = ann['class_id']
            bbox = ann['bbox']
            yolo_bbox = bbox_to_yolo(bbox, img_width, img_height)
            # Ensure values are within [0, 1]
            yolo_bbox = [max(0, min(1, v)) for v in yolo_bbox]
            f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

def stratified_split_by_source(pairs, master_index_path, split_config='train_val_test'):
    """
    Split source images into train/val/test BEFORE creating patches.
    Ensures each split has representation of all defect types.
    """
    config = SPLIT_CONFIGS[split_config]
    
    # Load master index to understand defect distribution
    with open(master_index_path, 'r') as f:
        master_index = json.load(f)

    # Build image-to-defects mapping
    image_defects = defaultdict(set)
    for entry in master_index:
        if entry['class'] in VALID_CLASSES.values():
            image_defects[entry['image_id']].add(entry['class'])

    # Group images by their defect profile
    profile_groups = defaultdict(list)
    for img_id in pairs.keys():
        defect_profile = tuple(sorted(image_defects.get(img_id, [])))
        profile_groups[defect_profile].append(img_id)

    # Initialize splits
    splits = {name: [] for name in config.keys()}

    # Distribute each profile group across splits
    for profile, images in profile_groups.items():
        random.shuffle(images)
        
        n_images = len(images)
        current_idx = 0
        
        config_items = list(config.items())
        for i, (split_name, ratio) in enumerate(config_items):
            if i == len(config_items) - 1:  # Last split gets remainder
                splits[split_name].extend(images[current_idx:])
            else:
                n_split = int(round(n_images * ratio))
                splits[split_name].extend(images[current_idx : current_idx + n_split])
                current_idx += n_split
    
    # Log distribution
    logging.info("\nSource image split distribution:")
    for split_name, image_list in splits.items():
        defect_counts = Counter()
        for img_id in image_list:
            for defect in image_defects.get(img_id, []):
                defect_counts[defect] += 1
        
        logging.info(f"{split_name}: {len(image_list)} images")
        for defect, count in sorted(defect_counts.items()):
            logging.info(f"  {defect}: {count} occurrences")
    
    return splits

def process_dataset_with_proper_splits(split_config='train_val_test'):
    """
    Main processing function with all fixes:
    1. Overlap-based annotation inclusion (not center-based)
    2. Standardized 640x640 patch size
    3. Proper handling of border patches
    4. 50% overlap sliding window
    5. Maximum quality for X-ray images
    6. Contrast stretching before CLAHE
    """
    logging.info("="*70)
    logging.info("Starting preprocessing with proper image-level splitting")
    logging.info(f"Configuration:")
    logging.info(f"  - Overlap: {OVERLAP*100}%")
    logging.info(f"  - Standardized patch size: {STANDARD_PATCH_SIZE}x{STANDARD_PATCH_SIZE}")
    logging.info(f"  - Image quality: {'PNG (lossless)' if USE_PNG else f'JPEG (quality={JPEG_QUALITY})'}")
    logging.info(f"  - Min defect area: {MIN_DEFECT_AREA} pixels")
    logging.info("="*70)
    
    # Create main output directory
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for master_index.json
    master_index_path = output_base / 'master_index.json'
    if not master_index_path.exists():
        logging.error(f"{master_index_path} not found. Please run analyze.py first.")
        return None
        
    # Validate dataset
    pairs = validate_dataset()
    if not pairs:
        logging.error("No valid pairs found!")
        return None

    # Split images FIRST (before creating patches)
    image_splits = stratified_split_by_source(pairs, master_index_path, split_config)
    
    # Process each split separately
    all_metadata = []
    split_metadata = {split: [] for split in image_splits.keys()}
    
    # Track global patch counter for unique filenames
    global_patch_counter = 0
    
    for split_name, image_list in image_splits.items():
        logging.info(f"\nProcessing {split_name} split ({len(image_list)} images)...")

        # Create final directories directly
        split_dir = processed_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / 'images').mkdir(exist_ok=True)
        (split_dir / 'labels').mkdir(exist_ok=True)
        
        for base_name in tqdm(image_list, desc=f"Processing {split_name}"):
            if base_name not in pairs:
                continue
            
            files = pairs[base_name]

            try:
                # Load image
                img = cv2.imread(str(files['tif']), cv2.IMREAD_UNCHANGED)
                if img is None:
                    logging.error(f"Cannot load {files['tif']}")
                    continue
                
                # Skip very large images
                if img.shape[0] * img.shape[1] > 20000000:
                    logging.warning(f"Skipping {base_name} - too large")
                    continue
                
                # Load annotations
                with open(files['json'], 'r', encoding='utf-8') as f:
                    ann_data = json.load(f)
                
                # Parse valid annotations
                valid_annotations = []
                for shape in ann_data.get('shapes', []):
                    label = shape.get('label', '')
                    class_id = class_map.get(label, -1)
                    
                    if class_id in VALID_CLASSES:
                        valid_annotations.append({
                            'class_id': class_id,
                            'points': shape['points']
                        })

                # Sliding window parameters (from paper Section 3.2)
                h, w = img.shape[:2]
                window_size = min(h, w) // 2  # Half of shorter side
                window_size = max(window_size, 320)  # Minimum size
                
                if h < window_size or w < window_size:
                    window_size = min(h, w)
                
                stride = int(window_size * (1 - OVERLAP))  # 50% overlap

                # Track patch locations for this image
                patch_locations = []
                
                # Generate patches with sliding window
                for y in range(0, h, stride):
                    for x in range(0, w, stride):
                        # Handle border patches properly
                        y_end = min(y + window_size, h)
                        x_end = min(x + window_size, w)
                        
                        # Actual patch dimensions (important for border patches)
                        actual_patch_width = x_end - x
                        actual_patch_height = y_end - y
                        
                        # Skip tiny patches
                        if actual_patch_width < 100 or actual_patch_height < 100:
                            continue
                        
                        patch = img[y:y_end, x:x_end].copy()
                        
                        # CRITICAL: Use overlap-based inclusion with actual patch dimensions
                        patch_anns = adjust_annotations_for_patch(
                            valid_annotations, x, y, 
                            actual_patch_width, actual_patch_height
                        )
                        
                        # Apply enhancement (contrast stretching + CLAHE)
                        enhanced = enhance_image(patch)
                        
                        # Resize to standard size (640x640) as per paper
                        enhanced_resized = cv2.resize(
                            enhanced, 
                            (STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE),
                            interpolation=cv2.INTER_LINEAR
                        )
                        
                        # Scale annotations to match resized patch
                        scale_x = STANDARD_PATCH_SIZE / actual_patch_width
                        scale_y = STANDARD_PATCH_SIZE / actual_patch_height
                        
                        scaled_anns = []
                        for ann in patch_anns:
                            scaled_bbox = [
                                ann['bbox'][0] * scale_x,
                                ann['bbox'][1] * scale_y,
                                ann['bbox'][2] * scale_x,
                                ann['bbox'][3] * scale_y
                            ]
                            scaled_anns.append({
                                'class_id': ann['class_id'],
                                'bbox': scaled_bbox
                            })
                        
                        # Generate filename
                        filename = f"{base_name}_{global_patch_counter:06d}"
                        
                        # Save patch with maximum quality for X-ray images
                        if USE_PNG:
                            img_path = split_dir / 'images' / f"{filename}.png"
                            cv2.imwrite(str(img_path), enhanced_resized, 
                                      [cv2.IMWRITE_PNG_COMPRESSION, 1])
                        else:
                            img_path = split_dir / 'images' / f"{filename}.jpg"
                            cv2.imwrite(str(img_path), enhanced_resized, 
                                      [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                        
                        label_path = split_dir / 'labels' / f"{filename}.txt"
                        
                        # Track metadata
                        patch_info = {
                            'filename': filename,
                            'source_image': base_name,
                            'split': split_name,
                            'patch_coords': f"{x},{y},{actual_patch_width},{actual_patch_height}",
                            'original_patch_size': f"{actual_patch_width}x{actual_patch_height}",
                            'type': 'background',
                            'classes': [],
                            'num_defects': 0
                        }

                        if len(scaled_anns) > 0:
                            # Save YOLO annotations
                            save_yolo_annotation(
                                scaled_anns, label_path, 
                                STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE
                            )
                            
                            classes_in_patch = [ann['class_id'] for ann in scaled_anns]
                            patch_info['type'] = 'defect'
                            patch_info['classes'] = list(set(classes_in_patch))
                            patch_info['num_defects'] = len(scaled_anns)
                        else:
                            # Create empty label file for background patches
                            label_path.touch()
                        
                        split_metadata[split_name].append(patch_info)
                        all_metadata.append(patch_info)
                        patch_locations.append((x, y, x_end, y_end))
                        global_patch_counter += 1
                
                # Verify coverage
                logging.debug(f"Generated {len(patch_locations)} patches for {base_name}")
                
                # Free memory
                del img
                gc.collect()
            
            except Exception as e:
                logging.error(f"Failed processing {base_name}: {e}")
                traceback.print_exc()

    # Save metadata
    df_metadata = pd.DataFrame(all_metadata)
    df_metadata.to_csv(processed_dir / 'patch_metadata.csv', index=False)
    
    # Log final statistics per split
    final_stats = {}
    for split_name, metadata in split_metadata.items():
        defect_patches = [p for p in metadata if p['type'] == 'defect']
        background_patches = [p for p in metadata if p['type'] == 'background']
        
        class_counts = Counter()
        for patch in defect_patches:
            for class_id in patch['classes']:
                class_counts[class_id] += 1
                
        logging.info(f"\n{split_name} split statistics:")
        logging.info(f"  Total patches: {len(metadata)}")
        logging.info(f"  Defect patches: {len(defect_patches)}")
        logging.info(f"  Background patches: {len(background_patches)}")
        logging.info(f"  All patches are {STANDARD_PATCH_SIZE}x{STANDARD_PATCH_SIZE}")
        
        for class_id, class_name in VALID_CLASSES.items():
            count = class_counts.get(class_id, 0)
            logging.info(f"  {class_name}: {count} patches containing this defect")

        final_stats[split_name] = {
            'total_patches': len(metadata),
            'defect_patches': len(defect_patches),
            'background_patches': len(background_patches),
            'class_counts': {VALID_CLASSES[k]: v for k, v in class_counts.items()}
        }
            
    # Create dataset.yaml for YOLO
    yaml_content = f"""# SWRD Dataset Configuration
# Generated from paper: "SWRD: A Dataset of Radiographic Image of Seam Weld for Defect Detection"

path: {str(processed_dir.absolute()).replace(chr(92), '/')}
train: train/images
val: val/images
{'test: test/images' if split_config == 'train_val_test' else ''}

nc: 6
names: {list(VALID_CLASSES.values())}

# Dataset info
# All images are {STANDARD_PATCH_SIZE}x{STANDARD_PATCH_SIZE} pixels
# Preprocessing: Contrast stretching + CLAHE (as per paper Section 3.3)
# Sliding window: {OVERLAP*100}% overlap, window_size = min(h,w)/2
# Image format: {'PNG (lossless)' if USE_PNG else f'JPEG (quality={JPEG_QUALITY})'}
# Annotation method: Overlap-based inclusion (all overlapping defects included)
"""

    with open(processed_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
        
    logging.info("\n" + "="*70)
    logging.info("Processing complete!")
    logging.info(f"Output directory: {processed_dir}")
    logging.info(f"All patches standardized to {STANDARD_PATCH_SIZE}x{STANDARD_PATCH_SIZE}")
    logging.info("="*70)
    
    return {
        'total_patches': len(all_metadata),
        'splits': final_stats
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess SWRD dataset for YOLO training')
    parser.add_argument('--split', choices=['train_val', 'train_val_test'], 
                        default='train_val_test',
                        help='Split configuration to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    try:
        # Set random seed for reproducibility
        random.seed(args.seed)
        np.random.seed(args.seed)
        
        logging.info(f"Random seed set to {args.seed}")
        
        # Run preprocessing
        stats = process_dataset_with_proper_splits(args.split)
        if stats:
            print("\nFinal Statistics:")
            print(json.dumps(stats, indent=2))
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
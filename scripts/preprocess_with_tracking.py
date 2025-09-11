# scripts/preprocess_with_tracking.py - FIXED VERSION
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
# Note: The 'temp' directory is no longer needed with this new logic.
processed_dir = output_base / 'processed_balanced'

# Sliding window parameters
OVERLAP = 0.5
MIN_DEFECT_AREA = 100

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

# Class mapping
class_map = {
    '\u6c14\u5b54': 0, '气孔': 0,
    '\u5939\u6e23': 1, '夹渣': 1,
    '\u88c2\u7eb9': 2, '裂纹': 2,
    '\u54ac\u8fb9': 3, '咬边': 3,
    '\u672a\u878d\u5408': 4, '未熔合': 4,
    '\u672a\u710a\u900f': 5, '未焊透': 5,
    '内凹': 3,
    '夹钨': 1,
}

# -------------------------
# HELPER FUNCTIONS (UNCHANGED)
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
    if not points:
        return [0, 0, 0, 0]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def bbox_to_yolo(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    x_center = (x_min + width/2) / img_width
    y_center = (y_min + height/2) / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    return [x_center, y_center, norm_width, norm_height]

def adjust_annotations_for_patch(annotations, patch_x, patch_y, patch_size):
    adjusted_anns = []
    
    for ann in annotations:
        points = ann['points']
        bbox = polygon_to_bbox(points)
        
        center_x = bbox[0] + bbox[2]/2
        center_y = bbox[1] + bbox[3]/2
        
        if (patch_x <= center_x < patch_x + patch_size and 
            patch_y <= center_y < patch_y + patch_size):
            
            adjusted_points = []
            for x, y in points:
                new_x = max(0, min(patch_size-1, x - patch_x))
                new_y = max(0, min(patch_size-1, y - patch_y))
                adjusted_points.append([new_x, new_y])
            
            adj_bbox = polygon_to_bbox(adjusted_points)
            
            if adj_bbox[2] * adj_bbox[3] >= MIN_DEFECT_AREA:
                adjusted_anns.append({
                    'class_id': ann['class_id'],
                    'points': adjusted_points,
                    'bbox': adj_bbox
                })
    
    return adjusted_anns

def enhance_image(image):
    try:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        stretched = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        img_8bit = stretched.astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img_8bit)
        
        final = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return final
    except Exception as e:
        logging.error(f"Enhancement failed: {e}")
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

def save_yolo_annotation(annotations, output_path, img_width, img_height):
    with open(output_path, 'w') as f:
        for ann in annotations:
            class_id = ann['class_id']
            bbox = ann['bbox']
            yolo_bbox = bbox_to_yolo(bbox, img_width, img_height)
            yolo_bbox = [max(0, min(1, v)) for v in yolo_bbox]
            f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

# -------------------------
# NEW AND UPDATED FUNCTIONS
# -------------------------

### NEW FUNCTION ###
def stratified_split_by_source(pairs, master_index_path, split_config='train_val_test'):
    """
    Split source images into train/val/test BEFORE creating patches.
    Uses master_index.json to ensure each split has all defect types.
    """
    config = SPLIT_CONFIGS[split_config]
    
    # Load master index to understand defect distribution
    with open(master_index_path, 'r') as f:
        master_index = json.load(f)

    # Build image-to-defects mapping
    image_defects = defaultdict(set)
    for entry in master_index:
        # Check if the class is one of the valid defects
        if entry['class'] in VALID_CLASSES.values():
            image_defects[entry['image_id']].add(entry['class'])

    # Group images by their defect profile
    # This ensures diverse defect types in each split
    profile_groups = defaultdict(list)
    for img_id in pairs.keys():
        defect_profile = tuple(sorted(image_defects.get(img_id, [])))
        profile_groups[defect_profile].append(img_id)

    # Initialize splits
    splits = {name: [] for name in config.keys()}

    # Distribute each profile group across splits
    for profile, images in profile_groups.items():
        random.shuffle(images)
        
        # Calculate how many images go to each split
        n_images = len(images)
        current_idx = 0
        
        # Use a copy of the config items to ensure order
        config_items = list(config.items())
        for i, (split_name, ratio) in enumerate(config_items):
            if i == len(config_items) - 1: # Last split gets the remainder
                splits[split_name].extend(images[current_idx:])
            else:
                n_split = int(round(n_images * ratio)) # Use round for better distribution
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


### REPLACED FUNCTION ###
# This single function replaces process_dataset, process_and_save_patches, 
# balance_dataset, create_splits_by_source, and save_splits.
def process_dataset_with_proper_splits(split_config='train_val_test'):
    """
    Main processing with image-level splitting BEFORE patch creation.
    """
    logging.info("Starting preprocessing with proper image-level splitting...")
    
    # Create main output directory
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # First run analyze.py if master_index.json doesn't exist
    # master_index.json should be in the project's root directory.
    master_index_path = output_base / 'master_index.json'
    if not master_index_path.exists():
        logging.error(f"{master_index_path} not found. Please run analyze.py first.")
        return None
        
    # Validate dataset
    pairs = validate_dataset()
    if not pairs:
        logging.error("No valid pairs found!")
        return None

    # ### CORE CHANGE 1: Split images FIRST (before creating patches) ###
    image_splits = stratified_split_by_source(pairs, master_index_path, split_config)
    
    # Process each split separately
    all_metadata = []
    split_metadata = {split: [] for split in image_splits.keys()}
    
    for split_name, image_list in image_splits.items():
        logging.info(f"\nProcessing {split_name} split ({len(image_list)} images)...")

        # ### CORE CHANGE 2: Create final directories directly. No temp folder. ###
        split_dir = processed_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / 'images').mkdir(exist_ok=True)
        (split_dir / 'labels').mkdir(exist_ok=True)
        
        patch_counter = 0 # Counter is now per-split for logging, but filename is global
        
        for base_name in tqdm(image_list, desc=f"Processing {split_name}"):
            if base_name not in pairs:
                continue
            
            files = pairs[base_name]

            try:
                img = cv2.imread(str(files['tif']), cv2.IMREAD_UNCHANGED)
                if img is None:
                    logging.error(f"Cannot load {files['tif']}")
                    continue
                
                if img.shape[0] * img.shape[1] > 20000000:
                    logging.warning(f"Skipping {base_name} - too large")
                    continue
                
                with open(files['json'], 'r', encoding='utf-8') as f:
                    ann_data = json.load(f)
                
                valid_annotations = []
                for shape in ann_data.get('shapes', []):
                    label = shape.get('label', '')
                    class_id = class_map.get(label, -1)
                    
                    if class_id in VALID_CLASSES:
                        valid_annotations.append({
                            'class_id': class_id,
                            'points': shape['points']
                        })

                # ### CORE CHANGE 3: Patching logic is now inside the split loop ###
                h, w = img.shape[:2]
                window_size = min(h, w) // 2
                window_size = max(window_size, 320)
                
                if h < window_size or w < window_size:
                    window_size = min(h, w)
                
                stride = int(window_size * (1 - OVERLAP))

                for y in range(0, max(1, h - window_size + 1), stride):
                    for x in range(0, max(1, w - window_size + 1), stride):
                        y_end = min(y + window_size, h)
                        x_end = min(x + window_size, w)

                        patch = img[y:y_end, x:x_end].copy()
                        
                        if patch.shape[0] < 100 or patch.shape[1] < 100:
                            continue
                        
                        patch_anns = adjust_annotations_for_patch(valid_annotations, x, y, window_size)
                        
                        enhanced = enhance_image(patch)
                        
                        filename = f"{base_name}_{patch_counter:06d}"
                        
                        # Save directly to the correct split directory
                        img_path = split_dir / 'images' / f"{filename}.jpg"
                        cv2.imwrite(str(img_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        label_path = split_dir / 'labels' / f"{filename}.txt"
                        
                        # Track metadata
                        patch_info = {
                            'filename': filename,
                            'source_image': base_name,
                            'split': split_name, # IMPORTANT: Track which split this patch belongs to
                            'patch_coords': f"{x},{y},{window_size}",
                            'type': 'background',
                            'classes': [],
                            'num_defects': 0
                        }

                        if len(patch_anns) > 0:
                            h_patch, w_patch = enhanced.shape[:2]
                            save_yolo_annotation(patch_anns, label_path, w_patch, h_patch)
                            
                            classes_in_patch = [ann['class_id'] for ann in patch_anns]
                            patch_info['type'] = 'defect'
                            patch_info['classes'] = list(set(classes_in_patch))
                            patch_info['num_defects'] = len(patch_anns)
                        else:
                            label_path.touch() # Create empty file for background patches
                        
                        split_metadata[split_name].append(patch_info)
                        all_metadata.append(patch_info)
                        patch_counter += 1
                
                del img
                gc.collect()
            
            except Exception as e:
                logging.error(f"Failed {base_name}: {e}")

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
        for class_id, class_name in VALID_CLASSES.items():
            logging.info(f"  {class_name}: {class_counts.get(class_id, 0)}")

        final_stats[split_name] = {
            'total_patches': len(metadata),
            'defect_patches': len(defect_patches),
            'background_patches': len(background_patches),
            'class_counts': {VALID_CLASSES[k]: v for k, v in class_counts.items()}
        }
            
    # Create dataset.yaml
    yaml_content = f"""path: {str(processed_dir.absolute()).replace(chr(92), '/')}
train: train/images
val: val/images
{'test: test/images' if split_config == 'train_val_test' else ''}

nc: 6
names: {list(VALID_CLASSES.values())}"""

    with open(processed_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
        
    logging.info("\nProcessing complete!")
    return {
        'total_patches': len(all_metadata),
        'splits': final_stats
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train_val', 'train_val_test'], 
                        default='train_val_test',
                        help='Split configuration to use')
    args = parser.parse_args()
    
    try:
        # Call the new main function
        stats = process_dataset_with_proper_splits(args.split)
        if stats:
            print("\nFinal Statistics:")
            print(json.dumps(stats, indent=2))
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
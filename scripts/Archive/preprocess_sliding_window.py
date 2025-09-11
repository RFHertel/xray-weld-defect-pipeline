# scripts/preprocess_sliding_window_v2.py with # Streaming approach - Processes and saves patches immediately instead of keeping them all in memory
import os
import json
import cv2
import numpy as np
from collections import defaultdict
import random
from pathlib import Path
import shutil
from tqdm import tqdm
import sys
import traceback
import logging
import gc  # For garbage collection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
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
processed_dir = output_base / 'processed'
temp_dir = processed_dir / 'temp'
train_dir = processed_dir / 'train'
val_dir = processed_dir / 'val'

# Sliding window parameters
OVERLAP = 0.5
MIN_DEFECT_AREA = 100

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
    
    if tif_only:
        logging.warning(f"Found {len(tif_only)} TIF files without JSON annotations")
        with open('missing_jsons.txt', 'w') as f:
            for name in sorted(tif_only):
                f.write(f"{name}.tif\n")
    
    if json_only:
        logging.warning(f"Found {len(json_only)} JSON files without TIF images")
        with open('missing_tifs.txt', 'w') as f:
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
    """Convert polygon to bbox"""
    if not points:
        return [0, 0, 0, 0]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def bbox_to_yolo(bbox, img_width, img_height):
    """Convert to YOLO format"""
    x_min, y_min, width, height = bbox
    x_center = (x_min + width/2) / img_width
    y_center = (y_min + height/2) / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    return [x_center, y_center, norm_width, norm_height]

def adjust_annotations_for_patch(annotations, patch_x, patch_y, patch_size):
    """Adjust annotations for patch"""
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
    """Enhance image"""
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
    """Save YOLO format annotations"""
    with open(output_path, 'w') as f:
        for ann in annotations:
            class_id = ann['class_id']
            bbox = ann['bbox']
            yolo_bbox = bbox_to_yolo(bbox, img_width, img_height)
            yolo_bbox = [max(0, min(1, v)) for v in yolo_bbox]
            f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

def process_and_save_patches(image, annotations, base_name, patch_counter):
    """Process one image and immediately save patches to disk"""
    h, w = image.shape[:2]
    
    # Calculate window size
    window_size = min(h, w) // 2
    window_size = max(window_size, 320)
    
    # Handle small images
    if h < window_size or w < window_size:
        window_size = min(h, w)
    
    stride = int(window_size * (1 - OVERLAP))
    
    patches_with_defects = []
    patches_background = []
    
    # Generate and save patches one at a time
    for y in range(0, max(1, h - window_size + 1), stride):
        for x in range(0, max(1, w - window_size + 1), stride):
            try:
                # Ensure we don't go out of bounds
                y_end = min(y + window_size, h)
                x_end = min(x + window_size, w)
                
                # Extract patch
                patch = image[y:y_end, x:x_end].copy()
                
                # Skip if patch is too small
                if patch.shape[0] < 100 or patch.shape[1] < 100:
                    continue
                
                # Adjust annotations
                patch_anns = adjust_annotations_for_patch(annotations, x, y, window_size)
                
                # Enhance and save immediately
                enhanced = enhance_image(patch)
                
                # Generate filename
                filename = f"{base_name}_{patch_counter:06d}"
                
                # Save to temp directory
                img_path = temp_dir / 'images' / f"{filename}.jpg"
                cv2.imwrite(str(img_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Save label
                label_path = temp_dir / 'labels' / f"{filename}.txt"
                if len(patch_anns) > 0:
                    h_patch, w_patch = enhanced.shape[:2]
                    save_yolo_annotation(patch_anns, label_path, w_patch, h_patch)
                    patches_with_defects.append(filename)
                else:
                    label_path.touch()  # Empty file for background
                    patches_background.append(filename)
                
                patch_counter += 1
                
                # Clear patch from memory
                del patch
                del enhanced
                
            except Exception as e:
                logging.error(f"Error creating patch at ({x},{y}) for {base_name}: {e}")
                continue
    
    # Force garbage collection after each image
    gc.collect()
    
    return patches_with_defects, patches_background, patch_counter

# -------------------------
# MAIN PROCESSING
# -------------------------
def process_dataset():
    logging.info("Starting preprocessing...")
    
    # Validate and get pairs
    pairs = validate_dataset()
    if not pairs:
        logging.error("No valid image-annotation pairs found!")
        return None
    
    # Create directories
    for dir_path in [processed_dir, temp_dir, train_dir, val_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / 'images').mkdir(exist_ok=True)
        (dir_path / 'labels').mkdir(exist_ok=True)
    
    # Process images one by one
    all_defect_patches = []
    all_background_patches = []
    failed_images = []
    patch_counter = 0
    
    for base_name, files in tqdm(pairs.items(), desc="Processing"):
        try:
            # Load image
            img = cv2.imread(str(files['tif']), cv2.IMREAD_UNCHANGED)
            if img is None:
                logging.error(f"Cannot load {files['tif']}")
                failed_images.append(base_name)
                continue
            
            # Check image size
            if img.shape[0] * img.shape[1] > 20000000:  # Skip very large images
                logging.warning(f"Skipping {base_name} - image too large ({img.shape})")
                failed_images.append(base_name)
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
            
            # Process and save patches immediately
            defect_patches, bg_patches, patch_counter = process_and_save_patches(
                img, valid_annotations, base_name, patch_counter
            )
            
            all_defect_patches.extend(defect_patches)
            all_background_patches.extend(bg_patches)
            
            # Clear image from memory
            del img
            gc.collect()
            
        except Exception as e:
            logging.error(f"Failed processing {base_name}: {e}")
            failed_images.append(base_name)
            continue
    
    # Report
    if failed_images:
        logging.warning(f"Failed to process {len(failed_images)} images")
        with open('failed_images.txt', 'w') as f:
            for name in failed_images:
                f.write(f"{name}\n")
    
    logging.info(f"Total patches with defects: {len(all_defect_patches)}")
    logging.info(f"Total background patches: {len(all_background_patches)}")
    
    # Sample background patches
    n_defect = len(all_defect_patches)
    if len(all_background_patches) > n_defect:
        background_samples = random.sample(all_background_patches, n_defect)
    else:
        background_samples = all_background_patches
    
    # Combine and split
    all_patches = all_defect_patches + background_samples
    random.shuffle(all_patches)
    
    split_idx = int(len(all_patches) * 0.9)
    train_patches = all_patches[:split_idx]
    val_patches = all_patches[split_idx:]
    
    logging.info(f"Moving {len(train_patches)} patches to train...")
    logging.info(f"Moving {len(val_patches)} patches to val...")
    
    # Move files from temp to train/val
    for i, filename in enumerate(tqdm(train_patches, desc="Organizing train")):
        src_img = temp_dir / 'images' / f"{filename}.jpg"
        src_lbl = temp_dir / 'labels' / f"{filename}.txt"
        dst_img = train_dir / 'images' / f"{filename}.jpg"
        dst_lbl = train_dir / 'labels' / f"{filename}.txt"
        
        if src_img.exists():
            shutil.move(str(src_img), str(dst_img))
        if src_lbl.exists():
            shutil.move(str(src_lbl), str(dst_lbl))
    
    for i, filename in enumerate(tqdm(val_patches, desc="Organizing val")):
        src_img = temp_dir / 'images' / f"{filename}.jpg"
        src_lbl = temp_dir / 'labels' / f"{filename}.txt"
        dst_img = val_dir / 'images' / f"{filename}.jpg"
        dst_lbl = val_dir / 'labels' / f"{filename}.txt"
        
        if src_img.exists():
            shutil.move(str(src_img), str(dst_img))
        if src_lbl.exists():
            shutil.move(str(src_lbl), str(dst_lbl))
    
    # Clean up temp directory
    shutil.rmtree(temp_dir)
    
    # Create YAML
    yaml_content = f"""path: {str(processed_dir.absolute()).replace(chr(92), '/')}
train: train/images
val: val/images

nc: 6
names: {list(VALID_CLASSES.values())}"""
    
    with open(processed_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    stats = {
        'paired_images': len(pairs),
        'failed_images': len(failed_images),
        'patches_with_defects': len(all_defect_patches),
        'background_patches_sampled': len(background_samples),
        'train_patches': len(train_patches),
        'val_patches': len(val_patches)
    }
    
    with open(processed_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    logging.info("Processing complete!")
    return stats

if __name__ == "__main__":
    try:
        stats = process_dataset()
        if stats:
            print("\nStatistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
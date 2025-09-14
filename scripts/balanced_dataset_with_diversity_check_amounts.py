# scripts/balanced_dataset_with_diversity_check_amounts.py WITH PROGRESS BAR
import os
from pathlib import Path
from collections import Counter
from tqdm import tqdm

def check_actual_balance():
    """Check the actual balance in processed_balanced_final"""
    
    train_dir = Path(r'C:\AWrk\SWRD_YOLO_Project\processed_balanced_final\train')
    labels_dir = train_dir / 'labels'
    
    # Get list of all label files first
    print("Counting label files...")
    label_files = list(labels_dir.glob('*.txt'))
    total_files = len(label_files)
    print(f"Found {total_files:,} label files to process")
    
    # Count files with annotations vs empty files
    defect_patches = 0
    background_patches = 0
    patches_per_class = Counter()
    
    print("\nProcessing files...")
    for label_file in tqdm(label_files, desc="Analyzing labels"):
        with open(label_file, 'r') as f:
            content = f.read().strip()
            if content:  # Has annotations (defect patch)
                defect_patches += 1
                # Count instances of each class in this patch
                for line in content.split('\n'):
                    if line:
                        try:
                            class_id = int(float(line.split()[0]))
                            patches_per_class[class_id] += 1
                        except (ValueError, IndexError):
                            continue
            else:  # Empty file = background patch
                background_patches += 1
    
    total_patches = defect_patches + background_patches
    
    # Print results
    print(f"\n{'='*60}")
    print("TRAINING SET BALANCE ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total patches: {total_patches:,}")
    print(f"  Defect patches: {defect_patches:,} ({defect_patches/total_patches*100:.1f}%)")
    print(f"  Background patches: {background_patches:,} ({background_patches/total_patches*100:.1f}%)")
    
    if defect_patches > 0:
        ratio = background_patches / defect_patches
        print(f"  Background:Defect ratio: {ratio:.2f}:1")
    
    print(f"\nDEFECT CLASS DISTRIBUTION:")
    class_names = {
        0: 'porosity',
        1: 'inclusion',
        2: 'crack',
        3: 'undercut',
        4: 'lack_of_fusion',
        5: 'lack_of_penetration'
    }
    
    total_class_instances = sum(patches_per_class.values())
    for class_id in sorted(patches_per_class.keys()):
        count = patches_per_class[class_id]
        class_name = class_names.get(class_id, f'Unknown_{class_id}')
        percentage = count / total_class_instances * 100 if total_class_instances > 0 else 0
        print(f"  {class_name:20s}: {count:7,} patches ({percentage:5.1f}%)")
    
    # Warnings
    print(f"\n{'='*60}")
    if background_patches < defect_patches * 1.5:
        print("⚠️  WARNING: Insufficient background patches!")
        print(f"   Current ratio: {ratio:.2f}:1")
        print(f"   Recommended: 2-3:1 for better training")
        needed = int(defect_patches * 2 - background_patches)
        print(f"   Need {needed:,} more background patches for 2:1 ratio")
    else:
        print("✓ Background:Defect ratio looks good!")
    
    print(f"{'='*60}")
    
    return defect_patches, background_patches

if __name__ == "__main__":
    import time
    start = time.time()
    check_actual_balance()
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f} seconds")
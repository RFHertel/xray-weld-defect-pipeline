# scripts/balance_dataset.py
import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import cv2
import logging
from tqdm import tqdm
import albumentations as A

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('balancing.log'),
        logging.StreamHandler()
    ]
)

class DatasetBalancer:
    def __init__(self, source_dir, output_dir, strategy='mixed'):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.strategy = strategy
        
        self.class_names = {
            0: 'porosity',
            1: 'inclusion',
            2: 'crack',
            3: 'undercut',
            4: 'lack_of_fusion',
            5: 'lack_of_penetration'
        }
        
        # Augmentation pipeline for minority classes
        self.augment_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
            A.RandomGamma(gamma_limit=(90, 110), p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
    
    def load_metadata(self):
        """Load and parse the metadata"""
        metadata_path = self.source_dir / 'patch_metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        df = pd.read_csv(metadata_path)
        df['classes'] = df['classes'].apply(lambda x: eval(x) if pd.notna(x) and x != '[]' else [])
        return df
    
    def analyze_distribution(self, df):
        """Analyze class distribution"""
        patches_by_class = defaultdict(list)
        defect_df = df[df['type'] == 'defect']
        
        for _, row in defect_df.iterrows():
            for class_id in row['classes']:
                patches_by_class[class_id].append(row['filename'])
        
        logging.info("\nOriginal distribution:")
        for class_id, class_name in self.class_names.items():
            count = len(patches_by_class.get(class_id, []))
            logging.info(f"  {class_name}: {count}")
        
        return patches_by_class
    
    def balance_strategy(self, patches_by_class):
        """Determine balancing strategy"""
        counts = {k: len(v) for k, v in patches_by_class.items()}
        
        if self.strategy == 'undersample':
            # Match minority class
            target = min(counts.values())
            augment_factors = {k: 1 for k in counts.keys()}
            
        elif self.strategy == 'mixed':
            # Moderate undersampling + augmentation
            median_count = int(np.median(list(counts.values())))
            target = min(median_count, 5000)  # Cap at 5000
            
            # Calculate augmentation factors for classes below target
            augment_factors = {}
            for class_id, count in counts.items():
                if count < target:
                    augment_factors[class_id] = min(int(target / count), 10)  # Max 10x augmentation
                else:
                    augment_factors[class_id] = 1
        
        elif self.strategy == 'oversample':
            # Match majority class through augmentation
            target = max(counts.values())
            augment_factors = {k: int(target / v) for k, v in counts.items()}
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        logging.info(f"\nBalancing strategy: {self.strategy}")
        logging.info(f"Target samples per class: {target}")
        logging.info("Augmentation factors:")
        for class_id, factor in augment_factors.items():
            logging.info(f"  {self.class_names[class_id]}: {factor}x")
        
        return target, augment_factors
    
    def load_yolo_annotations(self, label_path):
        """Load YOLO format annotations"""
        annotations = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        annotations.append([float(x) for x in parts])
        return annotations
    
    def save_yolo_annotations(self, annotations, label_path):
        """Save YOLO format annotations"""
        with open(label_path, 'w') as f:
            for ann in annotations:
                f.write(' '.join(map(str, ann)) + '\n')
    
    def augment_patch(self, img_path, label_path, output_base, aug_index):
        """Augment a single patch"""
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        
        # Load annotations
        annotations = self.load_yolo_annotations(label_path)
        
        if annotations:
            # Prepare for albumentations
            bboxes = []
            class_labels = []
            for ann in annotations:
                class_id = int(ann[0])
                x_center, y_center, width, height = ann[1:]
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(class_id)
            
            # Apply augmentation
            try:
                augmented = self.augment_transform(
                    image=img,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']
                
                # Save augmented image
                aug_img_path = output_base.parent / 'images' / f"{output_base.stem}_aug{aug_index}.jpg"
                cv2.imwrite(str(aug_img_path), aug_img)
                
                # Save augmented annotations
                aug_label_path = output_base.parent / 'labels' / f"{output_base.stem}_aug{aug_index}.txt"
                aug_annotations = []
                for bbox, label in zip(aug_bboxes, aug_labels):
                    aug_annotations.append([label] + list(bbox))
                self.save_yolo_annotations(aug_annotations, aug_label_path)
                
                return True
            except Exception as e:
                logging.error(f"Augmentation failed: {e}")
                return False
        else:
            # No annotations (background), just apply image augmentation
            augmented = self.augment_transform(image=img)
            aug_img = augmented['image']
            aug_img_path = output_base.parent / 'images' / f"{output_base.stem}_aug{aug_index}.jpg"
            cv2.imwrite(str(aug_img_path), aug_img)
            
            # Create empty label file
            aug_label_path = output_base.parent / 'labels' / f"{output_base.stem}_aug{aug_index}.txt"
            aug_label_path.touch()
            
            return True
    
    def process_split(self, split_name, patches_by_class, target, augment_factors, df):
        """Process one split (train/val/test)"""
        source_split = self.source_dir / split_name
        output_split = self.output_dir / split_name
        
        if not source_split.exists():
            logging.warning(f"Split {split_name} not found, skipping")
            return
        
        # Create output directories
        (output_split / 'images').mkdir(parents=True, exist_ok=True)
        (output_split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Get all patches in this split
        split_patches = set()
        for img_path in (source_split / 'images').glob('*.jpg'):
            split_patches.add(img_path.stem)
        
        # Process each class
        split_stats = defaultdict(int)
        
        for class_id, class_patches in patches_by_class.items():
            class_name = self.class_names[class_id]
            
            # Find patches of this class in this split
            class_split_patches = [p for p in class_patches if p in split_patches]
            
            if not class_split_patches:
                continue
            
            logging.info(f"Processing {class_name} in {split_name}: {len(class_split_patches)} original patches")
            
            # Sample if needed
            n_samples = min(len(class_split_patches), int(target * (len(class_split_patches) / len(class_patches))))
            sampled_patches = random.sample(class_split_patches, n_samples) if n_samples < len(class_split_patches) else class_split_patches
            
            # Copy original patches
            for patch_name in tqdm(sampled_patches, desc=f"{class_name} originals"):
                src_img = source_split / 'images' / f"{patch_name}.jpg"
                src_lbl = source_split / 'labels' / f"{patch_name}.txt"
                dst_img = output_split / 'images' / f"{patch_name}.jpg"
                dst_lbl = output_split / 'labels' / f"{patch_name}.txt"
                
                if src_img.exists():
                    shutil.copy2(src_img, dst_img)
                if src_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)
                
                split_stats[class_id] += 1
            
            # Apply augmentation if needed
            aug_factor = augment_factors[class_id]
            if aug_factor > 1:
                n_augment = (aug_factor - 1) * len(sampled_patches)
                logging.info(f"  Augmenting {class_name}: creating {n_augment} augmented samples")
                
                for i in range(n_augment):
                    patch_name = random.choice(sampled_patches)
                    src_img = source_split / 'images' / f"{patch_name}.jpg"
                    src_lbl = source_split / 'labels' / f"{patch_name}.txt"
                    
                    if self.augment_patch(src_img, src_lbl, output_split / patch_name, i):
                        split_stats[class_id] += 1
        
        # Handle background patches
        background_df = df[(df['type'] == 'background')]
        bg_patches = [p for p in background_df['filename'].tolist() if p in split_patches]
        
        # Sample background to match defects
        n_defects = sum(split_stats.values())
        n_bg_target = min(n_defects, len(bg_patches))
        bg_sampled = random.sample(bg_patches, n_bg_target) if n_bg_target < len(bg_patches) else bg_patches
        
        logging.info(f"Copying {len(bg_sampled)} background patches to {split_name}")
        for patch_name in tqdm(bg_sampled, desc="Background"):
            src_img = source_split / 'images' / f"{patch_name}.jpg"
            src_lbl = source_split / 'labels' / f"{patch_name}.txt"
            dst_img = output_split / 'images' / f"{patch_name}.jpg"
            dst_lbl = output_split / 'labels' / f"{patch_name}.txt"
            
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
        
        return split_stats
    
    def run(self):
        """Main balancing pipeline"""
        logging.info(f"Starting dataset balancing with strategy: {self.strategy}")
        
        # Load metadata
        df = self.load_metadata()
        
        # Analyze distribution
        patches_by_class = self.analyze_distribution(df)
        
        # Determine strategy
        target, augment_factors = self.balance_strategy(patches_by_class)
        
        # Process each split
        all_stats = {}
        for split in ['train', 'val', 'test']:
            logging.info(f"\nProcessing {split} split...")
            stats = self.process_split(split, patches_by_class, target, augment_factors, df)
            if stats:
                all_stats[split] = dict(stats)
                logging.info(f"{split} final distribution:")
                for class_id, count in stats.items():
                    logging.info(f"  {self.class_names[class_id]}: {count}")
        
        # Create dataset.yaml
        yaml_content = f"""path: {str(self.output_dir.absolute()).replace(chr(92), '/')}
train: train/images
val: val/images
test: test/images

nc: 6
names: {list(self.class_names.values())}

# Balanced with strategy: {self.strategy}
"""
        with open(self.output_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)
        
        # Save statistics
        stats_output = {
            'strategy': self.strategy,
            'target_per_class': target,
            'augmentation_factors': {self.class_names[k]: v for k, v in augment_factors.items()},
            'splits': all_stats
        }
        
        with open(self.output_dir / 'balancing_stats.json', 'w') as f:
            json.dump(stats_output, f, indent=2)
        
        logging.info(f"\nBalancing complete! Output saved to {self.output_dir}")
        return stats_output

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='processed_balanced', help='Source directory')
    parser.add_argument('--output', default='processed_balanced_final', help='Output directory')
    parser.add_argument('--strategy', choices=['undersample', 'mixed', 'oversample'], 
                       default='mixed', help='Balancing strategy')
    args = parser.parse_args()
    
    balancer = DatasetBalancer(
        source_dir=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.source,
        output_dir=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.output,
        strategy=args.strategy
    )
    
    stats = balancer.run()
    print("\nFinal statistics:")
    print(json.dumps(stats, indent=2))
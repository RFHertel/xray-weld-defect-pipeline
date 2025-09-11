# scripts/balance_dataset.py - CORRECTLY FIXED
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
        
        # Augmentation for TRAINING ONLY
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
        metadata_path = self.source_dir / 'patch_metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        df = pd.read_csv(metadata_path)
        df['classes'] = df['classes'].apply(lambda x: eval(x) if pd.notna(x) and x != '[]' else [])
        return df
    
    def analyze_distribution(self, df):
        """Analyze distribution per split"""
        patches_by_class_split = {}
        
        for split in ['train', 'val', 'test']:
            patches_by_class_split[split] = defaultdict(list)
            split_df = df[(df['split'] == split) & (df['type'] == 'defect')]
            
            for _, row in split_df.iterrows():
                for class_id in row['classes']:
                    patches_by_class_split[split][class_id].append(row['filename'])
        
        logging.info("\nOriginal distribution:")
        total_counts = {'train': 0, 'val': 0, 'test': 0}
        
        for split in ['train', 'val', 'test']:
            logging.info(f"\n{split.upper()}:")
            for class_id, class_name in self.class_names.items():
                count = len(patches_by_class_split[split].get(class_id, []))
                total_counts[split] += count
                logging.info(f"  {class_name}: {count}")
            logging.info(f"  TOTAL: {total_counts[split]}")
        
        return patches_by_class_split
    
    def process_training(self, patches_by_class, df):
        """Balance training set only"""
        source_split = self.source_dir / 'train'
        output_split = self.output_dir / 'train'
        
        (output_split / 'images').mkdir(parents=True, exist_ok=True)
        (output_split / 'labels').mkdir(parents=True, exist_ok=True)
        
        train_df = df[df['split'] == 'train']
        train_patches = set(train_df['filename'].tolist())
        
        # Calculate target for balancing
        counts = {k: len(v) for k, v in patches_by_class.items()}
        
        if self.strategy == 'mixed':
            target = int(np.percentile(list(counts.values()), 75))
            target = max(target, min(counts.values()) * 2)
        elif self.strategy == 'oversample':
            target = max(counts.values())
        else:  # undersample
            target = min(counts.values())
        
        logging.info(f"\nTraining balance target: {target} per class")
        
        split_stats = defaultdict(int)
        
        # Process each class
        for class_id, class_name in self.class_names.items():
            class_patches = patches_by_class.get(class_id, [])
            
            if not class_patches:
                continue
            
            # Undersample if needed
            if len(class_patches) > target:
                sampled = random.sample(class_patches, target)
            else:
                sampled = class_patches
            
            logging.info(f"  {class_name}: {len(sampled)} original")
            
            # Copy originals
            for patch_name in tqdm(sampled, desc=f"{class_name}", leave=False):
                src_img = source_split / 'images' / f"{patch_name}.jpg"
                src_lbl = source_split / 'labels' / f"{patch_name}.txt"
                dst_img = output_split / 'images' / f"{patch_name}.jpg"
                dst_lbl = output_split / 'labels' / f"{patch_name}.txt"
                
                if src_img.exists():
                    shutil.copy2(src_img, dst_img)
                if src_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)
                
                split_stats[class_id] += 1
            
            # Augment if under target
            if len(sampled) < target:
                needed = target - len(sampled)
                logging.info(f"    Creating {needed} augmented samples")
                
                for i in range(needed):
                    patch_name = random.choice(sampled)
                    src_img = source_split / 'images' / f"{patch_name}.jpg"
                    src_lbl = source_split / 'labels' / f"{patch_name}.txt"
                    
                    if self.augment_patch(src_img, src_lbl, output_split / patch_name, i):
                        split_stats[class_id] += 1
        
        # Add background patches
        background_df = train_df[train_df['type'] == 'background']
        bg_patches = background_df['filename'].tolist()
        
        n_defects = sum(split_stats.values())
        n_bg = min(n_defects, len(bg_patches))
        bg_sampled = random.sample(bg_patches, n_bg) if n_bg < len(bg_patches) else bg_patches
        
        logging.info(f"  Background: {len(bg_sampled)} patches")
        for patch_name in tqdm(bg_sampled, desc="Background", leave=False):
            src_img = source_split / 'images' / f"{patch_name}.jpg"
            src_lbl = source_split / 'labels' / f"{patch_name}.txt"
            dst_img = output_split / 'images' / f"{patch_name}.jpg"
            dst_lbl = output_split / 'labels' / f"{patch_name}.txt"
            
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
        
        return split_stats
    
    def process_val_test(self, split_name, df):
        """Copy ALL val/test data without balancing"""
        source_split = self.source_dir / split_name
        output_split = self.output_dir / split_name
        
        (output_split / 'images').mkdir(parents=True, exist_ok=True)
        (output_split / 'labels').mkdir(parents=True, exist_ok=True)
        
        split_df = df[df['split'] == split_name]
        
        # Copy ALL patches
        copied = 0
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Copying {split_name}"):
            patch_name = row['filename']
            src_img = source_split / 'images' / f"{patch_name}.jpg"
            src_lbl = source_split / 'labels' / f"{patch_name}.txt"
            dst_img = output_split / 'images' / f"{patch_name}.jpg"
            dst_lbl = output_split / 'labels' / f"{patch_name}.txt"
            
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                if src_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)
                else:
                    dst_lbl.touch()  # Empty file for background
                copied += 1
        
        # Calculate stats
        split_stats = defaultdict(int)
        defect_df = split_df[split_df['type'] == 'defect']
        
        for _, row in defect_df.iterrows():
            for class_id in row['classes']:
                split_stats[class_id] += 1
        
        background_count = len(split_df[split_df['type'] == 'background'])
        
        logging.info(f"\n{split_name.upper()} (all real data, no balancing):")
        for class_id, count in sorted(split_stats.items()):
            logging.info(f"  {self.class_names[class_id]}: {count}")
        logging.info(f"  Background: {background_count}")
        logging.info(f"  TOTAL: {copied}")
        
        return dict(split_stats)
    
    def augment_patch(self, img_path, label_path, output_base, aug_index):
        """Augment single patch (training only)"""
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        
        annotations = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        annotations.append([float(x) for x in parts])
        
        if annotations:
            bboxes = []
            class_labels = []
            for ann in annotations:
                class_id = int(ann[0])
                x_center, y_center, width, height = ann[1:]
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(class_id)
            
            try:
                augmented = self.augment_transform(
                    image=img,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']
                
                aug_img_path = output_base.parent / 'images' / f"{output_base.stem}_aug{aug_index}.jpg"
                cv2.imwrite(str(aug_img_path), aug_img)
                
                aug_label_path = output_base.parent / 'labels' / f"{output_base.stem}_aug{aug_index}.txt"
                with open(aug_label_path, 'w') as f:
                    for bbox, label in zip(aug_bboxes, aug_labels):
                        f.write(f"{label} {' '.join(map(str, bbox))}\n")
                
                return True
            except:
                return False
        else:
            augmented = self.augment_transform(image=img)
            aug_img_path = output_base.parent / 'images' / f"{output_base.stem}_aug{aug_index}.jpg"
            cv2.imwrite(str(aug_img_path), augmented['image'])
            
            aug_label_path = output_base.parent / 'labels' / f"{output_base.stem}_aug{aug_index}.txt"
            aug_label_path.touch()
            return True
    
    def run(self):
        logging.info("="*60)
        logging.info(f"Dataset Balancing - Strategy: {self.strategy}")
        logging.info("Training: Will be balanced through augmentation")
        logging.info("Val/Test: ALL real data preserved (no balancing)")
        logging.info("="*60)
        
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)
        
        df = self.load_metadata()
        patches_by_class_split = self.analyze_distribution(df)
        
        all_stats = {}
        
        # Process training with balancing
        logging.info("\n" + "="*60)
        logging.info("TRAINING SET PROCESSING")
        train_stats = self.process_training(patches_by_class_split['train'], df)
        all_stats['train'] = train_stats
        
        logging.info("\nTraining final distribution:")
        for class_id, count in sorted(train_stats.items()):
            logging.info(f"  {self.class_names[class_id]}: {count}")
        
        # Process val/test WITHOUT balancing
        for split in ['val', 'test']:
            logging.info("\n" + "="*60)
            logging.info(f"{split.upper()} SET PROCESSING")
            split_stats = self.process_val_test(split, df)
            all_stats[split] = split_stats
        
        # Create dataset.yaml
        yaml_content = f"""path: {str(self.output_dir.absolute()).replace(chr(92), '/')}
train: train/images
val: val/images
test: test/images

nc: 6
names: {list(self.class_names.values())}

# Training: Balanced via {self.strategy} strategy
# Val/Test: Natural distribution (all real data)
"""
        with open(self.output_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)
        
        # Save statistics
        stats_output = {
            'strategy': self.strategy,
            'approach': {
                'train': 'Balanced through augmentation/undersampling',
                'val': 'Natural distribution - all real data',
                'test': 'Natural distribution - all real data'
            },
            'final_distribution': all_stats,
            'class_imbalance_ratios': {}
        }
        
        # Calculate imbalance ratios
        for split in ['val', 'test']:
            if all_stats[split]:
                counts = list(all_stats[split].values())
                if counts and min(counts) > 0:
                    ratio = max(counts) / min(counts)
                    stats_output['class_imbalance_ratios'][split] = f"{ratio:.1f}:1"
        
        with open(self.output_dir / 'balancing_stats.json', 'w') as f:
            json.dump(stats_output, f, indent=2)
        
        logging.info("\n" + "="*60)
        logging.info("COMPLETE!")
        logging.info(f"Output: {self.output_dir}")
        logging.info("Val/Test imbalance preserved for realistic evaluation")
        logging.info("="*60)
        
        return stats_output

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='processed_balanced')
    parser.add_argument('--output', default='processed_balanced_final')
    parser.add_argument('--strategy', choices=['undersample', 'mixed', 'oversample'], default='mixed')
    args = parser.parse_args()
    
    balancer = DatasetBalancer(
        source_dir=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.source,
        output_dir=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.output,
        strategy=args.strategy
    )
    
    stats = balancer.run()
    print("\nFinal statistics:")
    print(json.dumps(stats, indent=2))
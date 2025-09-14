# scripts/balance_dataset_with_diversity.py
# import json
# import shutil
# from pathlib import Path
# import pandas as pd
# import numpy as np
# import random
# from collections import defaultdict
# import cv2
# import logging
# from tqdm import tqdm
# import albumentations as A
# import re

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('balancing_diversity.log'),
#         logging.StreamHandler()
#     ]
# )

# class DiversityAwareBalancer:
#     def __init__(self, source_dir, output_dir, strategy='mixed'):
#         self.source_dir = Path(source_dir)
#         self.output_dir = Path(output_dir)
#         self.strategy = strategy
        
#         self.class_names = {
#             0: 'porosity',
#             1: 'inclusion',
#             2: 'crack',
#             3: 'undercut',
#             4: 'lack_of_fusion',
#             5: 'lack_of_penetration'
#         }
        
#         # Augmentation for TRAINING ONLY
#         self.augment_transform = A.Compose([
#             A.RandomRotate90(p=0.5),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
#             A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
#             A.RandomGamma(gamma_limit=(90, 110), p=0.3),
#             A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
#         ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
    
#     def extract_source_image_name(self, filename):
#         """Extract the original source image name from patch filename"""
#         # Pattern: source_name_XXXXXX (6 digit patch number)
#         # Remove the last _XXXXXX pattern
#         match = re.match(r'(.+)_\d{6}$', filename)
#         if match:
#             return match.group(1)
#         return filename  # Return as-is if pattern doesn't match
    
#     def load_metadata(self):
#         metadata_path = self.source_dir / 'patch_metadata.csv'
#         if not metadata_path.exists():
#             raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
#         df = pd.read_csv(metadata_path)
#         df['classes'] = df['classes'].apply(lambda x: eval(x) if pd.notna(x) and x != '[]' else [])
        
#         # Add source image name column
#         df['source_name'] = df['filename'].apply(self.extract_source_image_name)
        
#         return df
    
#     def analyze_diversity(self, df, split='train'):
#         """Analyze diversity of source images per class"""
#         split_df = df[(df['split'] == split) & (df['type'] == 'defect')]
        
#         diversity_stats = {}
#         for class_id, class_name in self.class_names.items():
#             # Find all patches containing this class
#             class_patches = []
#             source_images = defaultdict(list)
            
#             for _, row in split_df.iterrows():
#                 if class_id in row['classes']:
#                     class_patches.append(row['filename'])
#                     source_images[row['source_name']].append(row['filename'])
            
#             diversity_stats[class_name] = {
#                 'total_patches': len(class_patches),
#                 'unique_sources': len(source_images),
#                 'source_distribution': {k: len(v) for k, v in source_images.items()}
#             }
        
#         logging.info(f"\nDiversity analysis for {split}:")
#         for class_name, stats in diversity_stats.items():
#             logging.info(f"  {class_name}:")
#             logging.info(f"    Total patches: {stats['total_patches']}")
#             logging.info(f"    Unique source images: {stats['unique_sources']}")
#             if stats['total_patches'] > 0:
#                 avg_per_source = stats['total_patches'] / max(1, stats['unique_sources'])
#                 logging.info(f"    Average patches per source: {avg_per_source:.1f}")
        
#         return diversity_stats
    
#     def organize_patches_by_source(self, df, split='train'):
#         """Organize patches by class and source image"""
#         split_df = df[df['split'] == split]
        
#         # Structure: {class_id: {source_name: [patch_names]}}
#         patches_by_class_source = defaultdict(lambda: defaultdict(list))
        
#         # Defect patches
#         defect_df = split_df[split_df['type'] == 'defect']
#         for _, row in defect_df.iterrows():
#             for class_id in row['classes']:
#                 patches_by_class_source[class_id][row['source_name']].append(row['filename'])
        
#         # Background patches organized by source
#         background_by_source = defaultdict(list)
#         bg_df = split_df[split_df['type'] == 'background']
#         for _, row in bg_df.iterrows():
#             background_by_source[row['source_name']].append(row['filename'])
        
#         return patches_by_class_source, background_by_source
    
#     def diversity_aware_sampling(self, patches_by_source, target_count):
#         """
#         Sample patches ensuring diversity across source images.
        
#         Strategy:
#         1. First, ensure we sample from as many source images as possible
#         2. Within each source, randomly sample patches
#         3. If we need more samples, go back and sample more from each source
#         """
#         all_sources = list(patches_by_source.keys())
#         if not all_sources:
#             return []
        
#         random.shuffle(all_sources)
#         sampled_patches = []
        
#         # First pass: sample at least one from each source
#         patches_per_source_initial = max(1, target_count // len(all_sources))
        
#         for source in all_sources:
#             source_patches = patches_by_source[source]
#             n_to_sample = min(len(source_patches), patches_per_source_initial)
#             sampled = random.sample(source_patches, n_to_sample)
#             sampled_patches.extend(sampled)
            
#             if len(sampled_patches) >= target_count:
#                 break
        
#         # Second pass: if we need more, sample additional patches
#         if len(sampled_patches) < target_count:
#             remaining_needed = target_count - len(sampled_patches)
            
#             # Collect all remaining patches not yet sampled
#             remaining_patches = []
#             for source, patches in patches_by_source.items():
#                 already_sampled = set(p for p in sampled_patches if p in patches)
#                 remaining = [p for p in patches if p not in already_sampled]
#                 remaining_patches.extend(remaining)
            
#             if remaining_patches:
#                 additional = min(remaining_needed, len(remaining_patches))
#                 sampled_patches.extend(random.sample(remaining_patches, additional))
        
#         # Trim if we oversampled
#         if len(sampled_patches) > target_count:
#             sampled_patches = random.sample(sampled_patches, target_count)
        
#         return sampled_patches
    
#     def process_training_with_diversity(self, patches_by_class_source, background_by_source, df):
#         """Balance training set with diversity awareness"""
#         source_split = self.source_dir / 'train'
#         output_split = self.output_dir / 'train'
        
#         (output_split / 'images').mkdir(parents=True, exist_ok=True)
#         (output_split / 'labels').mkdir(parents=True, exist_ok=True)
        
#         # Calculate target for balancing
#         counts = {k: sum(len(patches) for patches in v.values()) 
#                  for k, v in patches_by_class_source.items()}
        
#         if self.strategy == 'mixed':
#             target = int(np.percentile(list(counts.values()), 75))
#             target = max(target, min(counts.values()) * 2)
#             # Cap at a reasonable number to avoid too much augmentation
#             target = min(target, 15000)  
#         elif self.strategy == 'oversample':
#             target = max(counts.values())
#         else:  # undersample
#             target = min(counts.values())
        
#         logging.info(f"\nTraining balance target: {target} per class")
#         logging.info("Ensuring diversity across source images...")
        
#         split_stats = defaultdict(int)
#         source_usage_stats = defaultdict(lambda: defaultdict(int))
        
#         # Process each class with diversity
#         for class_id, class_name in self.class_names.items():
#             patches_by_source = patches_by_class_source.get(class_id, {})
            
#             if not patches_by_source:
#                 logging.warning(f"  No patches found for {class_name}")
#                 continue
            
#             # Diversity-aware sampling
#             sampled = self.diversity_aware_sampling(patches_by_source, min(target, counts[class_id]))
            
#             # Track source usage
#             for patch in sampled:
#                 source = self.extract_source_image_name(patch)
#                 source_usage_stats[class_name][source] += 1
            
#             logging.info(f"  {class_name}:")
#             logging.info(f"    Sampled {len(sampled)} patches from {len(patches_by_source)} source images")
            
#             # Copy originals
#             for patch_name in tqdm(sampled, desc=f"Copying {class_name}", leave=False):
#                 src_img = source_split / 'images' / f"{patch_name}.jpg"
#                 src_lbl = source_split / 'labels' / f"{patch_name}.txt"
#                 dst_img = output_split / 'images' / f"{patch_name}.jpg"
#                 dst_lbl = output_split / 'labels' / f"{patch_name}.txt"
                
#                 if src_img.exists():
#                     shutil.copy2(src_img, dst_img)
#                 if src_lbl.exists():
#                     shutil.copy2(src_lbl, dst_lbl)
                
#                 split_stats[class_id] += 1
            
#             # Augment if under target
#             if len(sampled) < target:
#                 needed = target - len(sampled)
#                 logging.info(f"    Creating {needed} augmented samples")
                
#                 # Select patches to augment with diversity
#                 aug_sources = random.choices(sampled, k=needed)
                
#                 for i, patch_name in enumerate(tqdm(aug_sources, desc=f"Augmenting {class_name}", leave=False)):
#                     src_img = source_split / 'images' / f"{patch_name}.jpg"
#                     src_lbl = source_split / 'labels' / f"{patch_name}.txt"
                    
#                     if self.augment_patch(src_img, src_lbl, output_split / patch_name, i):
#                         split_stats[class_id] += 1
        
#         # Add background patches with diversity
#         logging.info("\n  Processing background patches with diversity...")
#         n_defects = sum(split_stats.values())
        
#         # Sample background with diversity
#         bg_target = min(n_defects, sum(len(patches) for patches in background_by_source.values()))
#         bg_sampled = self.diversity_aware_sampling(background_by_source, bg_target)
        
#         logging.info(f"  Background: {len(bg_sampled)} patches from {len(background_by_source)} source images")
        
#         for patch_name in tqdm(bg_sampled, desc="Copying background", leave=False):
#             src_img = source_split / 'images' / f"{patch_name}.jpg"
#             src_lbl = source_split / 'labels' / f"{patch_name}.txt"
#             dst_img = output_split / 'images' / f"{patch_name}.jpg"
#             dst_lbl = output_split / 'labels' / f"{patch_name}.txt"
            
#             if src_img.exists():
#                 shutil.copy2(src_img, dst_img)
#             if src_lbl.exists():
#                 shutil.copy2(src_lbl, dst_lbl)
#             else:
#                 dst_lbl.touch()
        
#         # Log source diversity statistics
#         logging.info("\nSource diversity in final training set:")
#         for class_name, sources in source_usage_stats.items():
#             logging.info(f"  {class_name}: {len(sources)} unique source images used")
        
#         return split_stats, source_usage_stats
    
#     def process_val_test(self, split_name, df):
#         """Copy ALL val/test data without balancing - preserves natural distribution"""
#         source_split = self.source_dir / split_name
#         output_split = self.output_dir / split_name
        
#         (output_split / 'images').mkdir(parents=True, exist_ok=True)
#         (output_split / 'labels').mkdir(parents=True, exist_ok=True)
        
#         split_df = df[df['split'] == split_name]
        
#         # Copy ALL patches - no balancing for val/test
#         copied = 0
#         for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Copying {split_name}"):
#             patch_name = row['filename']
#             src_img = source_split / 'images' / f"{patch_name}.jpg"
#             src_lbl = source_split / 'labels' / f"{patch_name}.txt"
#             dst_img = output_split / 'images' / f"{patch_name}.jpg"
#             dst_lbl = output_split / 'labels' / f"{patch_name}.txt"
            
#             if src_img.exists():
#                 shutil.copy2(src_img, dst_img)
#                 if src_lbl.exists():
#                     shutil.copy2(src_lbl, dst_lbl)
#                 else:
#                     dst_lbl.touch()
#                 copied += 1
        
#         # Calculate stats
#         split_stats = defaultdict(int)
#         defect_df = split_df[split_df['type'] == 'defect']
        
#         for _, row in defect_df.iterrows():
#             for class_id in row['classes']:
#                 split_stats[class_id] += 1
        
#         background_count = len(split_df[split_df['type'] == 'background'])
        
#         logging.info(f"\n{split_name.upper()} (all real data, no balancing):")
#         for class_id, count in sorted(split_stats.items()):
#             logging.info(f"  {self.class_names[class_id]}: {count}")
#         logging.info(f"  Background: {background_count}")
#         logging.info(f"  TOTAL: {copied}")
        
#         return dict(split_stats)
    
#     def augment_patch(self, img_path, label_path, output_base, aug_index):
#         """Augment single patch (training only)"""
#         img = cv2.imread(str(img_path))
#         if img is None:
#             return False
        
#         annotations = []
#         if label_path.exists():
#             with open(label_path, 'r') as f:
#                 for line in f:
#                     parts = line.strip().split()
#                     if len(parts) == 5:
#                         annotations.append([float(x) for x in parts])
        
#         if annotations:
#             bboxes = []
#             class_labels = []
#             for ann in annotations:
#                 class_id = int(ann[0])
#                 x_center, y_center, width, height = ann[1:]
#                 bboxes.append([x_center, y_center, width, height])
#                 class_labels.append(class_id)
            
#             try:
#                 augmented = self.augment_transform(
#                     image=img,
#                     bboxes=bboxes,
#                     class_labels=class_labels
#                 )
                
#                 aug_img = augmented['image']
#                 aug_bboxes = augmented['bboxes']
#                 aug_labels = augmented['class_labels']
                
#                 aug_img_path = output_base.parent / 'images' / f"{output_base.stem}_aug{aug_index}.jpg"
#                 cv2.imwrite(str(aug_img_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                
#                 aug_label_path = output_base.parent / 'labels' / f"{output_base.stem}_aug{aug_index}.txt"
#                 with open(aug_label_path, 'w') as f:
#                     for bbox, label in zip(aug_bboxes, aug_labels):
#                         f.write(f"{label} {' '.join(map(str, bbox))}\n")
                
#                 return True
#             except:
#                 return False
#         else:
#             # Background augmentation
#             augmented = self.augment_transform(image=img)
#             aug_img_path = output_base.parent / 'images' / f"{output_base.stem}_aug{aug_index}.jpg"
#             cv2.imwrite(str(aug_img_path), augmented['image'], [cv2.IMWRITE_JPEG_QUALITY, 100])
            
#             aug_label_path = output_base.parent / 'labels' / f"{output_base.stem}_aug{aug_index}.txt"
#             aug_label_path.touch()
#             return True
    
#     def run(self):
#         logging.info("="*70)
#         logging.info(f"Diversity-Aware Dataset Balancing")
#         logging.info(f"Strategy: {self.strategy}")
#         logging.info("Training: Balanced with source image diversity")
#         logging.info("Val/Test: Natural distribution preserved")
#         logging.info("="*70)
        
#         if self.output_dir.exists():
#             shutil.rmtree(self.output_dir)
#         self.output_dir.mkdir(parents=True)
        
#         # Load and analyze
#         df = self.load_metadata()
        
#         # Analyze diversity
#         train_diversity = self.analyze_diversity(df, 'train')
        
#         # Organize patches
#         patches_by_class_source, background_by_source = self.organize_patches_by_source(df, 'train')
        
#         all_stats = {}
        
#         # Process training with diversity awareness
#         logging.info("\n" + "="*70)
#         logging.info("TRAINING SET PROCESSING (with diversity)")
#         train_stats, source_usage = self.process_training_with_diversity(
#             patches_by_class_source, background_by_source, df
#         )
#         all_stats['train'] = train_stats
        
#         # Process val/test WITHOUT balancing (preserve natural distribution)
#         for split in ['val', 'test']:
#             if len(df[df['split'] == split]) > 0:
#                 logging.info("\n" + "="*70)
#                 logging.info(f"{split.upper()} SET PROCESSING (no balancing)")
#                 split_stats = self.process_val_test(split, df)
#                 all_stats[split] = split_stats
        
#         # Create dataset.yaml
#         yaml_content = f"""path: {str(self.output_dir.absolute()).replace(chr(92), '/')}
# train: train/images
# val: val/images
# test: test/images

# nc: 6
# names: {list(self.class_names.values())}

# # Training: Balanced with diversity across source images
# # Val/Test: Natural distribution (all real data)
# # Strategy: {self.strategy}
# """
#         with open(self.output_dir / 'dataset.yaml', 'w') as f:
#             f.write(yaml_content)
        
#         # Save detailed statistics
#         stats_output = {
#             'strategy': self.strategy,
#             'diversity_aware': True,
#             'training_diversity': train_diversity,
#             'final_distribution': all_stats,
#             'source_usage': source_usage
#         }
        
#         with open(self.output_dir / 'diversity_stats.json', 'w') as f:
#             json.dump(stats_output, f, indent=2, default=str)
        
#         logging.info("\n" + "="*70)
#         logging.info("COMPLETE!")
#         logging.info(f"Output: {self.output_dir}")
#         logging.info("Training set balanced with source image diversity")
#         logging.info("Val/Test preserved with natural distribution")
#         logging.info("="*70)
        
#         return stats_output

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--source', default='processed_balanced')
#     parser.add_argument('--output', default='processed_balanced_final')
#     parser.add_argument('--strategy', choices=['undersample', 'mixed', 'oversample'], 
#                        default='mixed', help='Balancing strategy')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed')
#     args = parser.parse_args()
    
#     # Set seed for reproducibility
#     random.seed(args.seed)
#     np.random.seed(args.seed)
    
#     balancer = DiversityAwareBalancer(
#         source_dir=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.source,
#         output_dir=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.output,
#         strategy=args.strategy
#     )
    
#     stats = balancer.run()



# scripts/balance_dataset_with_diversity.py - UPDATED WITH REALISTIC AUGMENTATIONS - error on script
# import json
# import shutil
# from pathlib import Path
# import pandas as pd
# import numpy as np
# import random
# from collections import defaultdict
# import cv2
# import logging
# from tqdm import tqdm
# import albumentations as A
# import re

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('balancing_diversity.log'),
#         logging.StreamHandler()
#     ]
# )

# class DiversityAwareBalancer:
#     def __init__(self, source_dir, output_dir, strategy='mixed'):
#         self.source_dir = Path(source_dir)
#         self.output_dir = Path(output_dir)
#         self.strategy = strategy
        
#         self.class_names = {
#             0: 'porosity',
#             1: 'inclusion',
#             2: 'crack',
#             3: 'undercut',
#             4: 'lack_of_fusion',
#             5: 'lack_of_penetration'
#         }
        
#         # Create class-specific augmentations that respect physical properties
#         self.class_augmentations = self._create_class_specific_augmentations()
    
#     def _create_class_specific_augmentations(self):
#         """
#         Create physically realistic augmentations for each defect type.
#         Each defect has unique physical properties that must be preserved.
#         """
        
#         # Base intensity augmentations (safe for all X-ray images)
#         base_intensity = [
#             A.RandomBrightnessContrast(
#                 brightness_limit=0.1,  # Simulates exposure variations
#                 contrast_limit=0.1,    # Simulates penetration differences
#                 p=0.4
#             ),
#             A.RandomGamma(
#                 gamma_limit=(90, 110),  # Simulates film/detector response
#                 p=0.3
#             ),
#             A.GaussNoise(
#                 var_limit=(5.0, 20.0),  # Low noise - simulates detector noise
#                 p=0.3
#             ),
#         ]
        
#         augmentations = {}
        
#         # POROSITY (gas bubbles - can appear at any orientation)
#         # Physical property: Round/spherical defects, orientation-independent
#         augmentations[0] = A.Compose([
#             A.RandomRotate90(p=0.5),  # Bubbles are round, rotation OK
#             A.Flip(p=0.5),  # Can flip freely
#             *base_intensity,
#             # NO shape distortion - would make bubbles non-circular
#         ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
#         # INCLUSION (solid particles trapped in weld)
#         # Physical property: Irregular shapes but maintain form
#         augmentations[1] = A.Compose([
#             A.RandomRotate90(p=0.4),  # Can be at various angles
#             A.Flip(p=0.4),
#             *base_intensity,
#             A.ShiftScaleRotate(
#                 shift_limit=0.02,  # Very small shift
#                 scale_limit=0.0,   # NO scaling - preserve size
#                 rotate_limit=10,   # Small rotation
#                 p=0.3
#             ),
#         ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
#         # CRACK (linear defects with sharp edges)
#         # Physical property: Linear, follows stress patterns, sharp edges crucial
#         augmentations[2] = A.Compose([
#             A.HorizontalFlip(p=0.3),  # OK along crack direction
#             A.VerticalFlip(p=0.2),    # Less common
#             *base_intensity,
#             # NO rotation - cracks follow specific stress patterns
#             # NO blur - would lose critical sharp edges
#             # NO geometric distortion
#         ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
#         # UNDERCUT (groove along weld toe)
#         # Physical property: Position-specific at weld edges
#         augmentations[3] = A.Compose([
#             A.HorizontalFlip(p=0.3),  # Can appear on either side
#             *base_intensity,
#             A.ShiftScaleRotate(
#                 shift_limit=0.01,  # Tiny shift only
#                 scale_limit=0.0,   # NO scaling
#                 rotate_limit=5,    # Very small rotation
#                 p=0.2
#             ),
#             # More aggressive noise since we have few samples
#             A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=0.3),
#         ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
#         # LACK OF FUSION (incomplete fusion at weld boundaries)
#         # Physical property: Linear along weld interfaces, position-critical
#         augmentations[4] = A.Compose([
#             A.HorizontalFlip(p=0.3),  # Can occur on either side
#             *base_intensity,
#             # NO rotation - follows weld geometry
#             # NO vertical flip - gravity affects formation
#             # NO geometric distortion
#         ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
#         # LACK OF PENETRATION (incomplete root penetration)
#         # Physical property: Always at weld centerline, straight lines
#         augmentations[5] = A.Compose([
#             A.HorizontalFlip(p=0.5),  # Can extend in either direction
#             *base_intensity,
#             # NO rotation - always aligned with weld
#             # NO vertical flip - position-specific at root
#             # NO geometric changes - very specific appearance
#         ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
#         return augmentations
    
#     def extract_source_image_name(self, filename):
#         """Extract the original source image name from patch filename"""
#         match = re.match(r'(.+)_\d{6}$', filename)
#         if match:
#             return match.group(1)
#         return filename
    
#     def load_metadata(self):
#         metadata_path = self.source_dir / 'patch_metadata.csv'
#         if not metadata_path.exists():
#             raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
#         df = pd.read_csv(metadata_path)
#         df['classes'] = df['classes'].apply(lambda x: eval(x) if pd.notna(x) and x != '[]' else [])
#         df['source_name'] = df['filename'].apply(self.extract_source_image_name)
        
#         return df
    
#     def analyze_diversity(self, df, split='train'):
#         """Analyze diversity of source images per class"""
#         split_df = df[(df['split'] == split) & (df['type'] == 'defect')]
        
#         diversity_stats = {}
#         for class_id, class_name in self.class_names.items():
#             class_patches = []
#             source_images = defaultdict(list)
            
#             for _, row in split_df.iterrows():
#                 if class_id in row['classes']:
#                     class_patches.append(row['filename'])
#                     source_images[row['source_name']].append(row['filename'])
            
#             diversity_stats[class_name] = {
#                 'total_patches': len(class_patches),
#                 'unique_sources': len(source_images),
#                 'source_distribution': {k: len(v) for k, v in source_images.items()}
#             }
        
#         logging.info(f"\nDiversity analysis for {split}:")
#         for class_name, stats in diversity_stats.items():
#             logging.info(f"  {class_name}:")
#             logging.info(f"    Total patches: {stats['total_patches']}")
#             logging.info(f"    Unique source images: {stats['unique_sources']}")
#             if stats['total_patches'] > 0:
#                 avg_per_source = stats['total_patches'] / max(1, stats['unique_sources'])
#                 logging.info(f"    Average patches per source: {avg_per_source:.1f}")
        
#         return diversity_stats
    
#     def organize_patches_by_source(self, df, split='train'):
#         """Organize patches by class and source image"""
#         split_df = df[df['split'] == split]
        
#         patches_by_class_source = defaultdict(lambda: defaultdict(list))
        
#         defect_df = split_df[split_df['type'] == 'defect']
#         for _, row in defect_df.iterrows():
#             for class_id in row['classes']:
#                 patches_by_class_source[class_id][row['source_name']].append(row['filename'])
        
#         background_by_source = defaultdict(list)
#         bg_df = split_df[split_df['type'] == 'background']
#         for _, row in bg_df.iterrows():
#             background_by_source[row['source_name']].append(row['filename'])
        
#         return patches_by_class_source, background_by_source
    
#     def diversity_aware_sampling(self, patches_by_source, target_count):
#         """Sample patches ensuring diversity across source images"""
#         all_sources = list(patches_by_source.keys())
#         if not all_sources:
#             return []
        
#         random.shuffle(all_sources)
#         sampled_patches = []
        
#         # First pass: sample from each source
#         patches_per_source_initial = max(1, target_count // len(all_sources))
        
#         for source in all_sources:
#             source_patches = patches_by_source[source]
#             n_to_sample = min(len(source_patches), patches_per_source_initial)
#             sampled = random.sample(source_patches, n_to_sample)
#             sampled_patches.extend(sampled)
            
#             if len(sampled_patches) >= target_count:
#                 break
        
#         # Second pass: fill remaining
#         if len(sampled_patches) < target_count:
#             remaining_needed = target_count - len(sampled_patches)
#             remaining_patches = []
#             for source, patches in patches_by_source.items():
#                 already_sampled = set(p for p in sampled_patches if p in patches)
#                 remaining = [p for p in patches if p not in already_sampled]
#                 remaining_patches.extend(remaining)
            
#             if remaining_patches:
#                 additional = min(remaining_needed, len(remaining_patches))
#                 sampled_patches.extend(random.sample(remaining_patches, additional))
        
#         if len(sampled_patches) > target_count:
#             sampled_patches = random.sample(sampled_patches, target_count)
        
#         return sampled_patches
    
#     def augment_patch_with_class_awareness(self, img_path, label_path, output_base, aug_index, primary_class_id):
#         """Augment patch using class-specific transformations"""
#         img = cv2.imread(str(img_path))
#         if img is None:
#             return False
        
#         # Select appropriate augmentation based on primary defect class
#         augment_transform = self.class_augmentations.get(primary_class_id, self.class_augmentations[0])
        
#         annotations = []
#         if label_path.exists():
#             with open(label_path, 'r') as f:
#                 for line in f:
#                     parts = line.strip().split()
#                     if len(parts) == 5:
#                         annotations.append([float(x) for x in parts])
        
#         if annotations:
#             bboxes = []
#             class_labels = []
#             for ann in annotations:
#                 class_id = int(ann[0])
#                 x_center, y_center, width, height = ann[1:]
#                 bboxes.append([x_center, y_center, width, height])
#                 class_labels.append(class_id)
            
#             try:
#                 augmented = augment_transform(
#                     image=img,
#                     bboxes=bboxes,
#                     class_labels=class_labels
#                 )
                
#                 aug_img_path = output_base.parent / 'images' / f"{output_base.stem}_aug{aug_index}.jpg"
#                 cv2.imwrite(str(aug_img_path), augmented['image'], [cv2.IMWRITE_JPEG_QUALITY, 100])
                
#                 aug_label_path = output_base.parent / 'labels' / f"{output_base.stem}_aug{aug_index}.txt"
#                 with open(aug_label_path, 'w') as f:
#                     for bbox, label in zip(augmented['bboxes'], augmented['class_labels']):
#                         f.write(f"{label} {' '.join(map(str, bbox))}\n")
                
#                 return True
#             except:
#                 return False
#         return False
    
#     def process_training_with_diversity(self, patches_by_class_source, background_by_source, df):
#         """Balance training set with diversity awareness and class-specific augmentation"""
#         source_split = self.source_dir / 'train'
#         output_split = self.output_dir / 'train'
        
#         (output_split / 'images').mkdir(parents=True, exist_ok=True)
#         (output_split / 'labels').mkdir(parents=True, exist_ok=True)
        
#         counts = {k: sum(len(patches) for patches in v.values()) 
#                  for k, v in patches_by_class_source.items()}
        
#         if self.strategy == 'mixed':
#             target = int(np.percentile(list(counts.values()), 75))
#             target = max(target, min(counts.values()) * 2)
#             target = min(target, 15000)  
#         elif self.strategy == 'oversample':
#             target = max(counts.values())
#         else:
#             target = min(counts.values())
        
#         logging.info(f"\nTraining balance target: {target} per class")
#         logging.info("Using class-specific physically realistic augmentations...")
        
#         split_stats = defaultdict(int)
#         source_usage_stats = defaultdict(lambda: defaultdict(int))
#         augmentation_stats = defaultdict(int)
        
#         for class_id, class_name in self.class_names.items():
#             patches_by_source = patches_by_class_source.get(class_id, {})
            
#             if not patches_by_source:
#                 logging.warning(f"  No patches found for {class_name}")
#                 continue
            
#             sampled = self.diversity_aware_sampling(patches_by_source, min(target, counts[class_id]))
            
#             for patch in sampled:
#                 source = self.extract_source_image_name(patch)
#                 source_usage_stats[class_name][source] += 1
            
#             logging.info(f"  {class_name}:")
#             logging.info(f"    Original: {len(sampled)} patches from {len(patches_by_source)} sources")
            
#             # Copy originals
#             for patch_name in tqdm(sampled, desc=f"Copying {class_name}", leave=False):
#                 src_img = source_split / 'images' / f"{patch_name}.jpg"
#                 src_lbl = source_split / 'labels' / f"{patch_name}.txt"
#                 dst_img = output_split / 'images' / f"{patch_name}.jpg"
#                 dst_lbl = output_split / 'labels' / f"{patch_name}.txt"
                
#                 if src_img.exists():
#                     shutil.copy2(src_img, dst_img)
#                 if src_lbl.exists():
#                     shutil.copy2(src_lbl, dst_lbl)
                
#                 split_stats[class_id] += 1
            
#             # Augment if under target
#             if len(sampled) < target:
#                 needed = target - len(sampled)
#                 logging.info(f"    Augmenting: {needed} samples (class-specific transforms)")
#                 augmentation_stats[class_name] = needed
                
#                 aug_sources = random.choices(sampled, k=needed)
                
#                 for i, patch_name in enumerate(tqdm(aug_sources, desc=f"Augmenting {class_name}", leave=False)):
#                     src_img = source_split / 'images' / f"{patch_name}.jpg"
#                     src_lbl = source_split / 'labels' / f"{patch_name}.txt"
                    
#                     if self.augment_patch_with_class_awareness(
#                         src_img, src_lbl, output_split / patch_name, i, class_id
#                     ):
#                         split_stats[class_id] += 1
        
#         # Process background patches
#         logging.info("\n  Processing background patches...")
#         n_defects = sum(split_stats.values())
#         bg_target = min(n_defects, sum(len(patches) for patches in background_by_source.values()))
#         bg_sampled = self.diversity_aware_sampling(background_by_source, bg_target)
        
#         logging.info(f"  Background: {len(bg_sampled)} patches from {len(background_by_source)} sources")
        
#         for patch_name in tqdm(bg_sampled, desc="Copying background", leave=False):
#             src_img = source_split / 'images' / f"{patch_name}.jpg"
#             src_lbl = source_split / 'labels' / f"{patch_name}.txt"
#             dst_img = output_split / 'images' / f"{patch_name}.jpg"
#             dst_lbl = output_split / 'labels' / f"{patch_name}.txt"
            
#             if src_img.exists():
#                 shutil.copy2(src_img, dst_img)
#             if src_lbl.exists():
#                 shutil.copy2(src_lbl, dst_lbl)
#             else:
#                 dst_lbl.touch()
        
#         # Log augmentation usage
#         logging.info("\nAugmentation statistics:")
#         for class_name, count in augmentation_stats.items():
#             logging.info(f"  {class_name}: {count} augmented images created")
        
#         return split_stats, source_usage_stats
    
#     def process_val_test(self, split_name, df):
#         """Copy ALL val/test data without any augmentation"""
#         source_split = self.source_dir / split_name
#         output_split = self.output_dir / split_name
        
#         (output_split / 'images').mkdir(parents=True, exist_ok=True)
#         (output_split / 'labels').mkdir(parents=True, exist_ok=True)
        
#         split_df = df[df['split'] == split_name]
        
#         copied = 0
#         for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Copying {split_name}"):
#             patch_name = row['filename']
#             src_img = source_split / 'images' / f"{patch_name}.jpg"
#             src_lbl = source_split / 'labels' / f"{patch_name}.txt"
#             dst_img = output_split / 'images' / f"{patch_name}.jpg"
#             dst_lbl = output_split / 'labels' / f"{patch_name}.txt"
            
#             if src_img.exists():
#                 shutil.copy2(src_img, dst_img)
#                 if src_lbl.exists():
#                     shutil.copy2(src_lbl, dst_lbl)
#                 else:
#                     dst_lbl.touch()
#                 copied += 1
        
#         split_stats = defaultdict(int)
#         defect_df = split_df[split_df['type'] == 'defect']
        
#         for _, row in defect_df.iterrows():
#             for class_id in row['classes']:
#                 split_stats[class_id] += 1
        
#         background_count = len(split_df[split_df['type'] == 'background'])
        
#         logging.info(f"\n{split_name.upper()} (no augmentation - real data only):")
#         for class_id, count in sorted(split_stats.items()):
#             logging.info(f"  {self.class_names[class_id]}: {count}")
#         logging.info(f"  Background: {background_count}")
#         logging.info(f"  TOTAL: {copied}")
        
#         return dict(split_stats)
    
#     def run(self):
#         # [Rest of run method remains the same]
#         logging.info("="*70)
#         logging.info(f"Diversity-Aware Dataset Balancing with Class-Specific Augmentation")
#         logging.info(f"Strategy: {self.strategy}")
#         logging.info("Training: Balanced with physically realistic augmentations")
#         logging.info("Val/Test: Natural distribution preserved (no augmentation)")
#         logging.info("="*70)
        
#         if self.output_dir.exists():
#             shutil.rmtree(self.output_dir)
#         self.output_dir.mkdir(parents=True)
        
#         df = self.load_metadata()
#         train_diversity = self.analyze_diversity(df, 'train')
#         patches_by_class_source, background_by_source = self.organize_patches_by_source(df, 'train')
        
#         all_stats = {}
        
#         logging.info("\n" + "="*70)
#         logging.info("TRAINING SET PROCESSING")
#         train_stats, source_usage = self.process_training_with_diversity(
#             patches_by_class_source, background_by_source, df
#         )
#         all_stats['train'] = train_stats
        
#         for split in ['val', 'test']:
#             if len(df[df['split'] == split]) > 0:
#                 logging.info("\n" + "="*70)
#                 logging.info(f"{split.upper()} SET PROCESSING")
#                 split_stats = self.process_val_test(split, df)
#                 all_stats[split] = split_stats
        
#         yaml_content = f"""path: {str(self.output_dir.absolute()).replace(chr(92), '/')}
# train: train/images
# val: val/images
# test: test/images

# nc: 6
# names: {list(self.class_names.values())}

# # Training: Balanced with class-specific physically realistic augmentations
# # Val/Test: Natural distribution (no augmentation)
# # Strategy: {self.strategy}
# """
#         with open(self.output_dir / 'dataset.yaml', 'w') as f:
#             f.write(yaml_content)
        
#         stats_output = {
#             'strategy': self.strategy,
#             'diversity_aware': True,
#             'class_specific_augmentation': True,
#             'training_diversity': train_diversity,
#             'final_distribution': all_stats,
#             'source_usage': source_usage
#         }
        
#         with open(self.output_dir / 'diversity_stats.json', 'w') as f:
#             json.dump(stats_output, f, indent=2, default=str)
        
#         logging.info("\n" + "="*70)
#         logging.info("COMPLETE!")
#         logging.info(f"Output: {self.output_dir}")
#         logging.info("="*70)
        
#         return stats_output

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--source', default='processed_balanced')
#     parser.add_argument('--output', default='processed_balanced_final')
#     parser.add_argument('--strategy', choices=['undersample', 'mixed', 'oversample'], 
#                        default='mixed', help='Balancing strategy')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed')
#     args = parser.parse_args()
    
#     # Set seed for reproducibility
#     random.seed(args.seed)
#     np.random.seed(args.seed)
    
#     balancer = DiversityAwareBalancer(
#         source_dir=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.source,
#         output_dir=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.output,
#         strategy=args.strategy
#     )
    
#     stats = balancer.run()


# scripts/balance_dataset_with_diversity.py - FIXED VERSION
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
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('balancing_diversity.log'),
        logging.StreamHandler()
    ]
)

class DiversityAwareBalancer:
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
        
        # Create class-specific augmentations that respect physical properties
        self.class_augmentations = self._create_class_specific_augmentations()
    
    def _create_class_specific_augmentations(self):
        """
        Create physically realistic augmentations for each defect type.
        Each defect has unique physical properties that must be preserved.
        """
        
        # Base intensity augmentations (safe for all X-ray images)
        base_intensity = [
            A.RandomBrightnessContrast(
                brightness_limit=0.1,  # Simulates exposure variations
                contrast_limit=0.1,    # Simulates penetration differences
                p=0.4
            ),
            A.RandomGamma(
                gamma_limit=(90, 110),  # Simulates film/detector response
                p=0.3
            ),
            A.GaussNoise(
                var_limit=(10.0, 50.0),  # Simulates detector noise
                mean=0,
                per_channel=True,
                p=0.3
            ),
        ]
        
        augmentations = {}
        
        # POROSITY (gas bubbles - can appear at any orientation)
        # Physical property: Round/spherical defects, orientation-independent
        augmentations[0] = A.Compose([
            A.RandomRotate90(p=0.5),  # Bubbles are round, rotation OK
            A.HorizontalFlip(p=0.5),  # Can flip freely
            A.VerticalFlip(p=0.5),    # Can flip freely
            *base_intensity,
            # NO shape distortion - would make bubbles non-circular
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
        # INCLUSION (solid particles trapped in weld)
        # Physical property: Irregular shapes but maintain form
        augmentations[1] = A.Compose([
            A.RandomRotate90(p=0.4),  # Can be at various angles
            A.HorizontalFlip(p=0.4),
            A.VerticalFlip(p=0.4),
            *base_intensity,
            A.ShiftScaleRotate(
                shift_limit=0.02,  # Very small shift
                scale_limit=0.0,   # NO scaling - preserve size
                rotate_limit=10,   # Small rotation
                p=0.3
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
        # CRACK (linear defects with sharp edges)
        # Physical property: Linear, follows stress patterns, sharp edges crucial
        augmentations[2] = A.Compose([
            A.HorizontalFlip(p=0.3),  # OK along crack direction
            A.VerticalFlip(p=0.2),    # Less common
            *base_intensity,
            # NO rotation - cracks follow specific stress patterns
            # NO blur - would lose critical sharp edges
            # NO geometric distortion
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
        # UNDERCUT (groove along weld toe)
        # Physical property: Position-specific at weld edges
        augmentations[3] = A.Compose([
            A.HorizontalFlip(p=0.3),  # Can appear on either side
            *base_intensity,
            A.ShiftScaleRotate(
                shift_limit=0.01,  # Tiny shift only
                scale_limit=0.0,   # NO scaling
                rotate_limit=5,    # Very small rotation
                p=0.2
            ),
            # More aggressive noise since we have few samples
            A.MultiplicativeNoise(multiplier=(0.95, 1.05), per_channel=True, p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
        # LACK OF FUSION (incomplete fusion at weld boundaries)
        # Physical property: Linear along weld interfaces, position-critical
        augmentations[4] = A.Compose([
            A.HorizontalFlip(p=0.3),  # Can occur on either side
            *base_intensity,
            # NO rotation - follows weld geometry
            # NO vertical flip - gravity affects formation
            # NO geometric distortion
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
        # LACK OF PENETRATION (incomplete root penetration)
        # Physical property: Always at weld centerline, straight lines
        augmentations[5] = A.Compose([
            A.HorizontalFlip(p=0.5),  # Can extend in either direction
            *base_intensity,
            # NO rotation - always aligned with weld
            # NO vertical flip - position-specific at root
            # NO geometric changes - very specific appearance
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
        return augmentations
    
    def extract_source_image_name(self, filename):
        """Extract the original source image name from patch filename"""
        match = re.match(r'(.+)_\d{6}$', filename)
        if match:
            return match.group(1)
        return filename
    
    def load_metadata(self):
        metadata_path = self.source_dir / 'patch_metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        df = pd.read_csv(metadata_path)
        df['classes'] = df['classes'].apply(lambda x: eval(x) if pd.notna(x) and x != '[]' else [])
        df['source_name'] = df['filename'].apply(self.extract_source_image_name)
        
        return df
    
    def analyze_diversity(self, df, split='train'):
        """Analyze diversity of source images per class"""
        split_df = df[(df['split'] == split) & (df['type'] == 'defect')]
        
        diversity_stats = {}
        for class_id, class_name in self.class_names.items():
            class_patches = []
            source_images = defaultdict(list)
            
            for _, row in split_df.iterrows():
                if class_id in row['classes']:
                    class_patches.append(row['filename'])
                    source_images[row['source_name']].append(row['filename'])
            
            diversity_stats[class_name] = {
                'total_patches': len(class_patches),
                'unique_sources': len(source_images),
                'source_distribution': {k: len(v) for k, v in source_images.items()}
            }
        
        logging.info(f"\nDiversity analysis for {split}:")
        for class_name, stats in diversity_stats.items():
            logging.info(f"  {class_name}:")
            logging.info(f"    Total patches: {stats['total_patches']}")
            logging.info(f"    Unique source images: {stats['unique_sources']}")
            if stats['total_patches'] > 0:
                avg_per_source = stats['total_patches'] / max(1, stats['unique_sources'])
                logging.info(f"    Average patches per source: {avg_per_source:.1f}")
        
        return diversity_stats
    
    def organize_patches_by_source(self, df, split='train'):
        """Organize patches by class and source image"""
        split_df = df[df['split'] == split]
        
        patches_by_class_source = defaultdict(lambda: defaultdict(list))
        
        defect_df = split_df[split_df['type'] == 'defect']
        for _, row in defect_df.iterrows():
            for class_id in row['classes']:
                patches_by_class_source[class_id][row['source_name']].append(row['filename'])
        
        background_by_source = defaultdict(list)
        bg_df = split_df[split_df['type'] == 'background']
        for _, row in bg_df.iterrows():
            background_by_source[row['source_name']].append(row['filename'])
        
        return patches_by_class_source, background_by_source
    
    def diversity_aware_sampling(self, patches_by_source, target_count):
        """Sample patches ensuring diversity across source images"""
        all_sources = list(patches_by_source.keys())
        if not all_sources:
            return []
        
        random.shuffle(all_sources)
        sampled_patches = []
        
        # First pass: sample from each source
        patches_per_source_initial = max(1, target_count // len(all_sources))
        
        for source in all_sources:
            source_patches = patches_by_source[source]
            n_to_sample = min(len(source_patches), patches_per_source_initial)
            sampled = random.sample(source_patches, n_to_sample)
            sampled_patches.extend(sampled)
            
            if len(sampled_patches) >= target_count:
                break
        
        # Second pass: fill remaining
        if len(sampled_patches) < target_count:
            remaining_needed = target_count - len(sampled_patches)
            remaining_patches = []
            for source, patches in patches_by_source.items():
                already_sampled = set(p for p in sampled_patches if p in patches)
                remaining = [p for p in patches if p not in already_sampled]
                remaining_patches.extend(remaining)
            
            if remaining_patches:
                additional = min(remaining_needed, len(remaining_patches))
                sampled_patches.extend(random.sample(remaining_patches, additional))
        
        if len(sampled_patches) > target_count:
            sampled_patches = random.sample(sampled_patches, target_count)
        
        return sampled_patches
    
    def augment_patch_with_class_awareness(self, img_path, label_path, output_base, aug_index, primary_class_id):
        """Augment patch using class-specific transformations"""
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        
        # Select appropriate augmentation based on primary defect class
        augment_transform = self.class_augmentations.get(primary_class_id, self.class_augmentations[0])
        
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
            
            # Pre-filter invalid boxes BEFORE augmentation
            for ann in annotations:
                class_id = int(ann[0])
                x_center, y_center, width, height = ann[1:]
                
                # Skip invalid boxes
                if width <= 0.01 or height <= 0.01:
                    continue
                if x_center <= 0 or x_center >= 1 or y_center <= 0 or y_center >= 1:
                    continue
                    
                # Ensure box is within bounds
                half_w = width / 2
                half_h = height / 2
                
                # Clip to ensure box stays within image
                if x_center - half_w < 0:
                    x_center = half_w
                if x_center + half_w > 1:
                    x_center = 1 - half_w
                if y_center - half_h < 0:
                    y_center = half_h
                if y_center + half_h > 1:
                    y_center = 1 - half_h
                    
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(class_id)
            
            # Only proceed if we have valid boxes
            if not bboxes:
                return False
                
            try:
                augmented = augment_transform(
                    image=img,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                # Additional validation after augmentation
                valid_bboxes = []
                valid_labels = []
                
                for bbox, label in zip(augmented['bboxes'], augmented['class_labels']):
                    x_center, y_center, width, height = bbox
                    
                    # Validate augmented box
                    if width > 0.01 and height > 0.01:
                        # Clip to valid range
                        x_center = np.clip(x_center, 0.01, 0.99)
                        y_center = np.clip(y_center, 0.01, 0.99)
                        width = min(width, 2 * min(x_center, 1 - x_center))
                        height = min(height, 2 * min(y_center, 1 - y_center))
                        
                        if width > 0.01 and height > 0.01:
                            valid_bboxes.append([x_center, y_center, width, height])
                            valid_labels.append(label)
                
                # Only save if we have valid boxes after augmentation
                if valid_bboxes:
                    aug_img_path = output_base.parent / 'images' / f"{output_base.stem}_aug{aug_index}.jpg"
                    cv2.imwrite(str(aug_img_path), augmented['image'], [cv2.IMWRITE_JPEG_QUALITY, 100])
                    
                    aug_label_path = output_base.parent / 'labels' / f"{output_base.stem}_aug{aug_index}.txt"
                    with open(aug_label_path, 'w') as f:
                        for bbox, label in zip(valid_bboxes, valid_labels):
                            x_center, y_center, width, height = bbox
                            f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    return True
                else:
                    return False
                    
            except Exception as e:
                # Fixed: removed patch_name reference
                logging.debug(f"Augmentation failed: {e}")
                return False

        return False
    
    def process_training_with_diversity(self, patches_by_class_source, background_by_source, df):
        """Balance training set with diversity awareness and class-specific augmentation"""
        source_split = self.source_dir / 'train'
        output_split = self.output_dir / 'train'
        
        (output_split / 'images').mkdir(parents=True, exist_ok=True)
        (output_split / 'labels').mkdir(parents=True, exist_ok=True)
        
        counts = {k: sum(len(patches) for patches in v.values()) 
                 for k, v in patches_by_class_source.items()}
        
        if self.strategy == 'mixed':
            target = int(np.percentile(list(counts.values()), 75))
            target = max(target, min(counts.values()) * 2)
            target = min(target, 15000)  
        elif self.strategy == 'oversample':
            target = max(counts.values())
        else:
            target = min(counts.values())
        
        logging.info(f"\nTraining balance target: {target} per class")
        logging.info("Using class-specific physically realistic augmentations...")
        
        split_stats = defaultdict(int)
        source_usage_stats = defaultdict(lambda: defaultdict(int))
        augmentation_stats = defaultdict(int)
        
        for class_id, class_name in self.class_names.items():
            patches_by_source = patches_by_class_source.get(class_id, {})
            
            if not patches_by_source:
                logging.warning(f"  No patches found for {class_name}")
                continue
            
            sampled = self.diversity_aware_sampling(patches_by_source, min(target, counts[class_id]))
            
            for patch in sampled:
                source = self.extract_source_image_name(patch)
                source_usage_stats[class_name][source] += 1
            
            logging.info(f"  {class_name}:")
            logging.info(f"    Original: {len(sampled)} patches from {len(patches_by_source)} sources")
            
            # Copy originals
            for patch_name in tqdm(sampled, desc=f"Copying {class_name}", leave=False):
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
                logging.info(f"    Augmenting: {needed} samples (class-specific transforms)")
                augmentation_stats[class_name] = needed
                
                aug_sources = random.choices(sampled, k=needed)
                
                for i, patch_name in enumerate(tqdm(aug_sources, desc=f"Augmenting {class_name}", leave=False)):
                    src_img = source_split / 'images' / f"{patch_name}.jpg"
                    src_lbl = source_split / 'labels' / f"{patch_name}.txt"
                    
                    if self.augment_patch_with_class_awareness(
                        src_img, src_lbl, output_split / patch_name, i, class_id
                    ):
                        split_stats[class_id] += 1
        
        # Process background patches
        logging.info("\n  Processing background patches...")
        n_defects = sum(split_stats.values())
        bg_target = min(n_defects, sum(len(patches) for patches in background_by_source.values()))
        bg_sampled = self.diversity_aware_sampling(background_by_source, bg_target)
        
        logging.info(f"  Background: {len(bg_sampled)} patches from {len(background_by_source)} sources")
        
        for patch_name in tqdm(bg_sampled, desc="Copying background", leave=False):
            src_img = source_split / 'images' / f"{patch_name}.jpg"
            src_lbl = source_split / 'labels' / f"{patch_name}.txt"
            dst_img = output_split / 'images' / f"{patch_name}.jpg"
            dst_lbl = output_split / 'labels' / f"{patch_name}.txt"
            
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                dst_lbl.touch()
        
        # Log augmentation usage
        logging.info("\nAugmentation statistics:")
        for class_name, count in augmentation_stats.items():
            logging.info(f"  {class_name}: {count} augmented images created")
        
        return split_stats, source_usage_stats
    
    def process_val_test(self, split_name, df):
        """Copy ALL val/test data without any augmentation"""
        source_split = self.source_dir / split_name
        output_split = self.output_dir / split_name
        
        (output_split / 'images').mkdir(parents=True, exist_ok=True)
        (output_split / 'labels').mkdir(parents=True, exist_ok=True)
        
        split_df = df[df['split'] == split_name]
        
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
                    dst_lbl.touch()
                copied += 1
        
        split_stats = defaultdict(int)
        defect_df = split_df[split_df['type'] == 'defect']
        
        for _, row in defect_df.iterrows():
            for class_id in row['classes']:
                split_stats[class_id] += 1
        
        background_count = len(split_df[split_df['type'] == 'background'])
        
        logging.info(f"\n{split_name.upper()} (no augmentation - real data only):")
        for class_id, count in sorted(split_stats.items()):
            logging.info(f"  {self.class_names[class_id]}: {count}")
        logging.info(f"  Background: {background_count}")
        logging.info(f"  TOTAL: {copied}")
        
        return dict(split_stats)
    
    def run(self):
        logging.info("="*70)
        logging.info(f"Diversity-Aware Dataset Balancing with Class-Specific Augmentation")
        logging.info(f"Strategy: {self.strategy}")
        logging.info("Training: Balanced with physically realistic augmentations")
        logging.info("Val/Test: Natural distribution preserved (no augmentation)")
        logging.info("="*70)
        
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)
        
        df = self.load_metadata()
        train_diversity = self.analyze_diversity(df, 'train')
        patches_by_class_source, background_by_source = self.organize_patches_by_source(df, 'train')
        
        all_stats = {}
        
        logging.info("\n" + "="*70)
        logging.info("TRAINING SET PROCESSING")
        train_stats, source_usage = self.process_training_with_diversity(
            patches_by_class_source, background_by_source, df
        )
        all_stats['train'] = train_stats
        
        for split in ['val', 'test']:
            if len(df[df['split'] == split]) > 0:
                logging.info("\n" + "="*70)
                logging.info(f"{split.upper()} SET PROCESSING")
                split_stats = self.process_val_test(split, df)
                all_stats[split] = split_stats
        
        yaml_content = f"""path: {str(self.output_dir.absolute()).replace(chr(92), '/')}
train: train/images
val: val/images
test: test/images

nc: 6
names: {list(self.class_names.values())}

# Training: Balanced with class-specific physically realistic augmentations
# Val/Test: Natural distribution (no augmentation)
# Strategy: {self.strategy}
"""
        with open(self.output_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)
        
        stats_output = {
            'strategy': self.strategy,
            'diversity_aware': True,
            'class_specific_augmentation': True,
            'training_diversity': train_diversity,
            'final_distribution': all_stats,
            'source_usage': source_usage
        }
        
        with open(self.output_dir / 'diversity_stats.json', 'w') as f:
            json.dump(stats_output, f, indent=2, default=str)
        
        logging.info("\n" + "="*70)
        logging.info("COMPLETE!")
        logging.info(f"Output: {self.output_dir}")
        logging.info("="*70)
        
        return stats_output

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='processed_balanced')
    parser.add_argument('--output', default='processed_balanced_final')
    parser.add_argument('--strategy', choices=['undersample', 'mixed', 'oversample'], 
                       default='mixed', help='Balancing strategy')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    balancer = DiversityAwareBalancer(
        source_dir=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.source,
        output_dir=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.output,
        strategy=args.strategy
    )
    
    stats = balancer.run()
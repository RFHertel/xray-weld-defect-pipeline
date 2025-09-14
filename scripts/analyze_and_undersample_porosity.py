# scripts/analyze_and_undersample_porosity.py
import pandas as pd
from pathlib import Path
from collections import defaultdict
import shutil
import json

def analyze_patches():
    """Analyze which patches contain which defects"""
    source_dir = Path(r'C:\AWrk\SWRD_YOLO_Project\processed_balanced_final\train')
    labels_dir = source_dir / 'labels'
    
    patch_info = []
    
    for label_file in labels_dir.glob('*.txt'):
        defect_classes = set()
        instance_counts = defaultdict(int)
        
        with open(label_file, 'r') as f:
            content = f.read().strip()
            if content:  # Has defects
                for line in content.split('\n'):
                    if line:
                        class_id = int(float(line.split()[0]))
                        defect_classes.add(class_id)
                        instance_counts[class_id] += 1
                
                patch_info.append({
                    'filename': label_file.stem,
                    'type': 'defect',
                    'classes': list(defect_classes),
                    'num_classes': len(defect_classes),
                    'porosity_count': instance_counts.get(0, 0),
                    'inclusion_count': instance_counts.get(1, 0),
                    'crack_count': instance_counts.get(2, 0),
                    'undercut_count': instance_counts.get(3, 0),
                    'lack_fusion_count': instance_counts.get(4, 0),
                    'lack_penetration_count': instance_counts.get(5, 0),
                    'is_porosity_only': defect_classes == {0}
                })
            else:  # Background
                patch_info.append({
                    'filename': label_file.stem,
                    'type': 'background',
                    'classes': [],
                    'num_classes': 0,
                    'is_porosity_only': False
                })
    
    df = pd.DataFrame(patch_info)
    
    # Analysis
    print("\n=== PATCH ANALYSIS ===")
    print(f"Total patches: {len(df)}")
    print(f"Background patches: {len(df[df['type'] == 'background'])}")
    print(f"Defect patches: {len(df[df['type'] == 'defect'])}")
    
    defect_df = df[df['type'] == 'defect']
    print(f"\nPorosity-only patches: {defect_df['is_porosity_only'].sum()}")
    print(f"Multi-class patches: {(defect_df['num_classes'] > 1).sum()}")
    
    # Class co-occurrence
    print("\n=== PATCHES CONTAINING EACH CLASS ===")
    for i in range(6):
        class_name = ['porosity', 'inclusion', 'crack', 'undercut', 'lack_fusion', 'lack_penetration'][i]
        patches_with_class = defect_df['classes'].apply(lambda x: i in x).sum()
        print(f"{class_name}: {patches_with_class} patches")
    
    return df

def undersample_porosity(df, target_reduction=0.5):
    """Remove porosity-only patches to balance dataset"""
    
    source_dir = Path(r'C:\AWrk\SWRD_YOLO_Project\processed_balanced_final\train')
    
    # Find porosity-only patches
    porosity_only = df[df['is_porosity_only'] == True].copy()
    print(f"\nFound {len(porosity_only)} porosity-only patches")
    
    # Sort by porosity count (remove patches with most porosity first)
    porosity_only = porosity_only.sort_values('porosity_count', ascending=False)
    
    # Remove top N% of porosity-only patches
    to_remove = porosity_only.head(int(len(porosity_only) * target_reduction))
    print(f"Removing {len(to_remove)} porosity-only patches ({target_reduction*100:.0f}%)")
    
    # Track what we're removing
    total_porosity_removed = to_remove['porosity_count'].sum()
    print(f"This removes {total_porosity_removed} porosity instances")
    
    # Actually remove files
    removed_list = []
    for filename in to_remove['filename']:
        img_path = source_dir / 'images' / f"{filename}.jpg"
        lbl_path = source_dir / 'labels' / f"{filename}.txt"
        
        if img_path.exists():
            img_path.unlink()
        if lbl_path.exists():
            lbl_path.unlink()
        removed_list.append(filename)
    
    # Save removal log
    with open(source_dir.parent / 'removed_patches.json', 'w') as f:
        json.dump({
            'removed_patches': removed_list,
            'count': len(removed_list),
            'porosity_instances_removed': int(total_porosity_removed)
        }, f, indent=2)
    
    print(f"\nRemoved {len(removed_list)} patches")
    
    # Recalculate balance
    remaining_df = df[~df['filename'].isin(removed_list)]
    print("\n=== NEW BALANCE ===")
    for i in range(6):
        class_name = ['porosity', 'inclusion', 'crack', 'undercut', 'lack_fusion', 'lack_penetration'][i]
        col_name = f"{class_name.replace('_', '_')}_count"
        if col_name in remaining_df.columns:
            total_instances = remaining_df[col_name].sum()
            print(f"{class_name}: {total_instances} instances")

if __name__ == "__main__":
    # Analyze current state
    df = analyze_patches()
    
    # Save analysis
    df.to_csv('patch_analysis.csv', index=False)
    print("\nSaved detailed analysis to patch_analysis.csv")
    
    # Ask before removing
    # response = input("\nRemove 50% of porosity-only patches? (y/n): ")
    # if response.lower() == 'y':
    #     undersample_porosity(df, target_reduction=0.5)


# How to read the CSV files
# Let me break down that exact line for you:
# A_bam5_390957,defect,"[0, 2, 5]",3,3.0,0.0,15.0,0.0,0.0,1.0,False
# Reading left to right:

# A_bam5_390957 = filename
# defect = this patch contains defects (not background)
# [0, 2, 5] = this patch contains classes 0, 2, and 5
# 3 = contains 3 different defect types
# 3.0 = 3 porosity defects (class 0)
# 0.0 = 0 inclusion defects (class 1)
# 15.0 = 15 crack defects (class 2)
# 0.0 = 0 undercut defects (class 3)
# 0.0 = 0 lack_of_fusion defects (class 4)
# 1.0 = 1 lack_of_penetration defect (class 5)
# False = NOT porosity-only (because it has cracks and lack_of_penetration too)

# So this patch A_bam5_390957 contains:

# 3 porosity instances
# 15 crack instances
# 1 lack_of_penetration instance
# Total: 19 defect bounding boxes

# The [0, 2, 5] matches the counts: class 0 has 3 instances, class 2 has 15 instances, class 5 has 1 instance.
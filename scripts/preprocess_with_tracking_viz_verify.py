# # preprocess_with_tracking_viz_verify.py
# # python scripts/preprocess_with_tracking_viz_verify.py   
# import cv2
# import json
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# # Config (adapt to your paths)
# BASE_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data')
# PROCESSED_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\processed_balanced')
# EXAMPLE_BASE_NAME = 'A_bam5'  # From your JSON example
# SPLIT = 'train'  # Check train/val/test as needed

# # Class names and colors (BGR for OpenCV)
# CLASS_NAMES = {0: 'porosity', 1: 'inclusion', 2: 'crack', 3: 'undercut', 4: 'lack_of_fusion', 5: 'lack_of_penetration'}
# COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255)}

# def find_original_files(base_name):
#     """Find TIF and JSON for original."""
#     tif_path = None
#     json_path = None
#     for subdir in ['crop_weld_images/L/1', 'crop_weld_images/L/2', 'crop_weld_images/T/1', 'crop_weld_images/T/2']:
#         candidate_tif = BASE_DIR / subdir / f'{base_name}.tif'
#         if candidate_tif.exists():
#             tif_path = candidate_tif
#             break
#     for subdir in ['crop_weld_jsons/L/1', 'crop_weld_jsons/L/2', 'crop_weld_jsons/T/1', 'crop_weld_jsons/T/2']:
#         candidate_json = BASE_DIR / subdir / f'{base_name}.json'
#         if candidate_json.exists():
#             json_path = candidate_json
#             break
#     return tif_path, json_path

# def draw_annotations_on_original(img, json_path):
#     """Draw JSON polygons/bboxes on original image."""
#     with open(json_path, 'r', encoding='utf-8') as f:
#         ann_data = json.load(f)
#     for shape in ann_data.get('shapes', []):
#         label = shape.get('label', '')
#         class_id = {  # Your class_map from script
#             '\u6c14\u5b54': 0, '气孔': 0,
#             '\u5939\u6e23': 1, '夹渣': 1,
#             '\u88c2\u7eb9': 2, '裂纹': 2,
#             '\u54ac\u8fb9': 3, '咬边': 3,
#             '\u672a\u878d\u5408': 4, '未熔合': 4,
#             '\u672a\u710a\u900f': 5, '未焊透': 5,
#             '内凹': 3,
#             '夹钨': 1,
#         }.get(label, -1)
#         if class_id == -1:
#             continue
#         points = np.array(shape['points'], np.int32)
#         cv2.polylines(img, [points], True, COLORS.get(class_id, (0, 0, 255)), 2)
#         # Optional: Draw bbox approximation too
#         x_min, y_min = np.min(points, axis=0)
#         x_max, y_max = np.max(points, axis=0)
#         cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), COLORS.get(class_id, (0, 0, 255)), 1)
#         cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def draw_yolo_bboxes_on_patch(img, label_path):
#     """Draw YOLO bboxes on patch image."""
#     h, w = img.shape[:2]
#     if not label_path.exists():
#         return img  # Empty label
#     with open(label_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue
#             class_id = int(parts[0])
#             cx, cy, bw, bh = map(float, parts[1:])
#             x_min = int((cx - bw / 2) * w)
#             y_min = int((cy - bh / 2) * h)
#             x_max = int((cx + bw / 2) * w)
#             y_max = int((cy + bh / 2) * h)
#             cv2.rectangle(img, (x_min, y_min), (x_max, y_max), COLORS.get(class_id, (0, 0, 255)), 2)
#             cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def visualize_example(base_name, split='train', max_patches=5):
#     """Visualize original vs patches (limit to avoid too many plots)."""
#     tif_path, json_path = find_original_files(base_name)
#     if not tif_path or not json_path:
#         print(f"Original files not found for {base_name}")
#         return

#     # Load and annotate original
#     original = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
#     if len(original.shape) == 2:  # Grayscale to BGR
#         original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
#     original_annotated = draw_annotations_on_original(original.copy(), json_path)

#     # Find matching patches
#     images_dir = PROCESSED_DIR / split / 'images'
#     labels_dir = PROCESSED_DIR / split / 'labels'
#     patch_files = sorted([f for f in images_dir.glob('*.jpg') if base_name in f.stem])
#     print(f"Found {len(patch_files)} patches for {base_name}")

#     # Plot
#     fig, axs = plt.subplots(2, min(max_patches + 1, len(patch_files) + 1), figsize=(15, 10), squeeze=False)
#     axs[0, 0].imshow(cv2.cvtColor(original_annotated, cv2.COLOR_BGR2RGB))
#     axs[0, 0].set_title('Original Annotated')
#     axs[0, 0].axis('off')
#     axs[1, 0].axis('off')  # Empty slot under original

#     for i, patch_path in enumerate(patch_files[:max_patches]):
#         label_path = labels_dir / f'{patch_path.stem}.txt'
#         patch_img = cv2.imread(str(patch_path))
#         patch_annotated = draw_yolo_bboxes_on_patch(patch_img.copy(), label_path)
#         axs[0, i + 1].imshow(cv2.cvtColor(patch_annotated, cv2.COLOR_BGR2RGB))
#         axs[0, i + 1].set_title(f'Patch {patch_path.stem}')
#         axs[0, i + 1].axis('off')
#         axs[1, i + 1].axis('off')  # Can add more if needed (e.g., zoomed)

#     plt.tight_layout()
#     plt.show()

# # Run for your example
# visualize_example(EXAMPLE_BASE_NAME, SPLIT)

# Working but need more visualization
# import cv2
# import json
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# # Config (adapt to your paths)
# BASE_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data')
# PROCESSED_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\processed_balanced')
# EXAMPLE_BASE_NAMES = ['A_bam5']  # Updated to 'A_bam5'; add more like ['A_bam5', 'another_file'] for batch
# SPLIT = 'train'  # Check train/val/test as needed

# # Class names and colors (BGR for OpenCV)
# CLASS_NAMES = {0: 'porosity', 1: 'inclusion', 2: 'crack', 3: 'undercut', 4: 'lack_of_fusion', 5: 'lack_of_penetration'}
# COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255)}

# def find_original_files(base_name):
#     """Find TIF and JSON for original, with logging."""
#     tif_path = None
#     json_path = None
#     for subdir in ['crop_weld_images/L/1', 'crop_weld_images/L/2', 'crop_weld_images/T/1', 'crop_weld_images/T/2']:
#         candidate_tif = BASE_DIR / subdir / f'{base_name}.tif'
#         if candidate_tif.exists():
#             tif_path = candidate_tif
#             print(f"Found TIF: {tif_path}")
#             break
#     for subdir in ['crop_weld_jsons/L/1', 'crop_weld_jsons/L/2', 'crop_weld_jsons/T/1', 'crop_weld_jsons/T/2']:
#         candidate_json = BASE_DIR / subdir / f'{base_name}.json'
#         if candidate_json.exists():
#             json_path = candidate_json
#             print(f"Found JSON: {json_path}")
#             break
#     if not tif_path or not json_path:
#         print(f"Warning: Files not found for {base_name}. Check prefixes (e.g., 'A_') or subdirs.")
#     return tif_path, json_path

# def draw_annotations_on_original(img, json_path):
#     """Draw JSON polygons/bboxes on original image."""
#     with open(json_path, 'r', encoding='utf-8') as f:
#         ann_data = json.load(f)
#     for shape in ann_data.get('shapes', []):
#         label = shape.get('label', '')
#         class_id = {  # Your class_map from preprocessing script
#             '\u6c14\u5b54': 0, '气孔': 0,
#             '\u5939\u6e23': 1, '夹渣': 1,
#             '\u88c2\u7eb9': 2, '裂纹': 2,
#             '\u54ac\u8fb9': 3, '咬边': 3,
#             '\u672a\u878d\u5408': 4, '未熔合': 4,
#             '\u672a\u710a\u900f': 5, '未焊透': 5,
#             '内凹': 3,
#             '夹钨': 1,
#         }.get(label, -1)
#         if class_id == -1:
#             continue
#         points = np.array(shape['points'], np.int32)
#         cv2.polylines(img, [points], True, COLORS.get(class_id, (0, 0, 255)), 2)
#         # Optional: Draw bbox approximation too for comparison
#         x_min, y_min = np.min(points, axis=0)
#         x_max, y_max = np.max(points, axis=0)
#         cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), COLORS.get(class_id, (0, 0, 255)), 1)
#         cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def draw_yolo_bboxes_on_patch(img, label_path):
#     """Draw YOLO bboxes on patch image."""
#     h, w = img.shape[:2]
#     if not label_path.exists():
#         print(f"No label file at {label_path} (background patch)")
#         return img
#     with open(label_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue
#             class_id = int(parts[0])
#             cx, cy, bw, bh = map(float, parts[1:])
#             x_min = int((cx - bw / 2) * w)
#             y_min = int((cy - bh / 2) * h)
#             x_max = int((cx + bw / 2) * w)
#             y_max = int((cy + bh / 2) * h)
#             cv2.rectangle(img, (x_min, y_min), (x_max, y_max), COLORS.get(class_id, (0, 0, 255)), 2)
#             cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def visualize_example(base_name, split='train', max_patches=5):
#     """Visualize original vs patches (limit to avoid too many plots)."""
#     tif_path, json_path = find_original_files(base_name)
#     if not tif_path or not json_path:
#         return  # Skip if not found

#     # Load and annotate original (handle grayscale)
#     original = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
#     if original is None:
#         print(f"Error loading {tif_path}")
#         return
#     if len(original.shape) == 2:
#         original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
#     original_annotated = draw_annotations_on_original(original.copy(), json_path)

#     # Find matching patches
#     images_dir = PROCESSED_DIR / split / 'images'
#     labels_dir = PROCESSED_DIR / split / 'labels'
#     patch_files = sorted([f for f in images_dir.glob('*.jpg') if f.stem.startswith(f'{base_name}_')])
#     print(f"Found {len(patch_files)} patches for {base_name} in {split}")

#     if not patch_files:
#         return

#     # Plot
#     fig, axs = plt.subplots(2, min(max_patches + 1, len(patch_files) + 1), figsize=(15, 10), squeeze=False)
#     axs[0, 0].imshow(cv2.cvtColor(original_annotated, cv2.COLOR_BGR2RGB))
#     axs[0, 0].set_title(f'Original Annotated: {base_name}')
#     axs[0, 0].axis('off')
#     axs[1, 0].axis('off')  # Empty under original

#     for i, patch_path in enumerate(patch_files[:max_patches]):
#         label_path = labels_dir / f'{patch_path.stem}.txt'
#         patch_img = cv2.imread(str(patch_path))
#         if patch_img is None:
#             print(f"Error loading {patch_path}")
#             continue
#         patch_annotated = draw_yolo_bboxes_on_patch(patch_img.copy(), label_path)
#         axs[0, i + 1].imshow(cv2.cvtColor(patch_annotated, cv2.COLOR_BGR2RGB))
#         axs[0, i + 1].set_title(f'Patch {patch_path.stem}')
#         axs[0, i + 1].axis('off')
#         axs[1, i + 1].axis('off')  # Can add zoom/overlay if needed

#     plt.tight_layout()
#     plt.show()

# # Run for examples
# for base_name in EXAMPLE_BASE_NAMES:
#     visualize_example(base_name, SPLIT)

#Better Montage:
# import cv2
# import json
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# # Config (adapt to your paths)
# BASE_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data')
# PROCESSED_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\processed_balanced')
# EXAMPLE_BASE_NAMES = ['A_bam5']  # Add more if needed
# SPLIT = 'train'  # Check train/val/test as needed

# # Class names and colors (BGR for OpenCV)
# CLASS_NAMES = {0: 'porosity', 1: 'inclusion', 2: 'crack', 3: 'undercut', 4: 'lack_of_fusion', 5: 'lack_of_penetration'}
# COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255)}

# def find_original_files(base_name):
#     """Find TIF and JSON for original, with logging."""
#     tif_path = None
#     json_path = None
#     for subdir in ['crop_weld_images/L/1', 'crop_weld_images/L/2', 'crop_weld_images/T/1', 'crop_weld_images/T/2']:
#         candidate_tif = BASE_DIR / subdir / f'{base_name}.tif'
#         if candidate_tif.exists():
#             tif_path = candidate_tif
#             print(f"Found TIF: {tif_path}")
#             break
#     for subdir in ['crop_weld_jsons/L/1', 'crop_weld_jsons/L/2', 'crop_weld_jsons/T/1', 'crop_weld_jsons/T/2']:
#         candidate_json = BASE_DIR / subdir / f'{base_name}.json'
#         if candidate_json.exists():
#             json_path = candidate_json
#             print(f"Found JSON: {json_path}")
#             break
#     if not tif_path or not json_path:
#         print(f"Warning: Files not found for {base_name}. Check prefixes (e.g., 'A_') or subdirs.")
#     return tif_path, json_path

# def draw_annotations_on_original(img, json_path):
#     """Draw JSON polygons/bboxes on original image."""
#     with open(json_path, 'r', encoding='utf-8') as f:
#         ann_data = json.load(f)
#     for shape in ann_data.get('shapes', []):
#         label = shape.get('label', '')
#         class_id = {  # Your class_map from preprocessing script
#             '\u6c14\u5b54': 0, '气孔': 0,
#             '\u5939\u6e23': 1, '夹渣': 1,
#             '\u88c2\u7eb9': 2, '裂纹': 2,
#             '\u54ac\u8fb9': 3, '咬边': 3,
#             '\u672a\u878d\u5408': 4, '未熔合': 4,
#             '\u672a\u710a\u900f': 5, '未焊透': 5,
#             '内凹': 3,
#             '夹钨': 1,
#         }.get(label, -1)
#         if class_id == -1:
#             continue
#         points = np.array(shape['points'], np.int32)
#         cv2.polylines(img, [points], True, COLORS.get(class_id, (0, 0, 255)), 2)
#         # Draw bbox approximation for comparison
#         x_min, y_min = np.min(points, axis=0)
#         x_max, y_max = np.max(points, axis=0)
#         cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), COLORS.get(class_id, (0, 0, 255)), 1)
#         cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def draw_yolo_bboxes_on_patch(img, label_path):
#     """Draw YOLO bboxes on patch image."""
#     h, w = img.shape[:2]
#     if not label_path.exists():
#         print(f"No label file at {label_path} (background patch)")
#         return img
#     with open(label_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue
#             class_id = int(parts[0])
#             cx, cy, bw, bh = map(float, parts[1:])
#             x_min = int((cx - bw / 2) * w)
#             y_min = int((cy - bh / 2) * h)
#             x_max = int((cx + bw / 2) * w)
#             y_max = int((cy + bh / 2) * h)
#             cv2.rectangle(img, (x_min, y_min), (x_max, y_max), COLORS.get(class_id, (0, 0, 255)), 2)
#             cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def create_overview_montage(original_annotated, patch_annotated_list, base_name):
#     """Create a single overview image with original on top and patches in grid below."""
#     # Resize original to match patch width if needed (assuming patches are square)
#     patch_size = patch_annotated_list[0].shape[0] if patch_annotated_list else 320
#     original_resized = cv2.resize(original_annotated, (patch_size * 5, int(original_annotated.shape[0] * (patch_size * 5 / original_annotated.shape[1]))))  # Scale to 5 patches wide

#     # Create grid for patches
#     num_patches = len(patch_annotated_list)
#     cols = 5
#     rows = int(np.ceil(num_patches / cols))
#     montage_height = rows * patch_size
#     montage_width = cols * patch_size
#     montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)

#     for i, patch in enumerate(patch_annotated_list):
#         row = i // cols
#         col = i % cols
#         montage[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = patch

#     # Stack original on top
#     full_montage = np.vstack((original_resized, montage)) if num_patches > 0 else original_resized

#     overview_path = PROCESSED_DIR / f'{base_name}_overview.jpg'
#     cv2.imwrite(str(overview_path), full_montage)
#     print(f"Overview saved to {overview_path}")
#     cv2.imshow('Overview Montage', full_montage)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def interactive_flip_through_patches(patch_annotated_list, patch_files):
#     """Interactive high-res flipping through patches with OpenCV."""
#     if not patch_annotated_list:
#         return
#     idx = 0
#     while True:
#         img = patch_annotated_list[idx]
#         cv2.imshow(f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}', img)
#         key = cv2.waitKey(0)
#         if key == 27:  # ESC to exit
#             break
#         elif key == ord('n') or key == 83:  # 'n' or right arrow
#             idx = (idx + 1) % len(patch_annotated_list)
#         elif key == ord('p') or key == 81:  # 'p' or left arrow
#             idx = (idx - 1) % len(patch_annotated_list)
#     cv2.destroyAllWindows()

# def visualize_example(base_name, split='train'):
#     """Updated visualization: overview montage and interactive flipping."""
#     tif_path, json_path = find_original_files(base_name)
#     if not tif_path or not json_path:
#         return

#     # Load and annotate original
#     original = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
#     if original is None:
#         print(f"Error loading {tif_path}")
#         return
#     if len(original.shape) == 2:
#         original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
#     original_annotated = draw_annotations_on_original(original.copy(), json_path)

#     # Find all matching patches
#     images_dir = PROCESSED_DIR / split / 'images'
#     labels_dir = PROCESSED_DIR / split / 'labels'
#     patch_files = sorted([f for f in images_dir.glob('*.jpg') if f.stem.startswith(f'{base_name}_')])
#     print(f"Found {len(patch_files)} patches for {base_name} in {split}")

#     if not patch_files:
#         return

#     # Load and annotate all patches
#     patch_annotated_list = []
#     for patch_path in patch_files:
#         label_path = labels_dir / f'{patch_path.stem}.txt'
#         patch_img = cv2.imread(str(patch_path))
#         if patch_img is None:
#             print(f"Error loading {patch_path}")
#             continue
#         patch_annotated = draw_yolo_bboxes_on_patch(patch_img.copy(), label_path)
#         patch_annotated_list.append(patch_annotated)

#     # Create and show/save overview montage
#     create_overview_montage(original_annotated, patch_annotated_list, base_name)

#     # Interactive high-res flipping
#     print("Press 'n' or right arrow for next, 'p' or left arrow for previous, ESC to exit.")
#     interactive_flip_through_patches(patch_annotated_list, patch_files)

# # Run for examples
# for base_name in EXAMPLE_BASE_NAMES:
#     visualize_example(base_name, SPLIT)

# Showing all images in montage and saving in a folder
# import cv2
# import json
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import shutil

# # Config (adapt to your paths)
# BASE_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data')
# PROCESSED_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\processed_balanced')
# VISUALIZATIONS_DIR = PROCESSED_DIR / 'visualizations'  # New folder for visualizations
# EXAMPLE_BASE_NAMES = ['A_bam5']  # Add more if needed
# SPLIT = 'train'  # Check train/val/test as needed

# # Class names and colors (BGR for OpenCV)
# CLASS_NAMES = {0: 'porosity', 1: 'inclusion', 2: 'crack', 3: 'undercut', 4: 'lack_of_fusion', 5: 'lack_of_penetration'}
# COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255)}

# def setup_visualizations_dir():
#     """Create or empty the visualizations directory."""
#     if VISUALIZATIONS_DIR.exists():
#         shutil.rmtree(VISUALIZATIONS_DIR)  # Empty the folder
#     VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
#     print(f"Visualizations directory setup at {VISUALIZATIONS_DIR}")

# def find_original_files(base_name):
#     """Find TIF and JSON for original, with logging."""
#     tif_path = None
#     json_path = None
#     for subdir in ['crop_weld_images/L/1', 'crop_weld_images/L/2', 'crop_weld_images/T/1', 'crop_weld_images/T/2']:
#         candidate_tif = BASE_DIR / subdir / f'{base_name}.tif'
#         if candidate_tif.exists():
#             tif_path = candidate_tif
#             print(f"Found TIF: {tif_path}")
#             break
#     for subdir in ['crop_weld_jsons/L/1', 'crop_weld_jsons/L/2', 'crop_weld_jsons/T/1', 'crop_weld_jsons/T/2']:
#         candidate_json = BASE_DIR / subdir / f'{base_name}.json'
#         if candidate_json.exists():
#             json_path = candidate_json
#             print(f"Found JSON: {json_path}")
#             break
#     if not tif_path or not json_path:
#         print(f"Warning: Files not found for {base_name}. Check prefixes (e.g., 'A_') or subdirs.")
#     return tif_path, json_path

# def draw_annotations_on_original(img, json_path):
#     """Draw JSON polygons/bboxes on original image."""
#     with open(json_path, 'r', encoding='utf-8') as f:
#         ann_data = json.load(f)
#     for shape in ann_data.get('shapes', []):
#         label = shape.get('label', '')
#         class_id = {  # Your class_map from preprocessing script
#             '\u6c14\u5b54': 0, '气孔': 0,
#             '\u5939\u6e23': 1, '夹渣': 1,
#             '\u88c2\u7eb9': 2, '裂纹': 2,
#             '\u54ac\u8fb9': 3, '咬边': 3,
#             '\u672a\u878d\u5408': 4, '未熔合': 4,
#             '\u672a\u710a\u900f': 5, '未焊透': 5,
#             '内凹': 3,
#             '夹钨': 1,
#         }.get(label, -1)
#         if class_id == -1:
#             continue
#         points = np.array(shape['points'], np.int32)
#         cv2.polylines(img, [points], True, COLORS.get(class_id, (0, 0, 255)), 2)
#         # Draw bbox approximation for comparison
#         x_min, y_min = np.min(points, axis=0)
#         x_max, y_max = np.max(points, axis=0)
#         cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), COLORS.get(class_id, (0, 0, 255)), 1)
#         cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def draw_yolo_bboxes_on_patch(img, label_path):
#     """Draw YOLO bboxes on patch image."""
#     h, w = img.shape[:2]
#     if not label_path.exists():
#         print(f"No label file at {label_path} (background patch)")
#         return img
#     with open(label_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue
#             class_id = int(parts[0])
#             cx, cy, bw, bh = map(float, parts[1:])
#             x_min = int((cx - bw / 2) * w)
#             y_min = int((cy - bh / 2) * h)
#             x_max = int((cx + bw / 2) * w)
#             y_max = int((cy + bh / 2) * h)
#             cv2.rectangle(img, (x_min, y_min), (x_max, y_max), COLORS.get(class_id, (0, 0, 255)), 2)
#             cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def create_overview_montage(original_annotated, patch_annotated_list, base_name):
#     """Create a single overview image with original on top and patches in grid below."""
#     # Resize original to match patch width if needed (assuming patches are square)
#     patch_size = patch_annotated_list[0].shape[0] if patch_annotated_list else 320
#     original_resized = cv2.resize(original_annotated, (patch_size * 5, int(original_annotated.shape[0] * (patch_size * 5 / original_annotated.shape[1]))))  # Scale to 5 patches wide

#     # Create grid for patches
#     num_patches = len(patch_annotated_list)
#     cols = 5
#     rows = int(np.ceil(num_patches / cols))
#     montage_height = rows * patch_size
#     montage_width = cols * patch_size
#     montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)

#     for i, patch in enumerate(patch_annotated_list):
#         row = i // cols
#         col = i % cols
#         montage[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = patch

#     # Stack original on top
#     full_montage = np.vstack((original_resized, montage)) if num_patches > 0 else original_resized

#     overview_path = VISUALIZATIONS_DIR / f'{base_name}_overview.jpg'  # Save in visualizations folder
#     cv2.imwrite(str(overview_path), full_montage)
#     print(f"Overview saved to {overview_path}")
#     cv2.imshow('Overview Montage', full_montage)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def interactive_flip_through_patches(patch_annotated_list, patch_files):
#     """Interactive high-res flipping through patches with OpenCV."""
#     if not patch_annotated_list:
#         return
#     idx = 0
#     while True:
#         img = patch_annotated_list[idx]
#         cv2.imshow(f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}', img)
#         key = cv2.waitKey(0)
#         if key == 27:  # ESC to exit
#             break
#         elif key == ord('n') or key == 83:  # 'n' or right arrow
#             idx = (idx + 1) % len(patch_annotated_list)
#         elif key == ord('p') or key == 81:  # 'p' or left arrow
#             idx = (idx - 1) % len(patch_annotated_list)
#     cv2.destroyAllWindows()

# def visualize_example(base_name, split='train'):
#     """Updated visualization: overview montage and interactive flipping."""
#     tif_path, json_path = find_original_files(base_name)
#     if not tif_path or not json_path:
#         return

#     # Load and annotate original
#     original = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
#     if original is None:
#         print(f"Error loading {tif_path}")
#         return
#     if len(original.shape) == 2:
#         original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
#     original_annotated = draw_annotations_on_original(original.copy(), json_path)

#     # Find all matching patches
#     images_dir = PROCESSED_DIR / split / 'images'
#     labels_dir = PROCESSED_DIR / split / 'labels'
#     patch_files = sorted([f for f in images_dir.glob('*.jpg') if f.stem.startswith(f'{base_name}_')])
#     print(f"Found {len(patch_files)} patches for {base_name} in {split}")

#     if not patch_files:
#         return

#     # Load and annotate all patches
#     patch_annotated_list = []
#     for patch_path in patch_files:
#         label_path = labels_dir / f'{patch_path.stem}.txt'
#         patch_img = cv2.imread(str(patch_path))
#         if patch_img is None:
#             print(f"Error loading {patch_path}")
#             continue
#         patch_annotated = draw_yolo_bboxes_on_patch(patch_img.copy(), label_path)
#         patch_annotated_list.append(patch_annotated)

#     # Create and show/save overview montage
#     create_overview_montage(original_annotated, patch_annotated_list, base_name)

#     # Interactive high-res flipping
#     print("Press 'n' or right arrow for next, 'p' or left arrow for previous, ESC to exit.")
#     interactive_flip_through_patches(patch_annotated_list, patch_files)

# # Setup visualizations dir (empties on each run)
# setup_visualizations_dir()

# # Run for examples
# for base_name in EXAMPLE_BASE_NAMES:
#     visualize_example(base_name, SPLIT)


# Next attempt:

# import cv2
# import json
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import shutil

# # Config (adapt to your paths)
# BASE_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data')
# PROCESSED_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\processed_balanced')
# VISUALIZATIONS_DIR = PROCESSED_DIR / 'visualizations'  # New folder for visualizations
# EXAMPLE_BASE_NAMES = ['A_bam5']  # Add more if needed
# SPLIT = 'train'  # Check train/val/test as needed

# # Class names and colors (BGR for OpenCV)
# CLASS_NAMES = {0: 'porosity', 1: 'inclusion', 2: 'crack', 3: 'undercut', 4: 'lack_of_fusion', 5: 'lack_of_penetration'}
# COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255)}

# def setup_visualizations_dir():
#     """Create or empty the visualizations directory."""
#     if VISUALIZATIONS_DIR.exists():
#         shutil.rmtree(VISUALIZATIONS_DIR)  # Empty the folder
#     VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
#     print(f"Visualizations directory setup at {VISUALIZATIONS_DIR}")

# def find_original_files(base_name):
#     """Find TIF and JSON for original, with logging."""
#     tif_path = None
#     json_path = None
#     for subdir in ['crop_weld_images/L/1', 'crop_weld_images/L/2', 'crop_weld_images/T/1', 'crop_weld_images/T/2']:
#         candidate_tif = BASE_DIR / subdir / f'{base_name}.tif'
#         if candidate_tif.exists():
#             tif_path = candidate_tif
#             print(f"Found TIF: {tif_path}")
#             break
#     for subdir in ['crop_weld_jsons/L/1', 'crop_weld_jsons/L/2', 'crop_weld_jsons/T/1', 'crop_weld_jsons/T/2']:
#         candidate_json = BASE_DIR / subdir / f'{base_name}.json'
#         if candidate_json.exists():
#             json_path = candidate_json
#             print(f"Found JSON: {json_path}")
#             break
#     if not tif_path or not json_path:
#         print(f"Warning: Files not found for {base_name}. Check prefixes (e.g., 'A_') or subdirs.")
#     return tif_path, json_path

# def draw_annotations_on_original(img, json_path):
#     """Draw JSON polygons/bboxes on original image."""
#     with open(json_path, 'r', encoding='utf-8') as f:
#         ann_data = json.load(f)
#     for shape in ann_data.get('shapes', []):
#         label = shape.get('label', '')
#         class_id = {  # Your class_map from preprocessing script
#             '\u6c14\u5b54': 0, '气孔': 0,
#             '\u5939\u6e23': 1, '夹渣': 1,
#             '\u88c2\u7eb9': 2, '裂纹': 2,
#             '\u54ac\u8fb9': 3, '咬边': 3,
#             '\u672a\u878d\u5408': 4, '未熔合': 4,
#             '\u672a\u710a\u900f': 5, '未焊透': 5,
#             '内凹': 3,
#             '夹钨': 1,
#         }.get(label, -1)
#         if class_id == -1:
#             continue
#         points = np.array(shape['points'], np.int32)
#         cv2.polylines(img, [points], True, COLORS.get(class_id, (0, 0, 255)), 2)
#         # Draw bbox approximation for comparison
#         x_min, y_min = np.min(points, axis=0)
#         x_max, y_max = np.max(points, axis=0)
#         cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), COLORS.get(class_id, (0, 0, 255)), 1)
#         cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def draw_yolo_bboxes_on_patch(img, label_path):
#     """Draw YOLO bboxes on patch image."""
#     h, w = img.shape[:2]
#     if not label_path.exists():
#         print(f"No label file at {label_path} (background patch)")
#         return img
#     with open(label_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue
#             class_id = int(parts[0])
#             cx, cy, bw, bh = map(float, parts[1:])
#             x_min = int((cx - bw / 2) * w)
#             y_min = int((cy - bh / 2) * h)
#             x_max = int((cx + bw / 2) * w)
#             y_max = int((cy + bh / 2) * h)
#             cv2.rectangle(img, (x_min, y_min), (x_max, y_max), COLORS.get(class_id, (0, 0, 255)), 2)
#             cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def create_overview_montage(original_annotated, patch_annotated_list, base_name, patch_files):
#     """Create a single overview image with original on top and patches in grid below, with labels below images."""
#     # Resize original to match approx width (5 patches wide)
#     patch_size = patch_annotated_list[0].shape[0] if patch_annotated_list else 320
#     label_height = 30  # Space for labels below each image
#     original_resized = cv2.resize(original_annotated, (patch_size * 5, int(original_annotated.shape[0] * (patch_size * 5 / original_annotated.shape[1]))))

#     # Add label for original (below)
#     original_canvas_height = original_resized.shape[0] + label_height
#     original_canvas = np.zeros((original_canvas_height, original_resized.shape[1], 3), dtype=np.uint8)
#     original_canvas[:original_resized.shape[0], :] = original_resized
#     cv2.putText(original_canvas, f'Original Annotated: {base_name}', (10, original_resized.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     # Create grid for patches with label space
#     num_patches = len(patch_annotated_list)
#     cols = 5
#     rows = int(np.ceil(num_patches / cols))
#     montage_height = rows * (patch_size + label_height)
#     montage_width = cols * patch_size
#     montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)

#     for i, (patch, patch_file) in enumerate(zip(patch_annotated_list, patch_files)):
#         row = i // cols
#         col = i % cols
#         # Place patch
#         montage[row * (patch_size + label_height):row * (patch_size + label_height) + patch_size, col * patch_size:(col + 1) * patch_size] = patch
#         # Add label below
#         label_y = row * (patch_size + label_height) + patch_size + 20
#         cv2.putText(montage, f'Patch: {patch_file.stem}', (col * patch_size + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     # Stack original on top
#     full_montage = np.vstack((original_canvas, montage)) if num_patches > 0 else original_canvas

#     overview_path = VISUALIZATIONS_DIR / f'{base_name}_overview.jpg'  # Save in visualizations folder
#     cv2.imwrite(str(overview_path), full_montage)
#     print(f"Overview saved to {overview_path}")

#     # Display and wait for any key press or close (blocking)
#     window_name = 'Overview Montage'
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow resizing
#     cv2.imshow(window_name, full_montage)
#     print("Press any key or close the window to continue from montage.")
#     while True:
#         key = cv2.waitKey(0)  # Block until key press
#         if key != -1 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
#             break
#     cv2.destroyAllWindows()

# def interactive_flip_through_patches(patch_annotated_list, patch_files):
#     """Interactive high-res flipping through patches with OpenCV."""
#     if not patch_annotated_list:
#         return
#     idx = 0
#     window_name = 'Interactive Patch Viewer'
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow resizing
#     print("Interactive mode: Click on the window to focus it. Press 'n'/right arrow for next, 'p'/left arrow for previous, any other key or ESC/close window to exit.")
#     while True:
#         img = patch_annotated_list[idx]
#         cv2.imshow(window_name, img)
#         cv2.setWindowTitle(window_name, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
#         key = cv2.waitKey(0)  # Block until key press
#         if key == 27 or key == -1 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:  # ESC, no key, or close
#             break
#         elif key == ord('n') or key == 83:  # 'n' or right arrow
#             idx = (idx + 1) % len(patch_annotated_list)
#         elif key == ord('p') or key == 81:  # 'p' or left arrow
#             idx = (idx - 1) % len(patch_annotated_list)
#         # Any other key also exits, as per user request for proper exit
#         else:
#             break
#     cv2.destroyAllWindows()

# def visualize_example(base_name, split='train'):
#     """Updated visualization: overview montage and interactive flipping."""
#     tif_path, json_path = find_original_files(base_name)
#     if not tif_path or not json_path:
#         return

#     # Load and annotate original
#     original = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
#     if original is None:
#         print(f"Error loading {tif_path}")
#         return
#     if len(original.shape) == 2:
#         original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
#     original_annotated = draw_annotations_on_original(original.copy(), json_path)

#     # Find all matching patches
#     images_dir = PROCESSED_DIR / split / 'images'
#     labels_dir = PROCESSED_DIR / split / 'labels'
#     patch_files = sorted([f for f in images_dir.glob('*.jpg') if f.stem.startswith(f'{base_name}_')])
#     print(f"Found {len(patch_files)} patches for {base_name} in {split}")

#     if not patch_files:
#         return

#     # Load and annotate all patches
#     patch_annotated_list = []
#     for patch_path in patch_files:
#         label_path = labels_dir / f'{patch_path.stem}.txt'
#         patch_img = cv2.imread(str(patch_path))
#         if patch_img is None:
#             print(f"Error loading {patch_path}")
#             continue
#         patch_annotated = draw_yolo_bboxes_on_patch(patch_img.copy(), label_path)
#         patch_annotated_list.append(patch_annotated)

#     # Create and show/save overview montage
#     create_overview_montage(original_annotated, patch_annotated_list, base_name, patch_files)

#     # Interactive high-res flipping
#     interactive_flip_through_patches(patch_annotated_list, patch_files)

# # Setup visualizations dir (empties on each run)
# setup_visualizations_dir()

# # Run for examples
# for base_name in EXAMPLE_BASE_NAMES:
#     visualize_example(base_name, SPLIT)


# Best so far:
# import cv2
# import json
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import shutil

# # Config (adapt to your paths)
# BASE_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data')
# PROCESSED_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\processed_balanced')
# VISUALIZATIONS_DIR = PROCESSED_DIR / 'visualizations'  # New folder for visualizations
# EXAMPLE_BASE_NAMES = ['A_bam5']  # Add more if needed
# SPLIT = 'train'  # Check train/val/test as needed

# # Class names and colors (BGR for OpenCV)
# CLASS_NAMES = {0: 'porosity', 1: 'inclusion', 2: 'crack', 3: 'undercut', 4: 'lack_of_fusion', 5: 'lack_of_penetration'}
# COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255)}

# def setup_visualizations_dir():
#     """Create or empty the visualizations directory."""
#     if VISUALIZATIONS_DIR.exists():
#         shutil.rmtree(VISUALIZATIONS_DIR)  # Empty the folder
#     VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
#     print(f"Visualizations directory setup at {VISUALIZATIONS_DIR}")

# def find_original_files(base_name):
#     """Find TIF and JSON for original, with logging."""
#     tif_path = None
#     json_path = None
#     for subdir in ['crop_weld_images/L/1', 'crop_weld_images/L/2', 'crop_weld_images/T/1', 'crop_weld_images/T/2']:
#         candidate_tif = BASE_DIR / subdir / f'{base_name}.tif'
#         if candidate_tif.exists():
#             tif_path = candidate_tif
#             print(f"Found TIF: {tif_path}")
#             break
#     for subdir in ['crop_weld_jsons/L/1', 'crop_weld_jsons/L/2', 'crop_weld_jsons/T/1', 'crop_weld_jsons/T/2']:
#         candidate_json = BASE_DIR / subdir / f'{base_name}.json'
#         if candidate_json.exists():
#             json_path = candidate_json
#             print(f"Found JSON: {json_path}")
#             break
#     if not tif_path or not json_path:
#         print(f"Warning: Files not found for {base_name}. Check prefixes (e.g., 'A_') or subdirs.")
#     return tif_path, json_path

# def draw_annotations_on_original(img, json_path):
#     """Draw JSON polygons/bboxes on original image."""
#     with open(json_path, 'r', encoding='utf-8') as f:
#         ann_data = json.load(f)
#     for shape in ann_data.get('shapes', []):
#         label = shape.get('label', '')
#         class_id = {  # Your class_map from preprocessing script
#             '\u6c14\u5b54': 0, '气孔': 0,
#             '\u5939\u6e23': 1, '夹渣': 1,
#             '\u88c2\u7eb9': 2, '裂纹': 2,
#             '\u54ac\u8fb9': 3, '咬边': 3,
#             '\u672a\u878d\u5408': 4, '未熔合': 4,
#             '\u672a\u710a\u900f': 5, '未焊透': 5,
#             '内凹': 3,
#             '夹钨': 1,
#         }.get(label, -1)
#         if class_id == -1:
#             continue
#         points = np.array(shape['points'], np.int32)
#         cv2.polylines(img, [points], True, COLORS.get(class_id, (0, 0, 255)), 2)
#         # Draw bbox approximation for comparison
#         x_min, y_min = np.min(points, axis=0)
#         x_max, y_max = np.max(points, axis=0)
#         cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), COLORS.get(class_id, (0, 0, 255)), 1)
#         cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def draw_yolo_bboxes_on_patch(img, label_path):
#     """Draw YOLO bboxes on patch image."""
#     h, w = img.shape[:2]
#     if not label_path.exists():
#         print(f"No label file at {label_path} (background patch)")
#         return img
#     with open(label_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue
#             class_id = int(parts[0])
#             cx, cy, bw, bh = map(float, parts[1:])
#             x_min = int((cx - bw / 2) * w)
#             y_min = int((cy - bh / 2) * h)
#             x_max = int((cx + bw / 2) * w)
#             y_max = int((cy + bh / 2) * h)
#             cv2.rectangle(img, (x_min, y_min), (x_max, y_max), COLORS.get(class_id, (0, 0, 255)), 2)
#             cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def create_overview_montage(original_annotated, patch_annotated_list, base_name, patch_files):
#     """Create a single overview image with original on top and patches in grid below, with labels below images."""
#     # Resize original to match approx width (5 patches wide)
#     patch_size = patch_annotated_list[0].shape[0] if patch_annotated_list else 320
#     label_height = 30  # Space for labels below each image
#     original_resized = cv2.resize(original_annotated, (patch_size * 5, int(original_annotated.shape[0] * (patch_size * 5 / original_annotated.shape[1]))))

#     # Add label for original (below)
#     original_canvas_height = original_resized.shape[0] + label_height
#     original_canvas = np.zeros((original_canvas_height, original_resized.shape[1], 3), dtype=np.uint8)
#     original_canvas[:original_resized.shape[0], :] = original_resized
#     cv2.putText(original_canvas, f'Original Annotated: {base_name}', (10, original_resized.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     # Create grid for patches with label space
#     num_patches = len(patch_annotated_list)
#     cols = 5
#     rows = int(np.ceil(num_patches / cols))
#     montage_height = rows * (patch_size + label_height)
#     montage_width = cols * patch_size
#     montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)

#     for i, (patch, patch_file) in enumerate(zip(patch_annotated_list, patch_files)):
#         row = i // cols
#         col = i % cols
#         # Place patch
#         montage[row * (patch_size + label_height):row * (patch_size + label_height) + patch_size, col * patch_size:(col + 1) * patch_size] = patch
#         # Add label below
#         label_y = row * (patch_size + label_height) + patch_size + 20
#         cv2.putText(montage, f'Patch: {patch_file.stem}', (col * patch_size + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     # Stack original on top
#     full_montage = np.vstack((original_canvas, montage)) if num_patches > 0 else original_canvas

#     overview_path = VISUALIZATIONS_DIR / f'{base_name}_overview.jpg'  # Save in visualizations folder
#     cv2.imwrite(str(overview_path), full_montage)
#     print(f"Overview saved to {overview_path}")

#     # Display with loop for key and close detection
#     window_name = 'Overview Montage'
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow resizing
#     cv2.imshow(window_name, full_montage)
#     print("Montage window open. Focus the window and press any key to continue. Closing the window will also continue.")
#     while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
#         key = cv2.waitKey(1)
#         if key != -1:
#             break
#     cv2.destroyAllWindows()

# def interactive_flip_through_patches(patch_annotated_list, patch_files):
#     """Interactive high-res flipping through patches with OpenCV."""
#     if not patch_annotated_list:
#         return
#     idx = 0
#     window_name = 'Interactive Patch Viewer'
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow resizing
#     print("Interactive mode started. Focus the window! Press 'n' or right arrow for next, 'p' or left arrow for previous, ESC or any other key or close window to exit.")
#     while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
#         img = patch_annotated_list[idx]
#         cv2.imshow(window_name, img)
#         cv2.setWindowTitle(window_name, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
#         key = cv2.waitKey(1)
#         if key == 27 or key != -1:  # ESC or any key press
#             break
#         elif key == ord('n') or key == 83:  # 'n' or right arrow
#             idx = (idx + 1) % len(patch_annotated_list)
#         elif key == ord('p') or key == 81:  # 'p' or left arrow
#             idx = (idx - 1) % len(patch_annotated_list)
#     cv2.destroyAllWindows()

# def visualize_example(base_name, split='train'):
#     """Updated visualization: overview montage and interactive flipping."""
#     tif_path, json_path = find_original_files(base_name)
#     if not tif_path or not json_path:
#         return

#     # Load and annotate original
#     original = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
#     if original is None:
#         print(f"Error loading {tif_path}")
#         return
#     if len(original.shape) == 2:
#         original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
#     original_annotated = draw_annotations_on_original(original.copy(), json_path)

#     # Find all matching patches
#     images_dir = PROCESSED_DIR / split / 'images'
#     labels_dir = PROCESSED_DIR / split / 'labels'
#     patch_files = sorted([f for f in images_dir.glob('*.jpg') if f.stem.startswith(f'{base_name}_')])
#     print(f"Found {len(patch_files)} patches for {base_name} in {split}")

#     if not patch_files:
#         return

#     # Load and annotate all patches
#     patch_annotated_list = []
#     for patch_path in patch_files:
#         label_path = labels_dir / f'{patch_path.stem}.txt'
#         patch_img = cv2.imread(str(patch_path))
#         if patch_img is None:
#             print(f"Error loading {patch_path}")
#             continue
#         patch_annotated = draw_yolo_bboxes_on_patch(patch_img.copy(), label_path)
#         patch_annotated_list.append(patch_annotated)

#     # Create and show/save overview montage
#     create_overview_montage(original_annotated, patch_annotated_list, base_name, patch_files)

#     # Interactive high-res flipping
#     interactive_flip_through_patches(patch_annotated_list, patch_files)

# # Setup visualizations dir (empties on each run)
# setup_visualizations_dir()

# # Run for examples
# for base_name in EXAMPLE_BASE_NAMES:
#     visualize_example(base_name, SPLIT)



# Claude corrected: Nav works:

# import cv2
# import json
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import shutil

# # Config (adapt to your paths)
# BASE_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data')
# PROCESSED_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\processed_balanced')
# VISUALIZATIONS_DIR = PROCESSED_DIR / 'visualizations'  # New folder for visualizations
# EXAMPLE_BASE_NAMES = ['A_bam5']  # Add more if needed
# SPLIT = 'train'  # Check train/val/test as needed

# # Class names and colors (BGR for OpenCV)
# CLASS_NAMES = {0: 'porosity', 1: 'inclusion', 2: 'crack', 3: 'undercut', 4: 'lack_of_fusion', 5: 'lack_of_penetration'}
# COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255)}

# def setup_visualizations_dir():
#     """Create or empty the visualizations directory."""
#     if VISUALIZATIONS_DIR.exists():
#         shutil.rmtree(VISUALIZATIONS_DIR)  # Empty the folder
#     VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
#     print(f"Visualizations directory setup at {VISUALIZATIONS_DIR}")

# def find_original_files(base_name):
#     """Find TIF and JSON for original, with logging."""
#     tif_path = None
#     json_path = None
#     for subdir in ['crop_weld_images/L/1', 'crop_weld_images/L/2', 'crop_weld_images/T/1', 'crop_weld_images/T/2']:
#         candidate_tif = BASE_DIR / subdir / f'{base_name}.tif'
#         if candidate_tif.exists():
#             tif_path = candidate_tif
#             print(f"Found TIF: {tif_path}")
#             break
#     for subdir in ['crop_weld_jsons/L/1', 'crop_weld_jsons/L/2', 'crop_weld_jsons/T/1', 'crop_weld_jsons/T/2']:
#         candidate_json = BASE_DIR / subdir / f'{base_name}.json'
#         if candidate_json.exists():
#             json_path = candidate_json
#             print(f"Found JSON: {json_path}")
#             break
#     if not tif_path or not json_path:
#         print(f"Warning: Files not found for {base_name}. Check prefixes (e.g., 'A_') or subdirs.")
#     return tif_path, json_path

# def draw_annotations_on_original(img, json_path):
#     """Draw JSON polygons/bboxes on original image."""
#     with open(json_path, 'r', encoding='utf-8') as f:
#         ann_data = json.load(f)
#     for shape in ann_data.get('shapes', []):
#         label = shape.get('label', '')
#         class_id = {  # Your class_map from preprocessing script
#             '\u6c14\u5b54': 0, '气孔': 0,
#             '\u5939\u6e23': 1, '夹渣': 1,
#             '\u88c2\u7eb9': 2, '裂纹': 2,
#             '\u54ac\u8fb9': 3, '咬边': 3,
#             '\u672a\u878d\u5408': 4, '未熔合': 4,
#             '\u672a\u710a\u900f': 5, '未焊透': 5,
#             '内凹': 3,
#             '夹钨': 1,
#         }.get(label, -1)
#         if class_id == -1:
#             continue
#         points = np.array(shape['points'], np.int32)
#         cv2.polylines(img, [points], True, COLORS.get(class_id, (0, 0, 255)), 2)
#         # Draw bbox approximation for comparison
#         x_min, y_min = np.min(points, axis=0)
#         x_max, y_max = np.max(points, axis=0)
#         cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), COLORS.get(class_id, (0, 0, 255)), 1)
#         cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def draw_yolo_bboxes_on_patch(img, label_path):
#     """Draw YOLO bboxes on patch image."""
#     h, w = img.shape[:2]
#     if not label_path.exists():
#         print(f"No label file at {label_path} (background patch)")
#         return img
#     with open(label_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue
#             class_id = int(parts[0])
#             cx, cy, bw, bh = map(float, parts[1:])
#             x_min = int((cx - bw / 2) * w)
#             y_min = int((cy - bh / 2) * h)
#             x_max = int((cx + bw / 2) * w)
#             y_max = int((cy + bh / 2) * h)
#             cv2.rectangle(img, (x_min, y_min), (x_max, y_max), COLORS.get(class_id, (0, 0, 255)), 2)
#             cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def create_overview_montage(original_annotated, patch_annotated_list, base_name, patch_files):
#     """Create a single overview image with original on top and patches in grid below, with labels below images."""
#     # Resize original to match approx width (5 patches wide)
#     patch_size = patch_annotated_list[0].shape[0] if patch_annotated_list else 320
#     label_height = 30  # Space for labels below each image
#     original_resized = cv2.resize(original_annotated, (patch_size * 5, int(original_annotated.shape[0] * (patch_size * 5 / original_annotated.shape[1]))))

#     # Add label for original (below)
#     original_canvas_height = original_resized.shape[0] + label_height
#     original_canvas = np.zeros((original_canvas_height, original_resized.shape[1], 3), dtype=np.uint8)
#     original_canvas[:original_resized.shape[0], :] = original_resized
#     cv2.putText(original_canvas, f'Original Annotated: {base_name}', (10, original_resized.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     # Create grid for patches with label space
#     num_patches = len(patch_annotated_list)
#     cols = 5
#     rows = int(np.ceil(num_patches / cols))
#     montage_height = rows * (patch_size + label_height)
#     montage_width = cols * patch_size
#     montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)

#     for i, (patch, patch_file) in enumerate(zip(patch_annotated_list, patch_files)):
#         row = i // cols
#         col = i % cols
#         # Place patch
#         montage[row * (patch_size + label_height):row * (patch_size + label_height) + patch_size, col * patch_size:(col + 1) * patch_size] = patch
#         # Add label below
#         label_y = row * (patch_size + label_height) + patch_size + 20
#         cv2.putText(montage, f'Patch: {patch_file.stem}', (col * patch_size + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     # Stack original on top
#     full_montage = np.vstack((original_canvas, montage)) if num_patches > 0 else original_canvas

#     overview_path = VISUALIZATIONS_DIR / f'{base_name}_overview.jpg'  # Save in visualizations folder
#     cv2.imwrite(str(overview_path), full_montage)
#     print(f"Overview saved to {overview_path}")

#     # Display with loop for key and close detection
#     window_name = 'Overview Montage'
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow resizing
#     cv2.imshow(window_name, full_montage)
#     print("Montage window open. Focus the window and press any key to continue. Closing the window will also continue.")
#     while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
#         key = cv2.waitKey(1)
#         if key != -1:
#             break
#     cv2.destroyAllWindows()

# def interactive_flip_through_patches(patch_annotated_list, patch_files):
#     """Interactive high-res flipping through patches with OpenCV."""
#     if not patch_annotated_list:
#         return
    
#     idx = 0
#     window_name = 'Interactive Patch Viewer'
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow resizing
    
#     print("\n" + "="*60)
#     print("Interactive mode started. Focus the window!")
#     print("Controls:")
#     print("  'n': Next patch")
#     print("  'p': Previous patch")
#     print("  'd' or Right Arrow: Next patch")
#     print("  'a' or Left Arrow: Previous patch")
#     print("  'q' or ESC: Quit")
#     print("  Close window to exit")
#     print("="*60 + "\n")
    
#     # Display first image
#     cv2.imshow(window_name, patch_annotated_list[idx])
#     cv2.setWindowTitle(window_name, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
    
#     while True:
#         # Check if window is still open (this might return -1 if window is closed)
#         try:
#             if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
#                 print("Window closed by user")
#                 break
#         except:
#             print("Window no longer exists")
#             break
            
#         # Wait for key with short timeout
#         key = cv2.waitKey(30)  # 30ms timeout
        
#         # Only process if a key was actually pressed
#         if key != -1:
#             # Convert to ASCII if needed (handle special keys)
#             key_char = key & 0xFF
            
#             # Debug: show what key was pressed
#             if key_char < 127 and key_char > 32:  # Printable character
#                 print(f"Key pressed: '{chr(key_char)}' (code: {key})")
#             else:
#                 print(f"Special key pressed (code: {key})")
            
#             # Handle key presses
#             if key_char == 27 or key_char == ord('q'):  # ESC or 'q'
#                 print("Quit command received")
#                 break
#             elif key_char == ord('n') or key_char == ord('d') or key == 2555904:  # 'n', 'd' or right arrow
#                 idx = (idx + 1) % len(patch_annotated_list)
#                 print(f"Moving to next patch: {idx+1}/{len(patch_annotated_list)}")
#                 cv2.imshow(window_name, patch_annotated_list[idx])
#                 cv2.setWindowTitle(window_name, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
#             elif key_char == ord('p') or key_char == ord('a') or key == 2424832:  # 'p', 'a' or left arrow
#                 idx = (idx - 1) % len(patch_annotated_list)
#                 print(f"Moving to previous patch: {idx+1}/{len(patch_annotated_list)}")
#                 cv2.imshow(window_name, patch_annotated_list[idx])
#                 cv2.setWindowTitle(window_name, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
#             # For arrow keys on Windows, the codes might be different
#             elif key == 83 or key == 77:  # Possible right arrow codes
#                 idx = (idx + 1) % len(patch_annotated_list)
#                 print(f"Moving to next patch (arrow): {idx+1}/{len(patch_annotated_list)}")
#                 cv2.imshow(window_name, patch_annotated_list[idx])
#                 cv2.setWindowTitle(window_name, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
#             elif key == 81 or key == 75:  # Possible left arrow codes
#                 idx = (idx - 1) % len(patch_annotated_list)
#                 print(f"Moving to previous patch (arrow): {idx+1}/{len(patch_annotated_list)}")
#                 cv2.imshow(window_name, patch_annotated_list[idx])
#                 cv2.setWindowTitle(window_name, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
            
#     cv2.destroyAllWindows()
#     print("Interactive viewer closed")

# def visualize_example(base_name, split='train'):
#     """Updated visualization: overview montage and interactive flipping."""
#     tif_path, json_path = find_original_files(base_name)
#     if not tif_path or not json_path:
#         return

#     # Load and annotate original
#     original = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
#     if original is None:
#         print(f"Error loading {tif_path}")
#         return
#     if len(original.shape) == 2:
#         original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
#     original_annotated = draw_annotations_on_original(original.copy(), json_path)

#     # Find all matching patches
#     images_dir = PROCESSED_DIR / split / 'images'
#     labels_dir = PROCESSED_DIR / split / 'labels'
#     patch_files = sorted([f for f in images_dir.glob('*.jpg') if f.stem.startswith(f'{base_name}_')])
#     print(f"Found {len(patch_files)} patches for {base_name} in {split}")

#     if not patch_files:
#         return

#     # Load and annotate all patches
#     patch_annotated_list = []
#     for patch_path in patch_files:
#         label_path = labels_dir / f'{patch_path.stem}.txt'
#         patch_img = cv2.imread(str(patch_path))
#         if patch_img is None:
#             print(f"Error loading {patch_path}")
#             continue
#         patch_annotated = draw_yolo_bboxes_on_patch(patch_img.copy(), label_path)
#         patch_annotated_list.append(patch_annotated)

#     # Create and show/save overview montage
#     create_overview_montage(original_annotated, patch_annotated_list, base_name, patch_files)

#     # Interactive high-res flipping
#     interactive_flip_through_patches(patch_annotated_list, patch_files)

# # Setup visualizations dir (empties on each run)
# setup_visualizations_dir()

# # Run for examples
# for base_name in EXAMPLE_BASE_NAMES:
#     visualize_example(base_name, SPLIT)

# Claude Follow up:
# import cv2
# import json
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import shutil

# # Config (adapt to your paths)
# BASE_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data')
# PROCESSED_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\processed_balanced')
# VISUALIZATIONS_DIR = PROCESSED_DIR / 'visualizations'  # New folder for visualizations
# EXAMPLE_BASE_NAMES = ['A_bam5']  # Add more if needed
# SPLIT = 'train'  # Check train/val/test as needed

# # Class names and colors (BGR for OpenCV)
# CLASS_NAMES = {0: 'porosity', 1: 'inclusion', 2: 'crack', 3: 'undercut', 4: 'lack_of_fusion', 5: 'lack_of_penetration'}
# COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255)}

# def setup_visualizations_dir():
#     """Create or empty the visualizations directory."""
#     if VISUALIZATIONS_DIR.exists():
#         shutil.rmtree(VISUALIZATIONS_DIR)  # Empty the folder
#     VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
#     print(f"Visualizations directory setup at {VISUALIZATIONS_DIR}")

# def find_original_files(base_name):
#     """Find TIF and JSON for original, with logging."""
#     tif_path = None
#     json_path = None
#     for subdir in ['crop_weld_images/L/1', 'crop_weld_images/L/2', 'crop_weld_images/T/1', 'crop_weld_images/T/2']:
#         candidate_tif = BASE_DIR / subdir / f'{base_name}.tif'
#         if candidate_tif.exists():
#             tif_path = candidate_tif
#             print(f"Found TIF: {tif_path}")
#             break
#     for subdir in ['crop_weld_jsons/L/1', 'crop_weld_jsons/L/2', 'crop_weld_jsons/T/1', 'crop_weld_jsons/T/2']:
#         candidate_json = BASE_DIR / subdir / f'{base_name}.json'
#         if candidate_json.exists():
#             json_path = candidate_json
#             print(f"Found JSON: {json_path}")
#             break
#     if not tif_path or not json_path:
#         print(f"Warning: Files not found for {base_name}. Check prefixes (e.g., 'A_') or subdirs.")
#     return tif_path, json_path

# def draw_annotations_on_original(img, json_path):
#     """Draw JSON polygons/bboxes on original image."""
#     with open(json_path, 'r', encoding='utf-8') as f:
#         ann_data = json.load(f)
#     for shape in ann_data.get('shapes', []):
#         label = shape.get('label', '')
#         class_id = {  # Your class_map from preprocessing script
#             '\u6c14\u5b54': 0, '气孔': 0,
#             '\u5939\u6e23': 1, '夹渣': 1,
#             '\u88c2\u7eb9': 2, '裂纹': 2,
#             '\u54ac\u8fb9': 3, '咬边': 3,
#             '\u672a\u878d\u5408': 4, '未熔合': 4,
#             '\u672a\u710a\u900f': 5, '未焊透': 5,
#             '内凹': 3,
#             '夹钨': 1,
#         }.get(label, -1)
#         if class_id == -1:
#             continue
#         points = np.array(shape['points'], np.int32)
#         cv2.polylines(img, [points], True, COLORS.get(class_id, (0, 0, 255)), 2)
#         # Draw bbox approximation for comparison
#         x_min, y_min = np.min(points, axis=0)
#         x_max, y_max = np.max(points, axis=0)
#         cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), COLORS.get(class_id, (0, 0, 255)), 1)
#         cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def draw_yolo_bboxes_on_patch(img, label_path):
#     """Draw YOLO bboxes on patch image."""
#     h, w = img.shape[:2]
#     if not label_path.exists():
#         print(f"No label file at {label_path} (background patch)")
#         return img
#     with open(label_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue
#             class_id = int(parts[0])
#             cx, cy, bw, bh = map(float, parts[1:])
#             x_min = int((cx - bw / 2) * w)
#             y_min = int((cy - bh / 2) * h)
#             x_max = int((cx + bw / 2) * w)
#             y_max = int((cy + bh / 2) * h)
#             cv2.rectangle(img, (x_min, y_min), (x_max, y_max), COLORS.get(class_id, (0, 0, 255)), 2)
#             cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def create_overview_montage(original_annotated, patch_annotated_list, base_name, patch_files):
#     """Create a single overview image with original on top and patches in grid below, with labels below images."""
#     # Resize original to match approx width (5 patches wide)
#     patch_size = patch_annotated_list[0].shape[0] if patch_annotated_list else 320
#     label_height = 30  # Space for labels below each image
#     original_resized = cv2.resize(original_annotated, (patch_size * 5, int(original_annotated.shape[0] * (patch_size * 5 / original_annotated.shape[1]))))

#     # Add label for original (below)
#     original_canvas_height = original_resized.shape[0] + label_height
#     original_canvas = np.zeros((original_canvas_height, original_resized.shape[1], 3), dtype=np.uint8)
#     original_canvas[:original_resized.shape[0], :] = original_resized
#     cv2.putText(original_canvas, f'Original Annotated: {base_name}', (10, original_resized.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     # Create grid for patches with label space
#     num_patches = len(patch_annotated_list)
#     cols = 5
#     rows = int(np.ceil(num_patches / cols))
#     montage_height = rows * (patch_size + label_height)
#     montage_width = cols * patch_size
#     montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)

#     for i, (patch, patch_file) in enumerate(zip(patch_annotated_list, patch_files)):
#         row = i // cols
#         col = i % cols
#         # Place patch
#         montage[row * (patch_size + label_height):row * (patch_size + label_height) + patch_size, col * patch_size:(col + 1) * patch_size] = patch
#         # Add label below
#         label_y = row * (patch_size + label_height) + patch_size + 20
#         cv2.putText(montage, f'Patch: {patch_file.stem}', (col * patch_size + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     # Stack original on top
#     full_montage = np.vstack((original_canvas, montage)) if num_patches > 0 else original_canvas

#     overview_path = VISUALIZATIONS_DIR / f'{base_name}_overview.jpg'  # Save in visualizations folder
#     cv2.imwrite(str(overview_path), full_montage)
#     print(f"Overview saved to {overview_path}")

#     # Display with loop for key and close detection
#     window_name = 'Overview Montage'
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow resizing
#     cv2.imshow(window_name, full_montage)
#     print("Montage window open. Focus the window and press any key to continue. Closing the window will also continue.")
#     while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
#         key = cv2.waitKey(1)
#         if key != -1:
#             break
#     cv2.destroyAllWindows()

# def interactive_flip_through_patches(patch_annotated_list, patch_files):
#     """Interactive high-res flipping through patches with OpenCV."""
#     if not patch_annotated_list:
#         return
    
#     idx = 0
#     window_name = 'Interactive Patch Viewer'
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow resizing
    
#     print("\n" + "="*60)
#     print("Interactive mode started. Focus the window!")
#     print("Controls:")
#     print("  'n': Next patch")
#     print("  'p': Previous patch")
#     print("  'd' or Right Arrow: Next patch")
#     print("  'a' or Left Arrow: Previous patch")
#     print("  'q' or ESC: Quit")
#     print("  Close window to exit")
#     print("="*60 + "\n")
    
#     # Display first image
#     cv2.imshow(window_name, patch_annotated_list[idx])
#     cv2.setWindowTitle(window_name, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
    
#     while True:
#         # Check if window is still open (this might return -1 if window is closed)
#         try:
#             if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
#                 print("Window closed by user")
#                 break
#         except:
#             print("Window no longer exists")
#             break
            
#         # Wait for key with short timeout
#         key = cv2.waitKey(30)  # 30ms timeout
        
#         # Only process if a key was actually pressed
#         if key != -1:
#             # Convert to ASCII if needed (handle special keys)
#             key_char = key & 0xFF
            
#             # Debug: show what key was pressed
#             if key_char < 127 and key_char > 32:  # Printable character
#                 print(f"Key pressed: '{chr(key_char)}' (code: {key})")
#             else:
#                 print(f"Special key pressed (code: {key})")
            
#             # Handle key presses
#             if key_char == 27 or key_char == ord('q'):  # ESC or 'q'
#                 print("Quit command received")
#                 break
#             elif key_char == ord('n') or key_char == ord('d') or key == 2555904:  # 'n', 'd' or right arrow
#                 idx = (idx + 1) % len(patch_annotated_list)
#                 print(f"Moving to next patch: {idx+1}/{len(patch_annotated_list)}")
#                 cv2.imshow(window_name, patch_annotated_list[idx])
#                 cv2.setWindowTitle(window_name, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
#             elif key_char == ord('p') or key_char == ord('a') or key == 2424832:  # 'p', 'a' or left arrow
#                 idx = (idx - 1) % len(patch_annotated_list)
#                 print(f"Moving to previous patch: {idx+1}/{len(patch_annotated_list)}")
#                 cv2.imshow(window_name, patch_annotated_list[idx])
#                 cv2.setWindowTitle(window_name, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
#             # For arrow keys on Windows, the codes might be different
#             elif key == 83 or key == 77:  # Possible right arrow codes
#                 idx = (idx + 1) % len(patch_annotated_list)
#                 print(f"Moving to next patch (arrow): {idx+1}/{len(patch_annotated_list)}")
#                 cv2.imshow(window_name, patch_annotated_list[idx])
#                 cv2.setWindowTitle(window_name, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
#             elif key == 81 or key == 75:  # Possible left arrow codes
#                 idx = (idx - 1) % len(patch_annotated_list)
#                 print(f"Moving to previous patch (arrow): {idx+1}/{len(patch_annotated_list)}")
#                 cv2.imshow(window_name, patch_annotated_list[idx])
#                 cv2.setWindowTitle(window_name, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
            
#     cv2.destroyAllWindows()
#     print("Interactive viewer closed")

# def visualize_example(base_name, split='train'):
#     """Updated visualization: overview montage and interactive flipping."""
#     tif_path, json_path = find_original_files(base_name)
#     if not tif_path or not json_path:
#         return

#     # Load and annotate original
#     original = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
#     if original is None:
#         print(f"Error loading {tif_path}")
#         return
#     if len(original.shape) == 2:
#         original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
#     original_annotated = draw_annotations_on_original(original.copy(), json_path)

#     # Find all matching patches
#     images_dir = PROCESSED_DIR / split / 'images'
#     labels_dir = PROCESSED_DIR / split / 'labels'
#     patch_files = sorted([f for f in images_dir.glob('*.jpg') if f.stem.startswith(f'{base_name}_')])
#     print(f"Found {len(patch_files)} patches for {base_name} in {split}")

#     if not patch_files:
#         return

#     # Load and annotate all patches
#     patch_annotated_list = []
#     for patch_path in patch_files:
#         label_path = labels_dir / f'{patch_path.stem}.txt'
#         patch_img = cv2.imread(str(patch_path))
#         if patch_img is None:
#             print(f"Error loading {patch_path}")
#             continue
#         patch_annotated = draw_yolo_bboxes_on_patch(patch_img.copy(), label_path)
#         patch_annotated_list.append(patch_annotated)

#     # Create and show/save overview montage
#     create_overview_montage(original_annotated, patch_annotated_list, base_name, patch_files)

#     # Interactive high-res flipping
#     interactive_flip_through_patches(patch_annotated_list, patch_files)

# # Setup visualizations dir (empties on each run)
# setup_visualizations_dir()

# # Run for examples
# for base_name in EXAMPLE_BASE_NAMES:
#     visualize_example(base_name, SPLIT)


# Dual Mode (Verification and Validation):
# import cv2
# import json
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import shutil
# import atexit  # For cleanup on exit

# # Config (adapt to your paths)
# BASE_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data')
# PROCESSED_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\processed_balanced')
# VISUALIZATIONS_DIR = PROCESSED_DIR / 'visualizations'  # Temporary folder for visualizations
# EXAMPLE_BASE_NAMES = ['A_bam5']  # Add more if needed
# SPLIT = 'train'  # Check train/val/test as needed

# # Class names and colors (BGR for OpenCV)
# CLASS_NAMES = {0: 'porosity', 1: 'inclusion', 2: 'crack', 3: 'undercut', 4: 'lack_of_fusion', 5: 'lack_of_penetration'}
# COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255)}

# def cleanup_visualizations():
#     """Clean up the visualizations directory on exit."""
#     if VISUALIZATIONS_DIR.exists():
#         print(f"\nCleaning up temporary visualizations at {VISUALIZATIONS_DIR}")
#         shutil.rmtree(VISUALIZATIONS_DIR)
#         print("Cleanup complete - all temporary visualization files removed")

# # Register cleanup to run on script exit
# atexit.register(cleanup_visualizations)

# def setup_visualizations_dir():
#     """Create or empty the visualizations directory."""
#     if VISUALIZATIONS_DIR.exists():
#         shutil.rmtree(VISUALIZATIONS_DIR)  # Empty the folder
#     VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
#     print(f"Temporary visualizations directory created at {VISUALIZATIONS_DIR}")
#     print("NOTE: This folder will be automatically deleted when script ends\n")

# def find_original_files(base_name):
#     """Find TIF and JSON for original, with logging."""
#     tif_path = None
#     json_path = None
#     for subdir in ['crop_weld_images/L/1', 'crop_weld_images/L/2', 'crop_weld_images/T/1', 'crop_weld_images/T/2']:
#         candidate_tif = BASE_DIR / subdir / f'{base_name}.tif'
#         if candidate_tif.exists():
#             tif_path = candidate_tif
#             print(f"Found TIF: {tif_path}")
#             break
#     for subdir in ['crop_weld_jsons/L/1', 'crop_weld_jsons/L/2', 'crop_weld_jsons/T/1', 'crop_weld_jsons/T/2']:
#         candidate_json = BASE_DIR / subdir / f'{base_name}.json'
#         if candidate_json.exists():
#             json_path = candidate_json
#             print(f"Found JSON: {json_path}")
#             break
#     if not tif_path or not json_path:
#         print(f"Warning: Files not found for {base_name}. Check prefixes (e.g., 'A_') or subdirs.")
#     return tif_path, json_path

# def draw_annotations_on_original(img, json_path):
#     """Draw JSON polygons/bboxes on original image."""
#     with open(json_path, 'r', encoding='utf-8') as f:
#         ann_data = json.load(f)
#     for shape in ann_data.get('shapes', []):
#         label = shape.get('label', '')
#         class_id = {  # Your class_map from preprocessing script
#             '\u6c14\u5b54': 0, '气孔': 0,
#             '\u5939\u6e23': 1, '夹渣': 1,
#             '\u88c2\u7eb9': 2, '裂纹': 2,
#             '\u54ac\u8fb9': 3, '咬边': 3,
#             '\u672a\u878d\u5408': 4, '未熔合': 4,
#             '\u672a\u710a\u900f': 5, '未焊透': 5,
#             '内凹': 3,
#             '夹钨': 1,
#         }.get(label, -1)
#         if class_id == -1:
#             continue
#         points = np.array(shape['points'], np.int32)
#         cv2.polylines(img, [points], True, COLORS.get(class_id, (0, 0, 255)), 2)
#         # Draw bbox approximation for comparison
#         x_min, y_min = np.min(points, axis=0)
#         x_max, y_max = np.max(points, axis=0)
#         cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), COLORS.get(class_id, (0, 0, 255)), 1)
#         cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def draw_yolo_bboxes_on_patch(img, label_path):
#     """Draw YOLO bboxes on patch image."""
#     h, w = img.shape[:2]
#     if not label_path.exists():
#         # Don't print for every background patch - too verbose
#         return img
#     with open(label_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue
#             class_id = int(parts[0])
#             cx, cy, bw, bh = map(float, parts[1:])
#             x_min = int((cx - bw / 2) * w)
#             y_min = int((cy - bh / 2) * h)
#             x_max = int((cx + bw / 2) * w)
#             y_max = int((cy + bh / 2) * h)
#             cv2.rectangle(img, (x_min, y_min), (x_max, y_max), COLORS.get(class_id, (0, 0, 255)), 2)
#             cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def create_overview_montage(original_annotated, patch_annotated_list, base_name, patch_files):
#     """Create a single overview image with original on top and patches in grid below.
#     NOTE: This is saved temporarily and will be deleted when script ends."""
#     # Resize original to match approx width (5 patches wide)
#     patch_size = patch_annotated_list[0].shape[0] if patch_annotated_list else 320
#     label_height = 30  # Space for labels below each image
#     original_resized = cv2.resize(original_annotated, (patch_size * 5, int(original_annotated.shape[0] * (patch_size * 5 / original_annotated.shape[1]))))

#     # Add label for original (below)
#     original_canvas_height = original_resized.shape[0] + label_height
#     original_canvas = np.zeros((original_canvas_height, original_resized.shape[1], 3), dtype=np.uint8)
#     original_canvas[:original_resized.shape[0], :] = original_resized
#     cv2.putText(original_canvas, f'Original Annotated: {base_name}', (10, original_resized.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     # Create grid for patches with label space
#     num_patches = len(patch_annotated_list)
#     cols = 5
#     rows = int(np.ceil(num_patches / cols))
#     montage_height = rows * (patch_size + label_height)
#     montage_width = cols * patch_size
#     montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)

#     for i, (patch, patch_file) in enumerate(zip(patch_annotated_list, patch_files)):
#         row = i // cols
#         col = i % cols
#         # Place patch
#         montage[row * (patch_size + label_height):row * (patch_size + label_height) + patch_size, col * patch_size:(col + 1) * patch_size] = patch
#         # Add label below
#         label_y = row * (patch_size + label_height) + patch_size + 20
#         cv2.putText(montage, f'Patch: {patch_file.stem}', (col * patch_size + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     # Stack original on top
#     full_montage = np.vstack((original_canvas, montage)) if num_patches > 0 else original_canvas

#     overview_path = VISUALIZATIONS_DIR / f'{base_name}_overview.jpg'  # Temporary save
#     cv2.imwrite(str(overview_path), full_montage)
#     print(f"Overview temporarily saved to {overview_path} (will be deleted on exit)")

#     # Display with loop for key and close detection
#     window_name = 'Overview Montage'
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow resizing
#     cv2.imshow(window_name, full_montage)
#     print("Montage window open. Focus the window and press any key to continue. Closing the window will also continue.")
#     while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
#         key = cv2.waitKey(1)
#         if key != -1:
#             break
#     cv2.destroyAllWindows()

# def interactive_flip_through_patches_with_original(patch_annotated_list, patch_files, original_annotated, base_name):
#     """Interactive flipping with TWO windows: patches and original for comparison."""
#     if not patch_annotated_list:
#         return
    
#     idx = 0
#     patch_window = 'Patch Viewer (Use Keys to Navigate)'
#     original_window = f'Original Annotated: {base_name}'
    
#     # Create both windows
#     cv2.namedWindow(patch_window, cv2.WINDOW_NORMAL)
#     cv2.namedWindow(original_window, cv2.WINDOW_NORMAL)
    
#     # Resize windows to reasonable sizes
#     # Get screen dimensions (approximate)
#     screen_width = 1920  # Adjust if needed
#     screen_height = 1080  # Adjust if needed
    
#     # Size windows to fit side by side
#     window_width = screen_width // 2 - 50
#     window_height = screen_height - 200
    
#     # Resize and position windows
#     cv2.resizeWindow(patch_window, window_width, window_height)
#     cv2.resizeWindow(original_window, window_width, window_height)
#     cv2.moveWindow(patch_window, 10, 50)
#     cv2.moveWindow(original_window, window_width + 30, 50)
    
#     # Display original image (stays constant) - MAKE SURE THIS SHOWS
#     print(f"\nShowing original annotated image for {base_name} in right window")
#     cv2.imshow(original_window, original_annotated)
#     cv2.waitKey(1)  # Force update
    
#     print("\n" + "="*60)
#     print("INTERACTIVE MODE WITH REFERENCE")
#     print("="*60)
#     print("\nTwo windows should be visible:")
#     print(f"  LEFT:  Patch Viewer - Shows current patch")
#     print(f"  RIGHT: Original Annotated {base_name} - Reference image")
#     print("\nControls (focus on left window):")
#     print("  'n' or 'd': Next patch")
#     print("  'p' or 'a': Previous patch")
#     print("  'q' or ESC: Quit")
#     print("  Close either window to exit")
#     print("="*60 + "\n")
    
#     # Display first patch
#     cv2.imshow(patch_window, patch_annotated_list[idx])
#     cv2.setWindowTitle(patch_window, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
#     cv2.waitKey(1)  # Force update
    
#     print(f"Starting with patch 1/{len(patch_annotated_list)}: {patch_files[idx].stem}")
#     print("Both windows should now be visible. If not, check your taskbar.\n")
    
#     while True:
#         # Check if either window is closed
#         try:
#             patch_visible = cv2.getWindowProperty(patch_window, cv2.WND_PROP_VISIBLE)
#             original_visible = cv2.getWindowProperty(original_window, cv2.WND_PROP_VISIBLE)
            
#             if patch_visible < 1:
#                 print("Patch window closed by user")
#                 break
#             if original_visible < 1:
#                 print("Original window closed by user")
#                 break
#         except:
#             print("Window no longer exists")
#             break
            
#         # Wait for key with short timeout
#         key = cv2.waitKey(30)  # 30ms timeout
        
#         # Only process if a key was actually pressed
#         if key != -1:
#             # Convert to ASCII if needed
#             key_char = key & 0xFF
            
#             # Handle key presses
#             if key_char == 27 or key_char == ord('q'):  # ESC or 'q'
#                 print("Quit command received")
#                 break
#             elif key_char == ord('n') or key_char == ord('d'):  # 'n' or 'd' for next
#                 idx = (idx + 1) % len(patch_annotated_list)
#                 print(f"Showing patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}")
#                 cv2.imshow(patch_window, patch_annotated_list[idx])
#                 cv2.setWindowTitle(patch_window, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
#             elif key_char == ord('p') or key_char == ord('a'):  # 'p' or 'a' for previous
#                 idx = (idx - 1) % len(patch_annotated_list)
#                 print(f"Showing patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}")
#                 cv2.imshow(patch_window, patch_annotated_list[idx])
#                 cv2.setWindowTitle(patch_window, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
            
#     cv2.destroyAllWindows()
#     print("Interactive viewer closed")

# def visualize_example(base_name, split='train'):
#     """Updated visualization: overview montage and interactive flipping with original reference."""
#     print(f"\n{'='*60}")
#     print(f"Processing: {base_name}")
#     print(f"{'='*60}")
    
#     tif_path, json_path = find_original_files(base_name)
#     if not tif_path or not json_path:
#         return

#     # Load and annotate original (IN MEMORY ONLY)
#     original = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
#     if original is None:
#         print(f"Error loading {tif_path}")
#         return
#     if len(original.shape) == 2:
#         original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
#     original_annotated = draw_annotations_on_original(original.copy(), json_path)
#     print("Original image annotated (in memory only, not saved)")

#     # Find all matching patches
#     images_dir = PROCESSED_DIR / split / 'images'
#     labels_dir = PROCESSED_DIR / split / 'labels'
#     patch_files = sorted([f for f in images_dir.glob('*.jpg') if f.stem.startswith(f'{base_name}_')])
#     print(f"Found {len(patch_files)} patches for {base_name} in {split}")

#     if not patch_files:
#         return

#     # Load and annotate all patches (IN MEMORY ONLY)
#     patch_annotated_list = []
#     for patch_path in patch_files:
#         label_path = labels_dir / f'{patch_path.stem}.txt'
#         patch_img = cv2.imread(str(patch_path))
#         if patch_img is None:
#             print(f"Error loading {patch_path}")
#             continue
#         patch_annotated = draw_yolo_bboxes_on_patch(patch_img.copy(), label_path)
#         patch_annotated_list.append(patch_annotated)
#     print(f"All {len(patch_annotated_list)} patches annotated (in memory only, not saved)")

#     # MODE 1: Create and show/save overview montage (TEMPORARY FILE)
#     print("\n--- MODE 1: Overview Montage ---")
#     create_overview_montage(original_annotated, patch_annotated_list, base_name, patch_files)

#     # MODE 2: Interactive high-res flipping WITH ORIGINAL REFERENCE WINDOW
#     print("\n--- MODE 2: Interactive Patch Viewer with Original Reference ---")
#     interactive_flip_through_patches_with_original(patch_annotated_list, patch_files, original_annotated, base_name)

# # Setup visualizations dir (will be cleaned up on exit)
# setup_visualizations_dir()

# try:
#     # Run for examples
#     for base_name in EXAMPLE_BASE_NAMES:
#         visualize_example(base_name, SPLIT)
    
#     print("\n" + "="*60)
#     print("All visualizations complete!")
#     print("Temporary files will now be cleaned up...")
#     print("="*60)
    
# except KeyboardInterrupt:
#     print("\nScript interrupted by user")
# except Exception as e:
#     print(f"\nError occurred: {e}")

# # Cleanup happens automatically via atexit

#Very close. Nav buttons work and original annoatted window is in dual mode:

# import cv2
# import json
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import shutil
# import atexit  # For cleanup on exit

# # Config (adapt to your paths)
# BASE_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data')
# PROCESSED_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\processed_balanced')
# VISUALIZATIONS_DIR = PROCESSED_DIR / 'visualizations'  # Temporary folder for visualizations
# EXAMPLE_BASE_NAMES = ['A_bam5']  # Add more if needed
# SPLIT = 'train'  # Check train/val/test as needed

# # Class names and colors (BGR for OpenCV)
# CLASS_NAMES = {0: 'porosity', 1: 'inclusion', 2: 'crack', 3: 'undercut', 4: 'lack_of_fusion', 5: 'lack_of_penetration'}
# COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255)}

# def cleanup_visualizations():
#     """Clean up the visualizations directory on exit."""
#     if VISUALIZATIONS_DIR.exists():
#         print(f"\nCleaning up temporary visualizations at {VISUALIZATIONS_DIR}")
#         shutil.rmtree(VISUALIZATIONS_DIR)
#         print("Cleanup complete - all temporary visualization files removed")

# # Register cleanup to run on script exit
# atexit.register(cleanup_visualizations)

# def setup_visualizations_dir():
#     """Create or empty the visualizations directory."""
#     if VISUALIZATIONS_DIR.exists():
#         shutil.rmtree(VISUALIZATIONS_DIR)  # Empty the folder
#     VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
#     print(f"Temporary visualizations directory created at {VISUALIZATIONS_DIR}")
#     print("NOTE: This folder will be automatically deleted when script ends\n")

# def find_original_files(base_name):
#     """Find TIF and JSON for original, with logging."""
#     tif_path = None
#     json_path = None
#     for subdir in ['crop_weld_images/L/1', 'crop_weld_images/L/2', 'crop_weld_images/T/1', 'crop_weld_images/T/2']:
#         candidate_tif = BASE_DIR / subdir / f'{base_name}.tif'
#         if candidate_tif.exists():
#             tif_path = candidate_tif
#             print(f"Found TIF: {tif_path}")
#             break
#     for subdir in ['crop_weld_jsons/L/1', 'crop_weld_jsons/L/2', 'crop_weld_jsons/T/1', 'crop_weld_jsons/T/2']:
#         candidate_json = BASE_DIR / subdir / f'{base_name}.json'
#         if candidate_json.exists():
#             json_path = candidate_json
#             print(f"Found JSON: {json_path}")
#             break
#     if not tif_path or not json_path:
#         print(f"Warning: Files not found for {base_name}. Check prefixes (e.g., 'A_') or subdirs.")
#     return tif_path, json_path

# def draw_annotations_on_original(img, json_path):
#     """Draw JSON polygons/bboxes on original image."""
#     with open(json_path, 'r', encoding='utf-8') as f:
#         ann_data = json.load(f)
#     for shape in ann_data.get('shapes', []):
#         label = shape.get('label', '')
#         class_id = {  # Your class_map from preprocessing script
#             '\u6c14\u5b54': 0, '气孔': 0,
#             '\u5939\u6e23': 1, '夹渣': 1,
#             '\u88c2\u7eb9': 2, '裂纹': 2,
#             '\u54ac\u8fb9': 3, '咬边': 3,
#             '\u672a\u878d\u5408': 4, '未熔合': 4,
#             '\u672a\u710a\u900f': 5, '未焊透': 5,
#             '内凹': 3,
#             '夹钨': 1,
#         }.get(label, -1)
#         if class_id == -1:
#             continue
#         points = np.array(shape['points'], np.int32)
#         cv2.polylines(img, [points], True, COLORS.get(class_id, (0, 0, 255)), 2)
#         # Draw bbox approximation for comparison
#         x_min, y_min = np.min(points, axis=0)
#         x_max, y_max = np.max(points, axis=0)
#         cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), COLORS.get(class_id, (0, 0, 255)), 1)
#         cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def draw_yolo_bboxes_on_patch(img, label_path):
#     """Draw YOLO bboxes on patch image."""
#     h, w = img.shape[:2]
#     if not label_path.exists():
#         # Don't print for every background patch - too verbose
#         return img
#     with open(label_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue
#             class_id = int(parts[0])
#             cx, cy, bw, bh = map(float, parts[1:])
#             x_min = int((cx - bw / 2) * w)
#             y_min = int((cy - bh / 2) * h)
#             x_max = int((cx + bw / 2) * w)
#             y_max = int((cy + bh / 2) * h)
#             cv2.rectangle(img, (x_min, y_min), (x_max, y_max), COLORS.get(class_id, (0, 0, 255)), 2)
#             cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
#     return img

# def create_overview_montage(original_annotated, patch_annotated_list, base_name, patch_files):
#     """Create a single overview image with original on top and patches in grid below.
#     NOTE: This is saved temporarily and will be deleted when script ends."""
#     # Resize original to match approx width (5 patches wide)
#     patch_size = patch_annotated_list[0].shape[0] if patch_annotated_list else 320
#     label_height = 30  # Space for labels below each image
#     original_resized = cv2.resize(original_annotated, (patch_size * 5, int(original_annotated.shape[0] * (patch_size * 5 / original_annotated.shape[1]))))

#     # Add label for original (below)
#     original_canvas_height = original_resized.shape[0] + label_height
#     original_canvas = np.zeros((original_canvas_height, original_resized.shape[1], 3), dtype=np.uint8)
#     original_canvas[:original_resized.shape[0], :] = original_resized
#     cv2.putText(original_canvas, f'Original Annotated: {base_name}', (10, original_resized.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     # Create grid for patches with label space
#     num_patches = len(patch_annotated_list)
#     cols = 5
#     rows = int(np.ceil(num_patches / cols))
#     montage_height = rows * (patch_size + label_height)
#     montage_width = cols * patch_size
#     montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)

#     for i, (patch, patch_file) in enumerate(zip(patch_annotated_list, patch_files)):
#         row = i // cols
#         col = i % cols
#         # Place patch
#         montage[row * (patch_size + label_height):row * (patch_size + label_height) + patch_size, col * patch_size:(col + 1) * patch_size] = patch
#         # Add label below
#         label_y = row * (patch_size + label_height) + patch_size + 20
#         cv2.putText(montage, f'Patch: {patch_file.stem}', (col * patch_size + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     # Stack original on top
#     full_montage = np.vstack((original_canvas, montage)) if num_patches > 0 else original_canvas

#     overview_path = VISUALIZATIONS_DIR / f'{base_name}_overview.jpg'  # Temporary save
#     cv2.imwrite(str(overview_path), full_montage)
#     print(f"Overview temporarily saved to {overview_path} (will be deleted on exit)")

#     # Display with loop for key and close detection
#     window_name = 'Overview Montage'
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow resizing
#     cv2.imshow(window_name, full_montage)
#     print("Montage window open. Focus the window and press any key to continue. Closing the window will also continue.")
#     while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
#         key = cv2.waitKey(1)
#         if key != -1:
#             break
#     cv2.destroyAllWindows()

# def interactive_flip_through_patches_with_original(patch_annotated_list, patch_files, original_annotated, base_name):
#     """Interactive flipping with TWO windows: patches and original for comparison."""
#     if not patch_annotated_list:
#         return
    
#     idx = 0
#     patch_window = 'Patch Viewer (Click here and use keys)'
#     original_window = f'Original Annotated: {base_name}'
    
#     # Create both windows
#     cv2.namedWindow(patch_window, cv2.WINDOW_NORMAL)
#     cv2.namedWindow(original_window, cv2.WINDOW_NORMAL)
    
#     # Size windows to fit side by side
#     window_width = 800
#     window_height = 600
    
#     # Resize and position windows
#     cv2.resizeWindow(patch_window, window_width, window_height)
#     cv2.resizeWindow(original_window, window_width, window_height)
#     cv2.moveWindow(patch_window, 50, 100)
#     cv2.moveWindow(original_window, window_width + 100, 100)
    
#     # Display original image (stays constant)
#     cv2.imshow(original_window, original_annotated)
    
#     # Display first patch
#     cv2.imshow(patch_window, patch_annotated_list[idx])
#     cv2.setWindowTitle(patch_window, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
    
#     print("\n" + "="*60)
#     print("INTERACTIVE MODE - TWO WINDOWS")
#     print("="*60)
#     print(f"LEFT:  Patch Viewer - Current patch")
#     print(f"RIGHT: Original Annotated {base_name} - Reference")
#     print("\n*** IMPORTANT: CLICK ON THE LEFT PATCH WINDOW TO GIVE IT FOCUS ***")
#     print("\nControls (after clicking left window):")
#     print("  'n' or 'd': Next patch")
#     print("  'p' or 'a': Previous patch") 
#     print("  SPACE: Also goes to next patch")
#     print("  'q' or ESC: Quit")
#     print("\nWaiting for keyboard input...")
#     print("If keys don't work, click on the Patch Viewer window first!")
#     print("="*60 + "\n")
    
#     # Make sure windows are updated
#     cv2.waitKey(1)
    
#     # Focus on patch window
#     cv2.setWindowProperty(patch_window, cv2.WND_PROP_TOPMOST, 1)
#     cv2.setWindowProperty(patch_window, cv2.WND_PROP_TOPMOST, 0)
    
#     print(f"Starting with patch 1/{len(patch_annotated_list)}: {patch_files[idx].stem}")
    
#     # Main loop with blocking wait for keys
#     while True:
#         # Check if windows are still open
#         try:
#             if cv2.getWindowProperty(patch_window, cv2.WND_PROP_VISIBLE) < 1:
#                 print("Patch window closed")
#                 break
#             if cv2.getWindowProperty(original_window, cv2.WND_PROP_VISIBLE) < 1:
#                 print("Original window closed")
#                 break
#         except:
#             break
        
#         # Wait for key press (blocking wait)
#         key = cv2.waitKeyEx(0)  # Use waitKeyEx for extended key codes, 0 for blocking
        
#         # Check for window close during wait
#         if key == -1:
#             continue
            
#         # Debug output
#         print(f"Key pressed: {key} (char: {chr(key & 0xFF) if (key & 0xFF) < 127 and (key & 0xFF) > 31 else 'special'})")
        
#         # Handle different key codes
#         if key == 27:  # ESC
#             print("ESC pressed - exiting")
#             break
#         elif key == ord('q') or key == ord('Q'):
#             print("Q pressed - exiting")
#             break
#         elif key == ord('n') or key == ord('N') or key == ord('d') or key == ord('D') or key == 32:  # 32 is SPACE
#             idx = (idx + 1) % len(patch_annotated_list)
#             print(f"Next -> Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}")
#             cv2.imshow(patch_window, patch_annotated_list[idx])
#             cv2.setWindowTitle(patch_window, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
#         elif key == ord('p') or key == ord('P') or key == ord('a') or key == ord('A'):
#             idx = (idx - 1) % len(patch_annotated_list)
#             print(f"Previous -> Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}")
#             cv2.imshow(patch_window, patch_annotated_list[idx])
#             cv2.setWindowTitle(patch_window, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
#         # Arrow keys (these codes might vary by system)
#         elif key == 2555904 or key == 2621440 or key == 0x270000:  # Right arrow variants
#             idx = (idx + 1) % len(patch_annotated_list)
#             print(f"Right Arrow -> Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}")
#             cv2.imshow(patch_window, patch_annotated_list[idx])
#             cv2.setWindowTitle(patch_window, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
#         elif key == 2424832 or key == 2490368 or key == 0x250000:  # Left arrow variants
#             idx = (idx - 1) % len(patch_annotated_list)
#             print(f"Left Arrow -> Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}")
#             cv2.imshow(patch_window, patch_annotated_list[idx])
#             cv2.setWindowTitle(patch_window, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
            
#     cv2.destroyAllWindows()
#     print("Interactive viewer closed")

# def visualize_example(base_name, split='train'):
#     """Updated visualization: overview montage and interactive flipping with original reference."""
#     print(f"\n{'='*60}")
#     print(f"Processing: {base_name}")
#     print(f"{'='*60}")
    
#     tif_path, json_path = find_original_files(base_name)
#     if not tif_path or not json_path:
#         return

#     # Load and annotate original (IN MEMORY ONLY)
#     original = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
#     if original is None:
#         print(f"Error loading {tif_path}")
#         return
#     if len(original.shape) == 2:
#         original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
#     original_annotated = draw_annotations_on_original(original.copy(), json_path)
#     print("Original image annotated (in memory only, not saved)")

#     # Find all matching patches
#     images_dir = PROCESSED_DIR / split / 'images'
#     labels_dir = PROCESSED_DIR / split / 'labels'
#     patch_files = sorted([f for f in images_dir.glob('*.jpg') if f.stem.startswith(f'{base_name}_')])
#     print(f"Found {len(patch_files)} patches for {base_name} in {split}")

#     if not patch_files:
#         return

#     # Load and annotate all patches (IN MEMORY ONLY)
#     patch_annotated_list = []
#     for patch_path in patch_files:
#         label_path = labels_dir / f'{patch_path.stem}.txt'
#         patch_img = cv2.imread(str(patch_path))
#         if patch_img is None:
#             print(f"Error loading {patch_path}")
#             continue
#         patch_annotated = draw_yolo_bboxes_on_patch(patch_img.copy(), label_path)
#         patch_annotated_list.append(patch_annotated)
#     print(f"All {len(patch_annotated_list)} patches annotated (in memory only, not saved)")

#     # MODE 1: Create and show/save overview montage (TEMPORARY FILE)
#     print("\n--- MODE 1: Overview Montage ---")
#     create_overview_montage(original_annotated, patch_annotated_list, base_name, patch_files)

#     # MODE 2: Interactive high-res flipping WITH ORIGINAL REFERENCE WINDOW
#     print("\n--- MODE 2: Interactive Patch Viewer with Original Reference ---")
#     interactive_flip_through_patches_with_original(patch_annotated_list, patch_files, original_annotated, base_name)

# # Setup visualizations dir (will be cleaned up on exit)
# setup_visualizations_dir()

# try:
#     # Run for examples
#     for base_name in EXAMPLE_BASE_NAMES:
#         visualize_example(base_name, SPLIT)
    
#     print("\n" + "="*60)
#     print("All visualizations complete!")
#     print("Temporary files will now be cleaned up...")
#     print("="*60)
    
# except KeyboardInterrupt:
#     print("\nScript interrupted by user")
# except Exception as e:
#     print(f"\nError occurred: {e}")

# # Cleanup happens automatically via atexit


import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import atexit  # For cleanup on exit

# Config (adapt to your paths)
BASE_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data')
PROCESSED_DIR = Path(r'C:\AWrk\SWRD_YOLO_Project\processed_balanced')
VISUALIZATIONS_DIR = PROCESSED_DIR / 'visualizations'  # Temporary folder for visualizations
EXAMPLE_BASE_NAMES = ['A_bam5']  # Add more if needed A_DJ-RT-20220621-1, A_bam5
SPLIT = 'train'  # Check train/val/test as needed

# Class names and colors (BGR for OpenCV)
CLASS_NAMES = {0: 'porosity', 1: 'inclusion', 2: 'crack', 3: 'undercut', 4: 'lack_of_fusion', 5: 'lack_of_penetration'}
COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255)}

def cleanup_visualizations():
    """Clean up the visualizations directory on exit."""
    if VISUALIZATIONS_DIR.exists():
        print(f"\nCleaning up temporary visualizations at {VISUALIZATIONS_DIR}")
        shutil.rmtree(VISUALIZATIONS_DIR)
        print("Cleanup complete - all temporary visualization files removed")

# Register cleanup to run on script exit
atexit.register(cleanup_visualizations)

def setup_visualizations_dir():
    """Create or empty the visualizations directory."""
    if VISUALIZATIONS_DIR.exists():
        shutil.rmtree(VISUALIZATIONS_DIR)  # Empty the folder
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Temporary visualizations directory created at {VISUALIZATIONS_DIR}")
    print("NOTE: This folder will be automatically deleted when script ends\n")

def find_original_files(base_name):
    """Find TIF and JSON for original, with logging."""
    tif_path = None
    json_path = None
    for subdir in ['crop_weld_images/L/1', 'crop_weld_images/L/2', 'crop_weld_images/T/1', 'crop_weld_images/T/2']:
        candidate_tif = BASE_DIR / subdir / f'{base_name}.tif'
        if candidate_tif.exists():
            tif_path = candidate_tif
            print(f"Found TIF: {tif_path}")
            break
    for subdir in ['crop_weld_jsons/L/1', 'crop_weld_jsons/L/2', 'crop_weld_jsons/T/1', 'crop_weld_jsons/T/2']:
        candidate_json = BASE_DIR / subdir / f'{base_name}.json'
        if candidate_json.exists():
            json_path = candidate_json
            print(f"Found JSON: {json_path}")
            break
    if not tif_path or not json_path:
        print(f"Warning: Files not found for {base_name}. Check prefixes (e.g., 'A_') or subdirs.")
    return tif_path, json_path

def draw_annotations_on_original(img, json_path):
    """Draw JSON polygons/bboxes on original image."""
    with open(json_path, 'r', encoding='utf-8') as f:
        ann_data = json.load(f)
    for shape in ann_data.get('shapes', []):
        label = shape.get('label', '')
        class_id = {  # Your class_map from preprocessing script
            '\u6c14\u5b54': 0, '气孔': 0,
            '\u5939\u6e23': 1, '夹渣': 1,
            '\u88c2\u7eb9': 2, '裂纹': 2,
            '\u54ac\u8fb9': 3, '咬边': 3,
            '\u672a\u878d\u5408': 4, '未熔合': 4,
            '\u672a\u710a\u900f': 5, '未焊透': 5,
            '内凹': 3,
            '夹钨': 1,
        }.get(label, -1)
        if class_id == -1:
            continue
        points = np.array(shape['points'], np.int32)
        cv2.polylines(img, [points], True, COLORS.get(class_id, (0, 0, 255)), 2)
        # Draw bbox approximation for comparison
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), COLORS.get(class_id, (0, 0, 255)), 1)
        cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
    return img

def draw_yolo_bboxes_on_patch(img, label_path):
    """Draw YOLO bboxes on patch image."""
    h, w = img.shape[:2]
    if not label_path.exists():
        # Don't print for every background patch - too verbose
        return img
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x_min = int((cx - bw / 2) * w)
            y_min = int((cy - bh / 2) * h)
            x_max = int((cx + bw / 2) * w)
            y_max = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), COLORS.get(class_id, (0, 0, 255)), 2)
            cv2.putText(img, CLASS_NAMES.get(class_id, 'unknown'), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(class_id, (0, 0, 255)), 1)
    return img

def create_overview_montage(original_annotated, patch_annotated_list, base_name, patch_files):
    """Create a single overview image with original on top and patches in grid below.
    NOTE: This is saved temporarily and will be deleted when script ends."""
    # Resize original to match approx width (5 patches wide)
    patch_size = patch_annotated_list[0].shape[0] if patch_annotated_list else 320
    label_height = 30  # Space for labels below each image
    original_resized = cv2.resize(original_annotated, (patch_size * 5, int(original_annotated.shape[0] * (patch_size * 5 / original_annotated.shape[1]))))

    # Add label for original (below)
    original_canvas_height = original_resized.shape[0] + label_height
    original_canvas = np.zeros((original_canvas_height, original_resized.shape[1], 3), dtype=np.uint8)
    original_canvas[:original_resized.shape[0], :] = original_resized
    cv2.putText(original_canvas, f'Original Annotated: {base_name}', (10, original_resized.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Create grid for patches with label space
    num_patches = len(patch_annotated_list)
    cols = 5
    rows = int(np.ceil(num_patches / cols))
    montage_height = rows * (patch_size + label_height)
    montage_width = cols * patch_size
    montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)

    for i, (patch, patch_file) in enumerate(zip(patch_annotated_list, patch_files)):
        row = i // cols
        col = i % cols
        # Place patch
        montage[row * (patch_size + label_height):row * (patch_size + label_height) + patch_size, col * patch_size:(col + 1) * patch_size] = patch
        # Add label below
        label_y = row * (patch_size + label_height) + patch_size + 20
        cv2.putText(montage, f'Patch: {patch_file.stem}', (col * patch_size + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Stack original on top
    full_montage = np.vstack((original_canvas, montage)) if num_patches > 0 else original_canvas

    overview_path = VISUALIZATIONS_DIR / f'{base_name}_overview.jpg'  # Temporary save
    cv2.imwrite(str(overview_path), full_montage)
    print(f"Overview temporarily saved to {overview_path} (will be deleted on exit)")

    # Display with loop for key and close detection
    window_name = 'Overview Montage'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow resizing
    cv2.imshow(window_name, full_montage)
    print("Montage window open. Focus the window and press any key to continue. Closing the window will also continue.")
    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        key = cv2.waitKey(1)
        if key != -1:
            break
    cv2.destroyAllWindows()

class ZoomableImageViewer:
    """Class to handle zooming and panning for the original image"""
    def __init__(self, image, window_name):
        self.original_image = image
        self.window_name = window_name
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.mouse_down = False
        self.mouse_x = 0
        self.mouse_y = 0
        self.window_width = 1000  # Larger window for better resolution
        self.window_height = 800
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for panning and zooming"""
        if event == cv2.EVENT_MOUSEWHEEL:
            # Zoom with mouse wheel
            if flags > 0:  # Scroll up - zoom in
                self.zoom_level = min(5.0, self.zoom_level * 1.2)
            else:  # Scroll down - zoom out
                self.zoom_level = max(0.2, self.zoom_level / 1.2)
            self.update_display()
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = True
            self.mouse_x = x
            self.mouse_y = y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_down:
                # Pan the image
                dx = x - self.mouse_x
                dy = y - self.mouse_y
                self.pan_x += dx
                self.pan_y += dy
                self.mouse_x = x
                self.mouse_y = y
                self.update_display()
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_down = False
            
        elif event == cv2.EVENT_RBUTTONDBLCLK:
            # Reset view on right double-click
            self.zoom_level = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.update_display()
    
    def update_display(self):
        """Update the displayed image with current zoom and pan"""
        h, w = self.original_image.shape[:2]
        
        # Calculate the size of the zoomed image
        new_w = int(w * self.zoom_level)
        new_h = int(h * self.zoom_level)
        
        # Resize the image
        if self.zoom_level != 1.0:
            zoomed = cv2.resize(self.original_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            zoomed = self.original_image.copy()
        
        # Calculate the region to display based on pan
        display_w = min(self.window_width, new_w)
        display_h = min(self.window_height, new_h)
        
        # Calculate crop coordinates with pan
        start_x = max(0, min(new_w - display_w, (new_w - display_w) // 2 - self.pan_x))
        start_y = max(0, min(new_h - display_h, (new_h - display_h) // 2 - self.pan_y))
        end_x = start_x + display_w
        end_y = start_y + display_h
        
        # Crop the zoomed image
        display_img = zoomed[start_y:end_y, start_x:end_x]
        
        # Add zoom level indicator
        cv2.putText(display_img, f'Zoom: {self.zoom_level:.1f}x (Scroll to zoom, drag to pan, double right-click to reset)', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, display_img)
    
    def show(self):
        """Initialize and show the window"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.update_display()

def interactive_flip_through_patches_with_original(patch_annotated_list, patch_files, original_annotated, base_name):
    """Interactive flipping with TWO windows: patches and zoomable original for comparison."""
    if not patch_annotated_list:
        return
    
    idx = 0
    patch_window = 'Patch Viewer (Click here and use keys)'
    original_window = f'Original Annotated: {base_name} (ZOOMABLE)'
    
    # Create patch window
    cv2.namedWindow(patch_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(patch_window, 700, 700)
    cv2.moveWindow(patch_window, 50, 100)
    
    # Create zoomable viewer for original
    original_viewer = ZoomableImageViewer(original_annotated, original_window)
    original_viewer.show()
    cv2.moveWindow(original_window, 800, 100)
    
    # Display first patch
    cv2.imshow(patch_window, patch_annotated_list[idx])
    cv2.setWindowTitle(patch_window, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
    
    print("\n" + "="*70)
    print("INTERACTIVE MODE - DUAL WINDOWS WITH ZOOMABLE ORIGINAL")
    print("="*70)
    print(f"LEFT:  Patch Viewer - Current patch (320x320)")
    print(f"RIGHT: Original Annotated {base_name} - FULL RESOLUTION with zoom/pan")
    print("\n*** ORIGINAL IMAGE CONTROLS (Right Window) ***")
    print("  - MOUSE WHEEL: Zoom in/out")
    print("  - CLICK & DRAG: Pan around the image")
    print("  - DOUBLE RIGHT-CLICK: Reset zoom and position")
    print("\n*** PATCH NAVIGATION (Left Window - click it first) ***")
    print("  'n', 'd', or SPACE: Next patch")
    print("  'p' or 'a': Previous patch") 
    print("  'q' or ESC: Quit")
    print("\nOriginal image dimensions:", original_annotated.shape[:2])
    print("="*70 + "\n")
    
    # Make sure windows are updated
    cv2.waitKey(1)
    
    print(f"Starting with patch 1/{len(patch_annotated_list)}: {patch_files[idx].stem}")
    
    # Main loop with blocking wait for keys
    while True:
        # Check if windows are still open
        try:
            if cv2.getWindowProperty(patch_window, cv2.WND_PROP_VISIBLE) < 1:
                print("Patch window closed")
                break
            if cv2.getWindowProperty(original_window, cv2.WND_PROP_VISIBLE) < 1:
                print("Original window closed")
                break
        except:
            break
        
        # Wait for key press (short timeout to allow mouse events)
        key = cv2.waitKey(10)
        
        if key == -1:
            continue
            
        # Handle different key codes
        if key == 27:  # ESC
            print("ESC pressed - exiting")
            break
        elif key == ord('q') or key == ord('Q'):
            print("Q pressed - exiting")
            break
        elif key == ord('n') or key == ord('N') or key == ord('d') or key == ord('D') or key == 32:  # 32 is SPACE
            idx = (idx + 1) % len(patch_annotated_list)
            print(f"Next -> Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}")
            cv2.imshow(patch_window, patch_annotated_list[idx])
            cv2.setWindowTitle(patch_window, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
        elif key == ord('p') or key == ord('P') or key == ord('a') or key == ord('A'):
            idx = (idx - 1) % len(patch_annotated_list)
            print(f"Previous -> Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}")
            cv2.imshow(patch_window, patch_annotated_list[idx])
            cv2.setWindowTitle(patch_window, f'Patch {idx+1}/{len(patch_annotated_list)}: {patch_files[idx].stem}')
            
    cv2.destroyAllWindows()
    print("Interactive viewer closed")

def visualize_example(base_name, split='train'):
    """Updated visualization: overview montage and interactive flipping with zoomable original reference."""
    print(f"\n{'='*60}")
    print(f"Processing: {base_name}")
    print(f"{'='*60}")
    
    tif_path, json_path = find_original_files(base_name)
    if not tif_path or not json_path:
        return

    # Load and annotate original (IN MEMORY ONLY) - Keep FULL RESOLUTION
    original = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
    if original is None:
        print(f"Error loading {tif_path}")
        return
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    # Draw annotations on full resolution image
    original_annotated = draw_annotations_on_original(original.copy(), json_path)
    print(f"Original image annotated at full resolution: {original_annotated.shape[:2]} (in memory only)")

    # Find all matching patches
    images_dir = PROCESSED_DIR / split / 'images'
    labels_dir = PROCESSED_DIR / split / 'labels'
    patch_files = sorted([f for f in images_dir.glob('*.jpg') if f.stem.startswith(f'{base_name}_')])
    print(f"Found {len(patch_files)} patches for {base_name} in {split}")

    if not patch_files:
        return

    # Load and annotate all patches (IN MEMORY ONLY)
    patch_annotated_list = []
    for patch_path in patch_files:
        label_path = labels_dir / f'{patch_path.stem}.txt'
        patch_img = cv2.imread(str(patch_path))
        if patch_img is None:
            print(f"Error loading {patch_path}")
            continue
        patch_annotated = draw_yolo_bboxes_on_patch(patch_img.copy(), label_path)
        patch_annotated_list.append(patch_annotated)
    print(f"All {len(patch_annotated_list)} patches annotated (in memory only, not saved)")

    # MODE 1: Create and show/save overview montage (TEMPORARY FILE)
    print("\n--- MODE 1: Overview Montage ---")
    create_overview_montage(original_annotated, patch_annotated_list, base_name, patch_files)

    # MODE 2: Interactive high-res flipping WITH ZOOMABLE ORIGINAL REFERENCE
    print("\n--- MODE 2: Interactive Patch Viewer with Zoomable Original Reference ---")
    interactive_flip_through_patches_with_original(patch_annotated_list, patch_files, original_annotated, base_name)

# Setup visualizations dir (will be cleaned up on exit)
setup_visualizations_dir()

try:
    # Run for examples
    for base_name in EXAMPLE_BASE_NAMES:
        visualize_example(base_name, SPLIT)
    
    print("\n" + "="*60)
    print("All visualizations complete!")
    print("Temporary files will now be cleaned up...")
    print("="*60)
    
except KeyboardInterrupt:
    print("\nScript interrupted by user")
except Exception as e:
    print(f"\nError occurred: {e}")

# Cleanup happens automatically via atexit
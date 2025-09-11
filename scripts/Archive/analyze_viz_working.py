# scripts/analyze.py:

import os
import json
from collections import defaultdict
import cv2
import numpy as np

base_dir = r'C:\AWrk\SWRD_YOLO_Project\data'

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

# -------------------------
# Class definitions
# -------------------------
class_map = {
    '\u6c14\u5b54': 0, '气孔': 0,             # Porosity
    '\u5939\u6e23': 1, '夹渣': 1,             # Inclusion
    '\u88c2\u7eb9': 2, '裂纹': 2,             # Crack
    '\u54ac\u8fb9': 3, '咬边': 3,             # Undercut
    '\u672a\u878d\u5408': 4, '未熔合': 4,     # Lack of fusion
    '\u672a\u710a\u900f': 5, '未焊透': 5,     # Lack of penetration
    '内凹': 3,                                # Variant: concave undercut
    '夹钨': 1,                                # Variant: tungsten inclusion
    '焊瘤': 6,                                # Weld tumor
    'None': -1,                               # Invalid
    '\u4f2a\u7f3a\u9677': 7, '伪缺陷': 7,     # Pseudo defect
    '\u710a\u7f1d': 8, '焊缝': 8              # Weld seam
}

# English-only labels
label_map = {
    0: 'porosity',
    1: 'inclusion',
    2: 'crack',
    3: 'undercut',
    4: 'lack_of_fusion',
    5: 'lack_of_penetration',
    6: 'weld_tumor',
    -1: 'invalid',
    7: 'pseudo_defect',
    8: 'weld_seam'
}

# Colors for visualization
color_map = {
    0: (0, 255, 0),   # Green defects
    1: (0, 255, 0),
    2: (0, 255, 0),
    3: (0, 255, 0),
    4: (0, 255, 0),
    5: (0, 255, 0),
    6: (0, 255, 0),
    7: (0, 255, 255), # Yellow pseudo defects
    8: (255, 0, 0),   # Blue weld seams
    -1: (0, 0, 255)   # Red invalids
}

# -------------------------
# File collection
# -------------------------
def get_files(subdirs, ext):
    files = []
    for sub in subdirs:
        d = os.path.join(base_dir, sub)
        if os.path.exists(d):
            files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(ext)]
    return files

tifs = get_files(img_subdirs, '.tif')
jsons = get_files(json_subdirs, '.json')

pairs = {}
unknown_labels = defaultdict(list)

for tif in tifs:
    base = os.path.splitext(os.path.basename(tif))[0]
    pairs[base] = {'tif': tif}

for js in jsons:
    base = os.path.splitext(os.path.basename(js))[0]
    if base in pairs:
        pairs[base]['json'] = js

# -------------------------
# Processing loop
# -------------------------
for base, files in pairs.items():
    if 'json' not in files:
        continue

    with open(files['json'], 'r', encoding='utf-8') as f:
        ann = json.load(f)

    shapes = ann.get('shapes', [])
    img = cv2.imread(files['tif'], cv2.IMREAD_UNCHANGED)
    if img is None:
        continue

    img_viz = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_viz = cv2.cvtColor(img_viz, cv2.COLOR_GRAY2BGR)

    for shape in shapes:
        label = shape.get('label', 'unknown')
        cls_id = class_map.get(label, None)

        if cls_id is None:
            cls_id = -99  # unknown numeric id
            unknown_labels[label].append(base + '.json')

        pts = np.array(shape['points'], np.int32)
        color = color_map.get(cls_id, (0, 0, 255))  # red fallback
        cv2.polylines(img_viz, [pts], True, color, 2)

        # Only English label
        text = label_map.get(cls_id, f'unknown({cls_id})')
        cv2.putText(img_viz, text, tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    out_path = os.path.join("viz", base + "_viz.png")
    cv2.imwrite(out_path, img_viz)

# -------------------------
# Log unknown labels
# -------------------------
with open("log_unknowns.txt", "w", encoding="utf-8") as f:
    for lbl, files in unknown_labels.items():
        f.write(f"{lbl} -> {files}\n")

print("✅ Done. Visuals saved in viz/. Unknown labels logged in log_unknowns.txt")

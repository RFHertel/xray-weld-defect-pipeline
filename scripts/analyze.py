# #scripts/analyze.py
import os
import json
from collections import defaultdict
import cv2
import numpy as np

# -------------------------
# CONFIG
# -------------------------
base_dir = r'C:\AWrk\SWRD_YOLO_Project\data'
viz_dir = "viz"
regen_viz = False   # <<== SET TO False to skip re-generating images

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
# CLASS MAPS
# -------------------------
class_map = {
    '\u6c14\u5b54': 0, '气孔': 0,             # Porosity
    '\u5939\u6e23': 1, '夹渣': 1,             # Inclusion
    '\u88c2\u7eb9': 2, '裂纹': 2,             # Crack
    '\u54ac\u8fb9': 3, '咬边': 3,             # Undercut
    '\u672a\u878d\u5408': 4, '未熔合': 4,     # Lack of fusion
    '\u672a\u710a\u900f': 5, '未焊透': 5,     # Lack of penetration
    '内凹': 3,                                # Variant
    '夹钨': 1,                                # Variant
    '焊瘤': 6,                                # Weld tumor
    'None': -1,                               # Invalid
    '\u4f2a\u7f3a\u9677': 7, '伪缺陷': 7,     # Pseudo defect
    '\u710a\u7f1d': 8, '焊缝': 8              # Weld seam
}

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
# HELPERS
# -------------------------
def get_files(subdirs, ext):
    files = []
    for sub in subdirs:
        d = os.path.join(base_dir, sub)
        if os.path.exists(d):
            files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(ext)]
    return files

def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]

# -------------------------
# COLLECT FILES
# -------------------------
tifs = get_files(img_subdirs, '.tif')
jsons = get_files(json_subdirs, '.json')

pairs = {}
for tif in tifs:
    base = os.path.splitext(os.path.basename(tif))[0]
    pairs[base] = {'tif': tif}
for js in jsons:
    base = os.path.splitext(os.path.basename(js))[0]
    if base in pairs:
        pairs[base]['json'] = js

# -------------------------
# DATA STRUCTURES
# -------------------------
image_stats = {}
dataset_totals = defaultdict(int)
master_index = []
unknown_labels = defaultdict(list)

# -------------------------
# PROCESS
# -------------------------
os.makedirs(viz_dir, exist_ok=True)

for base, files in pairs.items():
    if 'json' not in files:
        continue

    with open(files['json'], 'r', encoding='utf-8') as f:
        ann = json.load(f)

    shapes = ann.get('shapes', [])
    w, h = ann.get('imageWidth', 0), ann.get('imageHeight', 0)

    img = cv2.imread(files['tif'], cv2.IMREAD_UNCHANGED)
    if img is None:
        continue

    if regen_viz:
        img_viz = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_viz = cv2.cvtColor(img_viz, cv2.COLOR_GRAY2BGR)

    per_image_counts = defaultdict(int)

    for shape in shapes:
        label = shape.get('label', 'unknown')
        cls_id = class_map.get(label, None)

        if cls_id is None:
            cls_id = -99
            unknown_labels[label].append(base + '.json')

        per_image_counts[label_map.get(cls_id, f"unknown({cls_id})")] += 1
        dataset_totals[label_map.get(cls_id, f"unknown({cls_id})")] += 1

        bbox = polygon_to_bbox(shape['points'])

        master_index.append({
            "image_id": base,
            "class": label_map.get(cls_id, f"unknown({cls_id})"),
            "bbox": bbox,
            "polygon": shape['points']
        })

        if regen_viz:
            pts = np.array(shape['points'], np.int32)
            color = color_map.get(cls_id, (0, 0, 255))
            cv2.polylines(img_viz, [pts], True, color, 2)
            cv2.putText(img_viz, label_map.get(cls_id, f"unknown({cls_id})"),
                        tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if regen_viz:
        out_path = os.path.join(viz_dir, base + "_viz.png")
        cv2.imwrite(out_path, img_viz)

    image_stats[base] = {
        "width": w,
        "height": h,
        "defects": dict(per_image_counts)
    }

# -------------------------
# SAVE OUTPUTS
# -------------------------
def save_json_ascii(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)  # force ASCII

save_json_ascii("image_stats.json", image_stats)
save_json_ascii("dataset_totals.json", dataset_totals)
save_json_ascii("master_index.json", master_index)
save_json_ascii("log_unknowns.json", unknown_labels)

print("[OK] Done. Outputs written:")
print(" - image_stats.json (per image counts)")
print(" - dataset_totals.json (overall counts)")
print(" - master_index.json (detailed defect index)")
print(" - log_unknowns.json (unmapped labels)")
if regen_viz:
    print(" - viz/ folder updated with overlays")
else:
    print(" - viz/ folder left untouched")
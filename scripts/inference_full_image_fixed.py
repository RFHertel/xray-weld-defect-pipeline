# scripts/inference_full_image_fixed.py - ACTUALLY FIXED with Coloured Bounding Boxes for inference

# # Basic usage with image
# python scripts/inference_full_image_fixed.py --image "C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif"

# # Specify output directory
# python scripts/inference_full_image_fixed.py --image "C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif" --output "C:\AWrk\SWRD_YOLO_Project\inference_results"

# # Adjust confidence threshold
# python scripts/inference_full_image_fixed.py --image "C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif" --conf 0.3 --nms 0.5

# # Skip visualization (just save JSON results)
# python scripts/inference_full_image_fixed.py --image "C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif" --no-viz

# How the Inference Pipeline Works:
# 1. Sliding Window Approach:

# Takes the full weld image and slides a 320x320 window (or adaptive size) across it
# Uses 50% overlap between windows to ensure defects at boundaries aren't missed
# Each window becomes a patch for processing

# 2. Preprocessing Steps (matching training exactly):

# Grayscale conversion: Ensures consistent single-channel input
# Normalization: Stretches contrast to 0-255 range
# CLAHE enhancement: Applies Contrast Limited Adaptive Histogram Equalization with clipLimit=2.0 and 8x8 tiles
# BGR conversion: Converts to 3-channel for model (even though grayscale, model expects 3 channels)
# Resize to 640x640: Model was trained on this size

# 3. Detection & Post-processing:

# Each patch runs through the model independently
# Coordinates are scaled back from 640x640 to original patch size
# Then translated to full image coordinates
# NMS removes duplicate detections from overlapping windows

# 4. Visualization:

# Original image is enhanced with CLAHE for visibility
# Converted to BGR to allow colored annotations
# Each class gets a distinct color for its bounding boxes

# This ensures the model sees exactly the same preprocessing it was trained with, while the visualization remains in color.

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
from tqdm import tqdm
import argparse

class SlidingWindowInference:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.window_size = 320
        self.overlap = 0.5
        self.conf_threshold = 0.25
        self.nms_threshold = 0.45
        self.class_names = ['porosity', 'inclusion', 'crack', 'undercut', 
                           'lack_of_fusion', 'lack_of_penetration']
    
    def preprocess_patch(self, patch):
        """Apply EXACT same preprocessing as training"""
        # Convert to grayscale if needed
        if len(patch.shape) == 3:
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.shape[2] == 3 else patch[:,:,0]
        else:
            patch_gray = patch
        
        # Normalize and apply CLAHE
        stretched = cv2.normalize(patch_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(stretched)
        
        # Convert to 3-channel BGR
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def process_full_image(self, image_path, visualize=False):
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, None
        
        h, w = img.shape[:2] if len(img.shape) >= 2 else (img.shape[0], img.shape[1])
        
        # Window sizing
        window_size = max(min(h, w) // 2, 320)
        if h < window_size or w < window_size:
            window_size = min(h, w)
        stride = int(window_size * (1 - self.overlap))
        
        all_detections = []
        print(f"Processing image {w}x{h} with window {window_size}, stride {stride}")
        
        n_windows_x = max(1, (w - window_size + stride - 1) // stride)
        n_windows_y = max(1, (h - window_size + stride - 1) // stride)
        total_windows = n_windows_x * n_windows_y
        print(f"Total windows to process: {total_windows}")
        
        with tqdm(total=total_windows, desc="Processing windows") as pbar:
            for y in range(0, max(1, h - window_size + 1), stride):
                for x in range(0, max(1, w - window_size + 1), stride):
                    y_end = min(y + window_size, h)
                    x_end = min(x + window_size, w)
                    
                    patch = img[y:y_end, x:x_end] if len(img.shape) == 2 else img[y:y_end, x:x_end, :]
                    
                    if patch.shape[0] < 100 or patch.shape[1] < 100:
                        pbar.update(1)
                        continue
                    
                    processed_patch = self.preprocess_patch(patch)
                    processed_patch = cv2.resize(processed_patch, (640, 640))
                    
                    results = self.model(processed_patch, conf=self.conf_threshold, verbose=False)
                    
                    for r in results:
                        if r.boxes is not None:
                            for box in r.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                scale_x = (x_end - x) / 640
                                scale_y = (y_end - y) / 640
                                x1 = x1 * scale_x + x
                                y1 = y1 * scale_y + y
                                x2 = x2 * scale_x + x
                                y2 = y2 * scale_y + y
                                
                                all_detections.append({
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'conf': float(box.conf[0]),
                                    'class': int(box.cls[0]),
                                    'class_name': self.class_names[int(box.cls[0])]
                                })
                    pbar.update(1)
        
        print(f"Found {len(all_detections)} raw detections before NMS")
        final_detections = self.apply_nms(all_detections)
        print(f"Found {len(final_detections)} defects after NMS")
        
        vis_img = self.create_visualization(img, final_detections) if visualize else None
        return final_detections, vis_img
    
    def create_visualization(self, original_img, detections):
        """Create color visualization - FIXED"""
        # Convert to BGR for color annotations
        if len(original_img.shape) == 2:
            # Grayscale image
            normalized = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(normalized)
            vis_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            # Already has channels
            normalized = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # Check actual shape after normalization
            if len(normalized.shape) == 2:
                vis_img = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
            elif normalized.shape[2] == 1:
                vis_img = cv2.cvtColor(normalized[:,:,0], cv2.COLOR_GRAY2BGR)
            else:
                vis_img = normalized
        
        colors = [
            (255, 0, 0),      # Blue - porosity
            (0, 255, 0),      # Green - inclusion  
            (0, 0, 255),      # Red - crack
            (255, 255, 0),    # Cyan - undercut
            (255, 0, 255),    # Magenta - lack_of_fusion
            (0, 255, 255),    # Yellow - lack_of_penetration
        ]
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cls = det['class']
            conf = det['conf']
            color = colors[cls % len(colors)]
            
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det['class_name']}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1 - 5, 20)
            
            cv2.rectangle(vis_img, (x1, label_y - 20), 
                         (x1 + label_size[0] + 5, label_y), color, -1)
            cv2.putText(vis_img, label, (x1 + 2, label_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_img
    
    def apply_nms(self, detections):
        if not detections:
            return []
        
        by_class = {}
        for det in detections:
            cls = det['class']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(det)
        
        final = []
        for cls, dets in by_class.items():
            boxes = np.array([d['bbox'] for d in dets])
            scores = np.array([d['conf'] for d in dets])
            indices = self.nms_boxes(boxes, scores, self.nms_threshold)
            for i in indices:
                final.append(dets[i])
        return final
    
    def nms_boxes(self, boxes, scores, threshold):
        if len(boxes) == 0:
            return []
        
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(iou <= threshold)[0] + 1]
        
        return keep

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', default=r"C:\AWrk\SWRD_YOLO_Project\models\yolov8n_20250907_233859\train\weights\best.pt")
    #best:
    #parser.add_argument('--model', default=r"C:\AWrk\SWRD_YOLO_Project\models\yolov8n_20250908_225351\train\weights\best.pt")
    #New:
    parser.add_argument('--model', default=r"C:\AWrk\SWRD_YOLO_Project\models\yolov8n_20250910_025458\train\weights\best.pt")
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--nms', type=float, default=0.45)
    parser.add_argument('--no-viz', action='store_true')
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    output_dir = Path(args.output) if args.output else image_path.parent
    output_dir.mkdir(exist_ok=True)
    
    detector = SlidingWindowInference(args.model)
    detector.conf_threshold = args.conf
    detector.nms_threshold = args.nms
    
    detections, vis_img = detector.process_full_image(image_path, visualize=not args.no_viz)
    
    if vis_img is not None:
        output_path = output_dir / f"{image_path.stem}_detected.jpg"
        cv2.imwrite(str(output_path), vis_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved visualization to {output_path}")
    
    results_path = output_dir / f"{image_path.stem}_detections.json"
    with open(results_path, 'w') as f:
        json.dump({
            'image': str(image_path),
            'total_defects': len(detections) if detections else 0,
            'detections': detections if detections else []
        }, f, indent=2)
    print(f"Saved results to {results_path}")
    
    if detections:
        print("\nDetection summary:")
        class_counts = {}
        for det in detections:
            cls = det['class_name']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")

if __name__ == "__main__":
    main()


# scripts/inference_full_image.py - FIXED
# python scripts\inference_full_image.py 
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
from tqdm import tqdm

class SlidingWindowInference:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.window_size = 320  # Same as training
        self.overlap = 0.5
        self.conf_threshold = 0.25
        self.nms_threshold = 0.45
        self.class_names = ['porosity', 'inclusion', 'crack', 'undercut', 
                           'lack_of_fusion', 'lack_of_penetration']
        
    def process_full_image(self, image_path, visualize=False):
        """Process a full weld image using sliding window"""
        # Read image
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Could not read image: {image_path}")
            return None
        
        # Convert to 3-channel if grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Ensure 8-bit
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        h, w = img.shape[:2]
        
        # Adjust window size if image is smaller
        actual_window_size = min(self.window_size, min(h, w))
        stride = int(actual_window_size * (1 - self.overlap))
        
        all_detections = []
        
        print(f"Processing image {w}x{h} with window {actual_window_size}, stride {stride}")
        
        # Calculate number of windows
        n_windows_x = max(1, (w - actual_window_size) // stride + 1)
        n_windows_y = max(1, (h - actual_window_size) // stride + 1)
        total_windows = n_windows_x * n_windows_y
        
        print(f"Total windows to process: {total_windows}")
        
        # Slide window across image
        with tqdm(total=total_windows, desc="Processing windows") as pbar:
            for y in range(0, max(1, h - actual_window_size + 1), stride):
                for x in range(0, max(1, w - actual_window_size + 1), stride):
                    # Ensure we don't go out of bounds
                    y_end = min(y + actual_window_size, h)
                    x_end = min(x + actual_window_size, w)
                    
                    # Extract patch
                    patch = img[y:y_end, x:x_end]
                    
                    # Resize to expected size if needed
                    if patch.shape[0] != 640 or patch.shape[1] != 640:
                        patch = cv2.resize(patch, (640, 640))
                    
                    # Run inference
                    results = self.model(patch, conf=self.conf_threshold, verbose=False)
                    
                    # Adjust coordinates back to full image
                    for r in results:
                        if r.boxes is not None:
                            for box in r.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                
                                # Scale back if we resized
                                if patch.shape[0] != (y_end - y):
                                    scale_x = (x_end - x) / 640
                                    scale_y = (y_end - y) / 640
                                    x1 *= scale_x
                                    y1 *= scale_y
                                    x2 *= scale_x
                                    y2 *= scale_y
                                
                                # Adjust to full image coordinates
                                x1 += x
                                y1 += y
                                x2 += x
                                y2 += y
                                
                                all_detections.append({
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'conf': float(box.conf[0]),
                                    'class': int(box.cls[0]),
                                    'class_name': self.class_names[int(box.cls[0])]
                                })
                    
                    pbar.update(1)
        
        print(f"Found {len(all_detections)} raw detections before NMS")
        
        # Apply NMS to remove duplicates
        final_detections = self.apply_nms(all_detections)
        
        print(f"Found {len(final_detections)} defects after NMS")
        
        # Visualize if requested
        if visualize:
            vis_img = self.visualize_detections(img, final_detections)
            output_path = Path(image_path).parent / f"{Path(image_path).stem}_detected.jpg"
            cv2.imwrite(str(output_path), vis_img)
            print(f"Saved visualization to {output_path}")
        
        return final_detections
    
    def visualize_detections(self, img, detections):
        """Draw detections on image"""
        vis_img = img.copy()
        colors = [
            (255, 0, 0),    # Blue - porosity
            (0, 255, 0),    # Green - inclusion
            (0, 0, 255),    # Red - crack
            (255, 255, 0),  # Cyan - undercut
            (255, 0, 255),  # Magenta - lack_of_fusion
            (0, 255, 255),  # Yellow - lack_of_penetration
        ]
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cls = det['class']
            conf = det['conf']
            
            color = colors[cls]
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det['class_name']}: {conf:.2f}"
            cv2.putText(vis_img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_img
    
    def apply_nms(self, detections):
        """Apply non-maximum suppression to remove duplicate detections"""
        if not detections:
            return []
        
        # Group by class
        by_class = {}
        for det in detections:
            cls = det['class']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(det)
        
        final = []
        for cls, dets in by_class.items():
            if not dets:
                continue
                
            # Convert to numpy arrays
            boxes = np.array([d['bbox'] for d in dets])
            scores = np.array([d['conf'] for d in dets])
            
            # Apply NMS
            indices = self.nms_boxes(boxes, scores, self.nms_threshold)
            
            for i in indices:
                final.append(dets[i])
        
        return final
    
    def nms_boxes(self, boxes, scores, threshold):
        """Simple NMS implementation"""
        if len(boxes) == 0:
            return []
            
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
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

    def process_directory(self, input_dir, output_dir=None):
        """Process all images in a directory"""
        input_path = Path(input_dir)
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        else:
            output_path = input_path / "detections"
            output_path.mkdir(exist_ok=True)
        
        # Find all tif images
        image_files = list(input_path.glob("*.tif"))
        print(f"Found {len(image_files)} images to process")
        
        all_results = {}
        
        for img_path in tqdm(image_files, desc="Processing images"):
            detections = self.process_full_image(img_path, visualize=True)
            
            # Count detections by class
            counts = {}
            for det in detections:
                cls_name = det['class_name']
                counts[cls_name] = counts.get(cls_name, 0) + 1
            
            all_results[img_path.name] = {
                'total_defects': len(detections),
                'counts_by_class': counts,
                'detections': detections
            }
        
        # Save results
        results_file = output_path / "detection_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        # Print summary
        print("\nSummary:")
        total_defects = sum(r['total_defects'] for r in all_results.values())
        print(f"Total defects found: {total_defects}")
        
        class_totals = {}
        for result in all_results.values():
            for cls, count in result['counts_by_class'].items():
                class_totals[cls] = class_totals.get(cls, 0) + count
        
        print("\nDefects by class:")
        for cls, count in class_totals.items():
            print(f"  {cls}: {count}")

if __name__ == "__main__":
    # Path to your trained model
    model_path = r"C:\AWrk\SWRD_YOLO_Project\models\yolov8n_20250907_233859\train\weights\best.pt"
    
    # Single image test
    test_image = r"C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1\A_DJ-RT-20220623-17.tif"
    
    detector = SlidingWindowInference(model_path)
    
    # Process single image with visualization
    detections = detector.process_full_image(test_image, visualize=True)
    
    if detections:
        print("\nDetection details:")
        for det in detections:
            print(f"  {det['class_name']}: conf={det['conf']:.3f}, bbox={[int(x) for x in det['bbox']]}")
    
    # Process entire directory (uncomment to use)
    # detector.process_directory(
    #     r"C:\AWrk\SWRD_YOLO_Project\data\crop_weld_data\crop_weld_images\L\1",
    #     output_dir="inference_results"
    # )
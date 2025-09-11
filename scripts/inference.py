# scripts/inference.py
import cv2
import torch
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import json

class WeldDefectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = ['porosity', 'inclusion', 'crack', 'undercut', 
                           'lack_of_fusion', 'lack_of_penetration']
        self.colors = [
            (255, 0, 0),    # Blue - porosity
            (0, 255, 0),    # Green - inclusion
            (0, 0, 255),    # Red - crack
            (255, 255, 0),  # Cyan - undercut
            (255, 0, 255),  # Magenta - lack_of_fusion
            (0, 255, 255),  # Yellow - lack_of_penetration
        ]
    
    def detect(self, image_path, conf_threshold=0.25):
        """Run detection on a single image"""
        results = self.model(image_path, conf=conf_threshold)
        return results[0]
    
    def visualize(self, image_path, output_path=None, conf_threshold=0.25):
        """Detect and visualize results"""
        img = cv2.imread(str(image_path))
        results = self.detect(image_path, conf_threshold)
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            # Draw box
            color = self.colors[cls]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{self.class_names[cls]} {conf:.2f}"
            cv2.putText(img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if output_path:
            cv2.imwrite(str(output_path), img)
        
        return img, results

    def detect_batch(self, image_dir, output_dir):
        """Process multiple images"""
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results_all = []
        for img_path in image_dir.glob('*.jpg'):
            _, results = self.visualize(
                img_path, 
                output_dir / f"detected_{img_path.name}"
            )
            results_all.append({
                'image': img_path.name,
                'detections': len(results.boxes),
                'classes': [self.class_names[int(c)] for c in results.boxes.cls]
            })
        
        return results_all
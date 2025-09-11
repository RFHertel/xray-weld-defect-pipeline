# scripts/train_models_optimized.py - FIXED VERSION
#Running the model:
# python scripts/train_models_optimized.py --size n --epochs 100
# This should work without memory errors. The training will be slightly less effective without mosaic augmentation, but it's a necessary tradeoff for your hardware. You'll still get decent results - probably around 0.50-0.55 mAP50 with YOLOv8n, which is quite usable.
# The speed should be around 3-4 iterations per second, so roughly 2-3 hours for 100 epochs. If you need to stop and resume:
# bash# Resume from last checkpoint
# python scripts/train_models_optimized.py --size n --epochs 100 --resume "models/yolov8n_[timestamp]/train/weights/last.pt
import os
import json
import yaml
from pathlib import Path
import subprocess
import sys
import logging
from datetime import datetime
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class OptimizedModelTrainer:
    def __init__(self, data_path, output_dir):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"GPU Memory: {gpu_mem:.1f} GB")
            self.limited_gpu = gpu_mem < 6  # Less than 6GB is limited
        else:
            self.limited_gpu = True
    
    def train_yolov8_optimized(self, model_size='n', epochs=100):
        """Optimized training for limited GPU"""
        from ultralytics import YOLO
        
        # GPU-specific settings for RTX 3050 4GB
        if self.limited_gpu:
            if model_size == 'm':
                batch_size = 8
                imgsz = 640
                workers = 0  # Critical for Windows memory issues
            elif model_size == 's':
                batch_size = 12
                imgsz = 640
                workers = 0
            elif model_size == 'n':
                batch_size = 8
                imgsz = 640
                workers = 2  # Changed from 4 to 0
            else:  # 'l' or 'x'
                batch_size = 4
                imgsz = 512
                workers = 0
        else:
            batch_size = 16
            imgsz = 640
            workers = 8
        
        logging.info(f"Training YOLOv8{model_size} with batch={batch_size}, imgsz={imgsz}")
        
        exp_name = f"yolov8{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir = self.output_dir / exp_name
        
        model = YOLO(f'yolov8{model_size}.pt')
        
        # Optimized parameters for 4GB GPU
        results = model.train(
            data=str(self.data_path / 'dataset.yaml'),
            device=0,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            patience=20,
            save=True,
            project=str(exp_dir),
            name='train',
            exist_ok=True,
            pretrained=True,
            optimizer='SGD',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            plots=True,
            val=True,
            cache=False,  # Don't cache - saves memory
            workers=workers,  # 0 for Windows
            close_mosaic=10,
            amp=True,  # Mixed precision
            # CRITICAL: Disable memory-heavy augmentations
            mosaic=0.0,  # DISABLED - this was causing the memory error
            mixup=0.0,   # DISABLED
            copy_paste=0.0,  # DISABLED
            # Light augmentations only
            hsv_h=0, # No hue for grayscale
            hsv_s=0, # No saturation for grayscale
            hsv_v=0.3, #Brightness variance (X-ray exposure differences)
            degrees=15,  # Welds can be at various angles
            translate=0.15,  # Weld position varies
            scale=0.3,  # Different weld sizes
            shear=5,  # Some shear is OK for industrial
            perspective=0.0002,  # Slight perspective from X-ray angle
            flipud=0.5,
            fliplr=0.5,
        )
        
        # Extract metrics from results
        info = {
            'model': f'yolov8{model_size}',
            'dataset': str(self.data_path),
            'epochs': epochs,
            'batch_size': batch_size,
            'image_size': imgsz,
            'best_weights': str(exp_dir / 'train/weights/best.pt'),
            'last_weights': str(exp_dir / 'train/weights/last.pt')
        }
        
        # Try to extract metrics if available
        try:
            if hasattr(results, 'results_dict'):
                info['metrics'] = {
                    'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                    'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0))
                }
        except:
            pass
        
        with open(exp_dir / 'training_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        logging.info(f"Training complete. Best weights: {info['best_weights']}")
        return info
    
    def train_with_resume(self, model_size='n', epochs=100, resume_from=None):
        """Train with ability to resume from checkpoint"""
        from ultralytics import YOLO
        
        if resume_from and Path(resume_from).exists():
            logging.info(f"Resuming from {resume_from}")
            model = YOLO(resume_from)
            resume = True
        else:
            model = YOLO(f'yolov8{model_size}.pt')
            resume = False
        
        exp_name = f"yolov8{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir = self.output_dir / exp_name
        
        results = model.train(
            data=str(self.data_path / 'dataset.yaml'),
            device=0,
            epochs=epochs,
            batch=16 if model_size == 'n' else 8,
            imgsz=640,
            resume=resume,
            patience=20,
            project=str(exp_dir),
            name='train',
            optimizer='SGD',
            lr0=0.01,
            momentum=0.937,
            workers=0,
            amp=True,
            mosaic=0.0,  # Keep disabled
            mixup=0.0,
            copy_paste=0.0,
            flipud=0.5,
            fliplr=0.5,
        )
        
        return str(exp_dir / 'train/weights/best.pt')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='processed_balanced_final', help='Dataset to use')
    parser.add_argument('--size', default='n', choices=['n', 's', 'm', 'l'],
                       help='Model size (n=nano is fastest)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--resume', help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    trainer = OptimizedModelTrainer(
        data_path=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.data,
        output_dir=Path(r'C:\AWrk\SWRD_YOLO_Project\models')
    )
    
    if args.resume:
        weights = trainer.train_with_resume(args.size, args.epochs, args.resume)
        print(f"Training complete: {weights}")
    else:
        info = trainer.train_yolov8_optimized(args.size, args.epochs)
        print(f"Training complete!")
        print(json.dumps(info, indent=2))

        
# scripts/train_models_balanced.py - WITH ARGUMENTS
import os
import json
from pathlib import Path
import logging
from datetime import datetime
from ultralytics import YOLO
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_balanced.log'),
        logging.StreamHandler()
    ]
)

class BalancedDatasetTrainer:
    def __init__(self, data_path, output_dir):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            self.gpu_memory = gpu_mem
        else:
            self.gpu_memory = 0
            
    def train_efficient(self, model_size='n', epochs=10, batch_size=None, workers=None):
        """
        Efficient training for large balanced dataset
        """
        
        # Auto-detect optimal settings based on GPU
        if batch_size is None:
            if self.gpu_memory >= 12:  # RTX 3060 12GB
                batch_defaults = {'n': 64, 's': 48, 'm': 32, 'l': 24, 'x': 16}
            elif self.gpu_memory >= 8:
                batch_defaults = {'n': 32, 's': 24, 'm': 16, 'l': 12, 'x': 8}
            else:  # RTX 3050 4GB
                batch_defaults = {'n': 16, 's': 12, 'm': 8, 'l': 4, 'x': 2}
            batch_size = batch_defaults.get(model_size, 8)
            
        if workers is None:
            workers = 4 #if self.gpu_memory >= 12 else 2
            
        # Calculate training time estimate
        total_images = 169147  # approximate
        iterations_per_epoch = total_images // batch_size
        
        # Speed estimates
        if self.gpu_memory >= 12:
            speed_estimate = 15 if model_size == 'n' else 10  # iterations/second
        else:
            speed_estimate = 3 if model_size == 'n' else 2
            
        estimated_hours = (iterations_per_epoch * epochs) / speed_estimate / 3600
        
        logging.info(f"Training YOLOv8{model_size}")
        logging.info(f"Dataset: ~{total_images:,} training images")
        logging.info(f"Epochs: {epochs}")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Workers: {workers}")
        logging.info(f"Estimated time: {estimated_hours:.1f} hours")
        
        exp_name = f"yolov8{model_size}_balanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir = self.output_dir / exp_name
        
        model = YOLO(f'yolov8{model_size}.pt')
        
        results = model.train(
            data=str(self.data_path / 'dataset.yaml'),
            device=0,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            
            # NO augmentation - already pre-augmented with class-specific transforms
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.00,  # Tiny brightness variation only
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.0,
            
            # Training settings
            patience=0,  # Early stopping
            cache=False,  # Too large to cache
            workers=workers,
            
            # Optimization
            # optimizer='AdamW',  # Better for pre-augmented data
            # lr0=0.001,
            # lrf=0.01,
            # momentum=0.937,
            # weight_decay=0.0005,
            # warmup_epochs=2,

            # Optimization (these are fine)
            optimizer='AdamW',
            lr0=0.001,  # Default is fine with balanced data
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,  # Default is fine
            
            # Performance
            amp=True,  # Mixed precision
            
            # Save settings
            save=True,
            save_period=5,  # Save every 5 epochs
            project=str(exp_dir),
            name='train',
            exist_ok=True,
            plots=True,
            val=True,
        )
        
        # Save training info
        info = {
            'model': f'yolov8{model_size}',
            'dataset': str(self.data_path),
            'dataset_size': f'~{total_images:,} training images',
            'augmentation': 'Pre-augmented (class-specific)',
            'epochs': epochs,
            'batch_size': batch_size,
            'gpu': f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
            'estimated_time': f'{estimated_hours:.1f} hours',
            'best_weights': str(exp_dir / 'train/weights/best.pt')
        }
        
        with open(exp_dir / 'training_info.json', 'w') as f:
            json.dump(info, f, indent=2)
            
        logging.info(f"Training complete: {info['best_weights']}")
        return info

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size (auto-detect if not specified)')
    parser.add_argument('--workers', type=int, default=None,
                       help='DataLoader workers (auto-detect if not specified)')
    args = parser.parse_args()
    
    trainer = BalancedDatasetTrainer(
        data_path=Path(r'C:\AWrk\SWRD_YOLO_Project\processed_balanced_final'),
        output_dir=Path(r'C:\AWrk\SWRD_YOLO_Project\models')
    )
    
    info = trainer.train_efficient(
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        workers=args.workers
    )
    
    print(json.dumps(info, indent=2))
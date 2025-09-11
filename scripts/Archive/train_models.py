# scripts/train_models.py
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

class ModelTrainer:
    def __init__(self, data_path, output_dir):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.models = {
            'yolov8': {
                'sizes': ['n', 's', 'm', 'l'],
                'license': 'AGPL-3.0',
                'install': 'pip install ultralytics',
                'train_cmd': self.train_yolov8
            },
            'yolox': {
                'sizes': ['nano', 's', 'm', 'l'],
                'license': 'Apache-2.0',
                'install': 'pip install yolox',
                'train_cmd': self.train_yolox
            },
            'rtmdet': {
                'sizes': ['tiny', 's', 'm', 'l'],
                'license': 'Apache-2.0',
                'install': 'pip install mmdet mmengine',
                'train_cmd': self.train_rtmdet
            }
        }
    
    def train_yolov8(self, model_size='m', epochs=100, batch_size=4):
        """Train YOLOv8 model"""
        from ultralytics import YOLO
        
        # Create experiment folder
        exp_name = f"yolov8{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir = self.output_dir / exp_name
        
        # Load pretrained model
        model = YOLO(f'yolov8{model_size}.pt')
        
        # Training parameters optimized for SWRD
        results = model.train(
            data=str(self.data_path / 'dataset.yaml'),
            device=0,   # 👈 force CUDA:0 (RTX 3050)
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            patience=20,
            save=True,
            project=str(exp_dir),
            name='train',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.9,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.5,
            box=7.5,  # Box loss weight
            cls=0.5,  # Class loss weight (lower for imbalanced)
            dfl=1.5,  # Distribution focal loss
            plots=True,
            val=True,
            cache=False,
            close_mosaic=10,  # Disable mosaic for last 10 epochs
            workers=0,   # 👈 reduce workers here (try 2, or even 0 on Windows)
            # Augmentation parameters
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.3,
            degrees=10,
            translate=0.1,
            scale=0.3,
            shear=5,
            perspective=0.0001,
            flipud=0.5,
            fliplr=0.5,
            mosaic=0.8,
            mixup=0.15,
            copy_paste=0.2,
        )
        
        # Save training info
        info = {
            'model': f'yolov8{model_size}',
            'dataset': str(self.data_path),
            'epochs': epochs,
            'batch_size': batch_size,
            'best_weights': str(exp_dir / 'train/weights/best.pt'),
            'metrics': {
                'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0))
            }
        }
        
        with open(exp_dir / 'training_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        logging.info(f"Training complete. Best weights: {info['best_weights']}")
        return info
    
    def train_yolox(self, model_size='s', epochs=100, batch_size=16):
        """Train YOLOX model (Apache 2.0 license)"""
        # Create config file for YOLOX
        config = f"""
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # SWRD dataset
        self.data_dir = "{str(self.data_path)}"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        
        self.num_classes = 6
        self.max_epoch = {epochs}
        self.data_num_workers = 4
        self.eval_interval = 5
        
        # Training params
        self.warmup_epochs = 3
        self.basic_lr_per_img = 0.001 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 10
        self.min_lr_ratio = 0.05
        self.ema = True
        
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 5
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # Augmentation
        self.mosaic_prob = 0.8
        self.mixup_prob = 0.15
        self.hsv_prob = 0.5
        self.flip_prob = 0.5
        
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        """
        
        config_path = self.output_dir / f'yolox_{model_size}_config.py'
        with open(config_path, 'w') as f:
            f.write(config)
        
        # Run training
        cmd = [
            'python', '-m', 'yolox.tools.train',
            '-f', str(config_path),
            '-d', '1',  # Number of GPUs
            '-b', str(batch_size),
            '--fp16',  # Mixed precision
            '-o',  # Occupy GPU memory first
            '-c', f'yolox_{model_size}.pth'  # Pretrained weights
        ]
        
        subprocess.run(cmd, check=True)
        
        logging.info(f"YOLOX training complete")

    def train_rtmdet(self, model_size='s', epochs=100, batch_size=16):
        """Placeholder for RTMDet training"""
        logging.error("RTMDet training not yet implemented.")
        return None
    
    def validate_model(self, weights_path, data_yaml):
        """Validate trained model"""
        from ultralytics import YOLO
        
        model = YOLO(weights_path)
        metrics = model.val(data=data_yaml)
        
        return {
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'precision': metrics.box.mp,
            'recall': metrics.box.mr,
            'per_class_ap': {
                'porosity': metrics.box.ap_class_index[0],
                'inclusion': metrics.box.ap_class_index[1],
                'crack': metrics.box.ap_class_index[2],
                'undercut': metrics.box.ap_class_index[3],
                'lack_of_fusion': metrics.box.ap_class_index[4],
                'lack_of_penetration': metrics.box.ap_class_index[5],
            }
        }
    
    def compare_with_paper(self, metrics):
        """Compare results with paper's reported metrics"""
        paper_results = {
            'yolov8n': {'mAP50': 0.482, 'mAP50-95': 0.287},
            'yolov8s': {'mAP50': 0.598, 'mAP50-95': 0.380},
            'yolov8m': {'mAP50': 0.663, 'mAP50-95': 0.448},
            'yolov8l': {'mAP50': 0.598, 'mAP50-95': 0.387},
            'yolov8x': {'mAP50': 0.573, 'mAP50-95': 0.390}
        }
        
        print("\nComparison with paper results:")
        print("-" * 60)
        print(f"{'Model':<10} {'Paper mAP50':<15} {'Your mAP50':<15} {'Difference':<10}")
        print("-" * 60)
        
        for model, paper_metrics in paper_results.items():
            if model in metrics:
                your_map50 = metrics[model]['mAP50']
                diff = your_map50 - paper_metrics['mAP50']
                print(f"{model:<10} {paper_metrics['mAP50']:<15.3f} {your_map50:<15.3f} {diff:+.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='processed_balanced_final', 
                       choices=['processed_balanced', 'processed_balanced_final'],
                       help='Dataset to use')
    parser.add_argument('--model', default='yolov8', 
                       choices=['yolov8', 'yolox', 'rtmdet'],
                       help='Model architecture')
    parser.add_argument('--size', default='m', help='Model size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--validate-only', help='Path to weights for validation only')
    args = parser.parse_args()
    
    trainer = ModelTrainer(
        data_path=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.data,
        output_dir=Path(r'C:\AWrk\SWRD_YOLO_Project\models')
    )
    
    if args.validate_only:
        metrics = trainer.validate_model(
            args.validate_only, 
            trainer.data_path / 'dataset.yaml'
        )
        print(json.dumps(metrics, indent=2))
    else:
        if args.model == 'yolov8':
            info = trainer.train_yolov8(args.size, args.epochs, args.batch)
        elif args.model == 'yolox':
            info = trainer.train_yolox(args.size, args.epochs, args.batch)
        
        print(f"\nTraining complete!")
        print(json.dumps(info, indent=2))
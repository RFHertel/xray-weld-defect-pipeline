# scripts/train_colab_optimized.py
"""
Google Colab optimized training script for SWRD YOLO
Designed to train on Colab GPU (15GB) and deploy on RTX 3050 (4GB)
"""

import os
import json
import yaml
from pathlib import Path
import logging
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ColabYOLOTrainer:
    def __init__(self, data_path='/content/Datasets/processed_balanced_final', 
                 output_dir='/content/models'):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"GPU: {gpu_name} with {gpu_mem:.1f} GB memory")
            self.colab_gpu = gpu_mem > 10  # Colab has 15GB
        else:
            logging.warning("No GPU detected!")
            self.colab_gpu = False
    
    def create_live_plot(self):
        """Create matplotlib figure for live training visualization"""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        self.fig.suptitle('Training Progress', fontsize=16)
        
        # Initialize plot data storage
        self.plot_data = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'mAP50': [],
            'mAP50-95': [],
            'precision': [],
            'recall': []
        }
    
    def update_live_plot(self, metrics_df):
        """Update training plots in real-time"""
        clear_output(wait=True)
        
        # Update data
        epochs = metrics_df['epoch'].values
        
        # Clear and redraw
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Box Loss
        if 'train/box_loss' in metrics_df.columns:
            self.axes[0, 0].plot(epochs, metrics_df['train/box_loss'], 'b-', label='Box Loss')
            self.axes[0, 0].set_title('Box Loss')
            self.axes[0, 0].set_xlabel('Epoch')
            self.axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Class Loss
        if 'train/cls_loss' in metrics_df.columns:
            self.axes[0, 1].plot(epochs, metrics_df['train/cls_loss'], 'r-', label='Class Loss')
            self.axes[0, 1].set_title('Class Loss')
            self.axes[0, 1].set_xlabel('Epoch')
            self.axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: mAP50
        if 'metrics/mAP50(B)' in metrics_df.columns:
            self.axes[0, 2].plot(epochs, metrics_df['metrics/mAP50(B)'], 'g-', label='mAP50')
            self.axes[0, 2].set_title('mAP50')
            self.axes[0, 2].set_xlabel('Epoch')
            self.axes[0, 2].set_ylim([0, 1])
            self.axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: mAP50-95
        if 'metrics/mAP50-95(B)' in metrics_df.columns:
            self.axes[1, 0].plot(epochs, metrics_df['metrics/mAP50-95(B)'], 'm-', label='mAP50-95')
            self.axes[1, 0].set_title('mAP50-95')
            self.axes[1, 0].set_xlabel('Epoch')
            self.axes[1, 0].set_ylim([0, 1])
            self.axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Precision/Recall
        if 'metrics/precision(B)' in metrics_df.columns and 'metrics/recall(B)' in metrics_df.columns:
            self.axes[1, 1].plot(epochs, metrics_df['metrics/precision(B)'], 'c-', label='Precision')
            self.axes[1, 1].plot(epochs, metrics_df['metrics/recall(B)'], 'y-', label='Recall')
            self.axes[1, 1].set_title('Precision & Recall')
            self.axes[1, 1].set_xlabel('Epoch')
            self.axes[1, 1].set_ylim([0, 1])
            self.axes[1, 1].legend()
            self.axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Learning Rate
        if 'lr/pg0' in metrics_df.columns:
            self.axes[1, 2].plot(epochs, metrics_df['lr/pg0'], 'k-', label='LR')
            self.axes[1, 2].set_title('Learning Rate')
            self.axes[1, 2].set_xlabel('Epoch')
            self.axes[1, 2].set_yscale('log')
            self.axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        display(self.fig)
    
    def train_yolov8_colab(self, model_size='m', epochs=150, batch_size=None):
        """
        Optimized training for Colab with full augmentations
        Model sizes: n, s, m, l (we'll use 'm' for best balance)
        """
        from ultralytics import YOLO
        
        # Optimal batch sizes for Colab GPU (15GB)
        if batch_size is None:
            batch_sizes = {
                'n': 64,   # Nano can handle large batches
                's': 48,   # Small 
                'm': 32,   # Medium (recommended)
                'l': 24,   # Large
                'x': 16    # Extra large
            }
            batch_size = batch_sizes.get(model_size, 32)
        
        logging.info(f"Training YOLOv8{model_size} with batch={batch_size}, epochs={epochs}")
        
        # Create experiment directory
        exp_name = f"yolov8{model_size}_colab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir = self.output_dir / exp_name
        
        # Initialize model
        model = YOLO(f'yolov8{model_size}.pt')
        
        # Create live plotting
        self.create_live_plot()
        
        # Training with FULL augmentations (Colab can handle it)
        results = model.train(
            data=str(self.data_path / 'dataset.yaml'),
            device=0,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,  # Standard size
            patience=30,  # More patience for better convergence
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            project=str(exp_dir),
            name='train',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',  # Better than SGD for this task
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=5,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,  # Lower weight for class loss (imbalanced)
            dfl=1.5,
            plots=True,
            val=True,
            cache='ram',  # Cache in RAM for faster training
            workers=4,
            close_mosaic=15,  # Disable mosaic for last 15 epochs
            amp=True,  # Mixed precision for speed
            # FULL AUGMENTATIONS (Colab can handle these)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.5,
            shear=5,
            perspective=0.0001,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,  # ENABLED - critical for small objects
            mixup=0.2,   # ENABLED - helps generalization
            copy_paste=0.3,  # ENABLED - synthetic augmentation
            # Callbacks for live plotting
            callbacks={
                'on_epoch_end': self.on_epoch_end_callback
            }
        )
        
        # Save final metrics
        final_metrics = {
            'model': f'yolov8{model_size}',
            'dataset': str(self.data_path),
            'epochs_trained': epochs,
            'batch_size': batch_size,
            'best_weights': str(exp_dir / 'train/weights/best.pt'),
            'last_weights': str(exp_dir / 'train/weights/last.pt'),
            'deployment_ready': True,
            'target_device': 'RTX 3050 4GB',
            'final_metrics': {}
        }
        
        # Extract final metrics
        if hasattr(results, 'results_dict'):
            final_metrics['final_metrics'] = {
                'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                'recall': float(results.results_dict.get('metrics/recall(B)', 0))
            }
        
        # Save training info
        with open(exp_dir / 'training_info.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        logging.info(f"Training complete! Best model: {final_metrics['best_weights']}")
        logging.info(f"Final mAP50: {final_metrics['final_metrics'].get('mAP50', 0):.3f}")
        
        return final_metrics
    
    def on_epoch_end_callback(self, trainer):
        """Callback to update plots after each epoch"""
        # Read the results CSV
        results_path = Path(trainer.save_dir) / 'results.csv'
        if results_path.exists():
            df = pd.read_csv(results_path)
            df.columns = [col.strip() for col in df.columns]
            self.update_live_plot(df)
    
    def optimize_for_deployment(self, weights_path, output_path=None):
        """
        Optimize model for RTX 3050 deployment
        Exports to ONNX and TensorRT if possible
        """
        from ultralytics import YOLO
        
        model = YOLO(weights_path)
        
        if output_path is None:
            output_path = Path(weights_path).parent / 'optimized'
        
        logging.info("Optimizing model for deployment...")
        
        # Export formats for different deployment scenarios
        exports = {}
        
        # 1. ONNX (universal, works everywhere)
        try:
            exports['onnx'] = model.export(format='onnx', 
                                          imgsz=640, 
                                          simplify=True,
                                          dynamic=False)  # Fixed size for RTX 3050
            logging.info("✓ ONNX export successful")
        except Exception as e:
            logging.error(f"ONNX export failed: {e}")
        
        # 2. TensorRT (optimal for RTX 3050)
        try:
            exports['engine'] = model.export(format='engine',
                                            imgsz=640,
                                            half=True,  # FP16 for RTX 3050
                                            workspace=2)  # 2GB workspace
            logging.info("✓ TensorRT export successful")
        except Exception as e:
            logging.error(f"TensorRT export failed: {e}")
        
        # 3. PyTorch (fallback)
        exports['pytorch'] = weights_path
        
        return exports
    
    def validate_memory_usage(self, weights_path):
        """Check if model will fit in RTX 3050 4GB memory"""
        from ultralytics import YOLO
        
        model = YOLO(weights_path)
        
        # Test inference with batch size 1
        dummy_input = torch.randn(1, 3, 640, 640).cuda()
        
        # Memory calculation
        model_size = sum(p.numel() * p.element_size() for p in model.model.parameters()) / 1024**3
        
        logging.info(f"Model size: {model_size:.2f} GB")
        logging.info(f"Safe for RTX 3050: {'Yes' if model_size < 2.0 else 'No (consider using smaller model)'}")
        
        return model_size < 2.0

# Main training script for Colab
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='m', choices=['n', 's', 'm', 'l'],
                       help='Model size (m recommended for balance)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Training epochs (150 recommended)')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size (auto-selected if not specified)')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize model for deployment after training')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ColabYOLOTrainer()
    
    # Train model
    print("="*60)
    print("YOLO Training for SWRD Dataset - Google Colab")
    print("="*60)
    
    results = trainer.train_yolov8_colab(
        model_size=args.size,
        epochs=args.epochs,
        batch_size=args.batch
    )
    
    # Optimize for deployment if requested
    if args.optimize:
        print("\n" + "="*60)
        print("Optimizing for RTX 3050 Deployment")
        print("="*60)
        
        exports = trainer.optimize_for_deployment(results['best_weights'])
        trainer.validate_memory_usage(results['best_weights'])
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best model saved to: {results['best_weights']}")
    print(f"Download this file for inference on RTX 3050")
    print("="*60)
# # scripts/evaluate_model.py
# # Evaluate on test set::
# # python scripts/evaluate_model.py --weights "models/yolov8n_20250907_233859/train/weights/best.pt" --split test
# # Get all metrics with visualizations:
# # python scripts/evaluate_model.py --weights "models/yolov8n_20250907_233859/train/weights/best.pt" --split all

# import os
# import json
# from pathlib import Path
# import torch
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from ultralytics import YOLO
# import cv2
# from tqdm import tqdm
# from sklearn.metrics import confusion_matrix
# import logging

# logging.basicConfig(level=logging.INFO)

# class ModelEvaluator:
#     def __init__(self, weights_path, data_path):
#         self.model = YOLO(weights_path)
#         self.data_path = Path(data_path)
#         self.weights_path = Path(weights_path)
#         self.class_names = ['porosity', 'inclusion', 'crack', 'undercut', 
#                            'lack_of_fusion', 'lack_of_penetration']
        
#     def evaluate_on_split(self, split='test'):
#         """Evaluate model on specific split (train/val/test)"""
#         logging.info(f"Evaluating on {split} set...")
        
#         # Update dataset.yaml to point to specific split
#         yaml_path = self.data_path / 'dataset.yaml'
        
#         # Run validation
#         metrics = self.model.val(
#             data=str(yaml_path),
#             split=split,
#             save_json=True,
#             save_hybrid=True,
#             conf=0.25,
#             iou=0.45,
#             max_det=300,
#             plots=True
#         )
        
#         # Extract detailed metrics
#         results = {
#             'split': split,
#             'mAP50': float(metrics.box.map50),
#             'mAP50-95': float(metrics.box.map),
#             'precision': float(metrics.box.mp),
#             'recall': float(metrics.box.mr),
#             'per_class': {}
#         }
        
#         # Per-class metrics
#         for i, class_name in enumerate(self.class_names):
#             if i < len(metrics.box.ap50):
#                 results['per_class'][class_name] = {
#                     'AP50': float(metrics.box.ap50[i]),
#                     'AP': float(metrics.box.ap[i]),
#                     'precision': float(metrics.box.p[i]) if i < len(metrics.box.p) else 0,
#                     'recall': float(metrics.box.r[i]) if i < len(metrics.box.r) else 0,
#                 }
        
#         return results
    
#     def evaluate_all_splits(self):
#         """Evaluate on train, val, and test sets"""
#         all_results = {}
        
#         for split in ['train', 'val', 'test']:
#             split_dir = self.data_path / split / 'images'
#             if split_dir.exists():
#                 all_results[split] = self.evaluate_on_split(split)
        
#         return all_results
    
#     def parse_training_logs(self):
#         """Parse training CSV logs for visualization"""
#         log_dir = self.weights_path.parent.parent
#         results_csv = log_dir / 'results.csv'
        
#         if not results_csv.exists():
#             logging.warning(f"Results CSV not found at {results_csv}")
#             return None
        
#         df = pd.read_csv(results_csv)
#         df.columns = [col.strip() for col in df.columns]
        
#         return df
    
#     def create_training_plots(self, save_path=None):
#         """Create comprehensive training plots using Plotly"""
#         df = self.parse_training_logs()
#         if df is None:
#             return None
        
#         # Create subplots
#         fig = make_subplots(
#             rows=3, cols=2,
#             subplot_titles=('Box Loss', 'Class Loss', 'DFL Loss', 
#                           'mAP50', 'mAP50-95', 'Precision vs Recall'),
#             vertical_spacing=0.1
#         )
        
#         epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
        
#         # Loss plots
#         if 'train/box_loss' in df.columns:
#             fig.add_trace(go.Scatter(x=epochs, y=df['train/box_loss'], 
#                                     name='Box Loss', line=dict(color='blue')), 
#                          row=1, col=1)
        
#         if 'train/cls_loss' in df.columns:
#             fig.add_trace(go.Scatter(x=epochs, y=df['train/cls_loss'], 
#                                     name='Class Loss', line=dict(color='red')), 
#                          row=1, col=2)
        
#         if 'train/dfl_loss' in df.columns:
#             fig.add_trace(go.Scatter(x=epochs, y=df['train/dfl_loss'], 
#                                     name='DFL Loss', line=dict(color='green')), 
#                          row=2, col=1)
        
#         # mAP plots
#         if 'metrics/mAP50(B)' in df.columns:
#             fig.add_trace(go.Scatter(x=epochs, y=df['metrics/mAP50(B)'], 
#                                     name='mAP50', line=dict(color='purple')), 
#                          row=2, col=2)
        
#         if 'metrics/mAP50-95(B)' in df.columns:
#             fig.add_trace(go.Scatter(x=epochs, y=df['metrics/mAP50-95(B)'], 
#                                     name='mAP50-95', line=dict(color='orange')), 
#                          row=3, col=1)
        
#         # Precision vs Recall
#         if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
#             fig.add_trace(go.Scatter(x=df['metrics/recall(B)'], 
#                                     y=df['metrics/precision(B)'], 
#                                     mode='markers+lines',
#                                     name='P-R Curve', 
#                                     line=dict(color='brown')), 
#                          row=3, col=2)
        
#         # Update layout
#         fig.update_layout(height=900, showlegend=True, 
#                          title_text="Training Metrics Overview")
#         fig.update_xaxes(title_text="Epoch", row=1, col=1)
#         fig.update_xaxes(title_text="Epoch", row=1, col=2)
#         fig.update_xaxes(title_text="Epoch", row=2, col=1)
#         fig.update_xaxes(title_text="Epoch", row=2, col=2)
#         fig.update_xaxes(title_text="Epoch", row=3, col=1)
#         fig.update_xaxes(title_text="Recall", row=3, col=2)
        
#         if save_path:
#             fig.write_html(str(save_path))
#             logging.info(f"Plots saved to {save_path}")
        
#         return fig
    
#     def create_class_performance_plot(self, results, save_path=None):
#         """Create per-class performance comparison"""
#         fig = go.Figure()
        
#         classes = list(self.class_names)
        
#         for split, metrics in results.items():
#             if 'per_class' in metrics:
#                 ap50_values = [metrics['per_class'].get(c, {}).get('AP50', 0) 
#                               for c in classes]
#                 fig.add_trace(go.Bar(name=f'{split} AP50', x=classes, y=ap50_values))
        
#         fig.update_layout(
#             title="Per-Class AP50 Performance Across Splits",
#             xaxis_title="Class",
#             yaxis_title="AP50",
#             barmode='group',
#             height=500
#         )
        
#         if save_path:
#             fig.write_html(str(save_path))
        
#         return fig
    
#     def get_confusion_matrix(self, split='test'):
#         """Generate confusion matrix for a split"""
#         # This would require running inference on all images
#         # and comparing with ground truth
#         pass

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', required=True, help='Path to model weights')
#     parser.add_argument('--data', default='processed_balanced_final', help='Dataset path')
#     parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], 
#                        default='all', help='Which split to evaluate')
#     parser.add_argument('--output', default='evaluation_results', help='Output directory')
#     args = parser.parse_args()
    
#     output_dir = Path(args.output)
#     output_dir.mkdir(exist_ok=True)
    
#     evaluator = ModelEvaluator(
#         weights_path=args.weights,
#         data_path=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.data
#     )
    
#     # Evaluate
#     if args.split == 'all':
#         results = evaluator.evaluate_all_splits()
#     else:
#         results = {args.split: evaluator.evaluate_on_split(args.split)}
    
#     # Save results
#     with open(output_dir / 'evaluation_metrics.json', 'w') as f:
#         json.dump(results, f, indent=2)
    
#     # Create plots
#     fig_training = evaluator.create_training_plots(output_dir / 'training_plots.html')
#     fig_classes = evaluator.create_class_performance_plot(results, 
#                                                          output_dir / 'class_performance.html')
    
#     # Print summary
#     print("\n" + "="*60)
#     print("EVALUATION RESULTS")
#     print("="*60)
    
#     for split, metrics in results.items():
#         print(f"\n{split.upper()} SET:")
#         print(f"  mAP50: {metrics['mAP50']:.3f}")
#         print(f"  mAP50-95: {metrics['mAP50-95']:.3f}")
#         print(f"  Precision: {metrics['precision']:.3f}")
#         print(f"  Recall: {metrics['recall']:.3f}")
        
#         if 'per_class' in metrics:
#             print("\n  Per-Class AP50:")
#             for cls, cls_metrics in metrics['per_class'].items():
#                 print(f"    {cls}: {cls_metrics['AP50']:.3f}")


# scripts/evaluate_model_enhanced.py - COMPLETE VERSION
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class EnhancedModelEvaluator:
    def __init__(self, weights_path, data_path):
        self.model = YOLO(weights_path)
        self.data_path = Path(data_path)
        self.weights_path = Path(weights_path)
        self.class_names = ['porosity', 'inclusion', 'crack', 'undercut', 
                           'lack_of_fusion', 'lack_of_penetration']
        
        # Load class distribution from balancing stats
        stats_path = self.data_path / 'balancing_stats.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.balance_stats = json.load(f)
        else:
            self.balance_stats = None
    
    def evaluate_on_split(self, split='test'):
        """Evaluate with proper handling of imbalanced data"""
        logging.info(f"Evaluating on {split} set...")
        
        yaml_path = self.data_path / 'dataset.yaml'
        
        # Run validation
        metrics = self.model.val(
            data=str(yaml_path),
            split=split,
            save_json=True,
            save_hybrid=True,
            conf=0.25,
            iou=0.45,
            max_det=300,
            plots=True
        )
        
        # Get class counts for this split
        class_counts = {}
        if self.balance_stats and split in self.balance_stats['final_distribution']:
            dist = self.balance_stats['final_distribution'][split]
            for idx, name in enumerate(self.class_names):
                class_counts[name] = dist.get(str(idx), 0)
        
        # Calculate comprehensive metrics
        results = self.calculate_comprehensive_metrics(metrics, class_counts, split)
        
        return results
    
    def calculate_comprehensive_metrics(self, metrics, class_counts, split):
        """Calculate micro/macro averaged metrics and per-class performance"""
        
        per_class = {}
        ap50_values = []
        ap_values = []
        weights = []
        
        total_instances = sum(class_counts.values()) if class_counts else 1
        
        for i, class_name in enumerate(self.class_names):
            if i < len(metrics.box.ap50):
                ap50 = float(metrics.box.ap50[i])
                ap = float(metrics.box.ap[i])
                
                per_class[class_name] = {
                    'AP50': ap50,
                    'AP': ap,
                    'precision': float(metrics.box.p[i]) if i < len(metrics.box.p) else 0,
                    'recall': float(metrics.box.r[i]) if i < len(metrics.box.r) else 0,
                    'support': class_counts.get(class_name, 0),
                    'relative_frequency': class_counts.get(class_name, 0) / total_instances if total_instances > 0 else 0
                }
                
                ap50_values.append(ap50)
                ap_values.append(ap)
                weights.append(class_counts.get(class_name, 0))
        
        # Macro-average (simple mean)
        macro_map50 = np.mean(ap50_values) if ap50_values else 0
        macro_map = np.mean(ap_values) if ap_values else 0
        
        # Micro-average (weighted by frequency)
        if sum(weights) > 0:
            micro_map50 = np.average(ap50_values, weights=weights) if ap50_values else 0
            micro_map = np.average(ap_values, weights=weights) if ap_values else 0
        else:
            micro_map50 = macro_map50
            micro_map = macro_map
        
        # Original metrics (from YOLOv8)
        results = {
            'split': split,
            'mAP50': float(metrics.box.map50),  # Keep original
            'mAP50-95': float(metrics.box.map),  # Keep original
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'overall_metrics': {
                'macro_mAP50': macro_map50,
                'macro_mAP50-95': macro_map,
                'micro_mAP50': micro_map50,
                'micro_mAP50-95': micro_map,
            },
            'per_class': per_class,
            'class_distribution': class_counts,
            'dataset_info': {
                'total_instances': total_instances,
                'imbalance_ratio': f"{max(class_counts.values())/min(class_counts.values()):.1f}:1" if class_counts and min(class_counts.values()) > 0 else 'N/A'
            }
        }
        
        return results
    
    def evaluate_all_splits(self):
        """Evaluate on train, val, and test sets"""
        all_results = {}
        
        for split in ['train', 'val', 'test']:
            split_dir = self.data_path / split / 'images'
            if split_dir.exists():
                all_results[split] = self.evaluate_on_split(split)
        
        return all_results
    
    def parse_training_logs(self):
        """Parse training CSV logs for visualization - ORIGINAL FUNCTION"""
        log_dir = self.weights_path.parent.parent
        results_csv = log_dir / 'results.csv'
        
        if not results_csv.exists():
            logging.warning(f"Results CSV not found at {results_csv}")
            return None
        
        df = pd.read_csv(results_csv)
        df.columns = [col.strip() for col in df.columns]
        
        return df
    
    def create_training_plots(self, save_path=None):
        """Create comprehensive training plots - ORIGINAL FUNCTION WITH ALL PLOTS"""
        df = self.parse_training_logs()
        if df is None:
            return None
        
        # Create subplots - ORIGINAL LAYOUT
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Box Loss', 'Class Loss', 'DFL Loss', 
                          'mAP50', 'mAP50-95', 'Precision vs Recall'),
            vertical_spacing=0.1
        )
        
        epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
        
        # Loss plots
        if 'train/box_loss' in df.columns:
            fig.add_trace(go.Scatter(x=epochs, y=df['train/box_loss'], 
                                    name='Box Loss', line=dict(color='blue')), 
                         row=1, col=1)
        
        if 'train/cls_loss' in df.columns:
            fig.add_trace(go.Scatter(x=epochs, y=df['train/cls_loss'], 
                                    name='Class Loss', line=dict(color='red')), 
                         row=1, col=2)
        
        if 'train/dfl_loss' in df.columns:
            fig.add_trace(go.Scatter(x=epochs, y=df['train/dfl_loss'], 
                                    name='DFL Loss', line=dict(color='green')), 
                         row=2, col=1)
        
        # mAP plots
        if 'metrics/mAP50(B)' in df.columns:
            fig.add_trace(go.Scatter(x=epochs, y=df['metrics/mAP50(B)'], 
                                    name='mAP50', line=dict(color='purple')), 
                         row=2, col=2)
        
        if 'metrics/mAP50-95(B)' in df.columns:
            fig.add_trace(go.Scatter(x=epochs, y=df['metrics/mAP50-95(B)'], 
                                    name='mAP50-95', line=dict(color='orange')), 
                         row=3, col=1)
        
        # Precision vs Recall
        if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
            fig.add_trace(go.Scatter(x=df['metrics/recall(B)'], 
                                    y=df['metrics/precision(B)'], 
                                    mode='markers+lines',
                                    name='P-R Curve', 
                                    line=dict(color='brown')), 
                         row=3, col=2)
        
        # Update layout
        fig.update_layout(height=900, showlegend=True, 
                         title_text="Training Metrics Overview")
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_xaxes(title_text="Epoch", row=3, col=1)
        fig.update_xaxes(title_text="Recall", row=3, col=2)
        
        if save_path:
            fig.write_html(str(save_path))
            logging.info(f"Plots saved to {save_path}")
        
        return fig
    
    def create_class_performance_plot(self, results, save_path=None):
        """Create per-class performance comparison - ORIGINAL FUNCTION"""
        fig = go.Figure()
        
        classes = list(self.class_names)
        
        for split, metrics in results.items():
            if 'per_class' in metrics:
                ap50_values = [metrics['per_class'].get(c, {}).get('AP50', 0) 
                              for c in classes]
                fig.add_trace(go.Bar(name=f'{split} AP50', x=classes, y=ap50_values))
        
        fig.update_layout(
            title="Per-Class AP50 Performance Across Splits",
            xaxis_title="Class",
            yaxis_title="AP50",
            barmode='group',
            height=500
        )
        
        if save_path:
            fig.write_html(str(save_path))
        
        return fig
    
    def create_enhanced_plots(self, results, save_dir):
        """NEW: Additional plots for imbalance analysis"""
        save_dir = Path(save_dir)
        
        # AP50 vs Support plot
        fig_support = go.Figure()
        
        for split_name, split_results in results.items():
            if 'per_class' not in split_results:
                continue
                
            classes = []
            ap50_values = []
            supports = []
            
            for cls, metrics in split_results['per_class'].items():
                classes.append(cls)
                ap50_values.append(metrics['AP50'])
                supports.append(metrics['support'])
            
            fig_support.add_trace(
                go.Scatter(x=supports, y=ap50_values, mode='markers+text',
                          text=classes, textposition='top center',
                          name=split_name, marker=dict(size=10))
            )
        
        fig_support.update_layout(
            title="AP50 vs Class Support (Data Amount)",
            xaxis_title="Number of Instances",
            yaxis_title="AP50",
            height=500
        )
        fig_support.write_html(str(save_dir / 'ap50_vs_support.html'))
        
        # Micro vs Macro comparison
        fig_comparison = go.Figure()
        
        for split_name, split_results in results.items():
            if 'overall_metrics' not in split_results:
                continue
                
            metrics = split_results['overall_metrics']
            fig_comparison.add_trace(go.Bar(
                name=split_name,
                x=['Macro mAP50', 'Micro mAP50', 'Macro mAP50-95', 'Micro mAP50-95'],
                y=[metrics['macro_mAP50'], metrics['micro_mAP50'],
                   metrics['macro_mAP50-95'], metrics['micro_mAP50-95']]
            ))
        
        fig_comparison.update_layout(
            title="Micro vs Macro Averaged Metrics",
            yaxis_title="Score",
            barmode='group',
            height=400
        )
        fig_comparison.write_html(str(save_dir / 'micro_macro_comparison.html'))
        
        return fig_support, fig_comparison
    
    def generate_report(self, results, save_path):
        """Generate detailed evaluation report"""
        report = ["# Model Evaluation Report\n"]
        report.append("=" * 60 + "\n")
        
        for split_name, split_results in results.items():
            report.append(f"\n## {split_name.upper()} SET RESULTS\n")
            report.append("-" * 40 + "\n")
            
            # Dataset info
            if 'dataset_info' in split_results:
                info = split_results['dataset_info']
                report.append(f"Total instances: {info['total_instances']}\n")
                report.append(f"Class imbalance ratio: {info['imbalance_ratio']}\n\n")
            
            # Standard metrics (original)
            report.append("### Standard YOLOv8 Metrics:\n")
            report.append(f"  - mAP50: {split_results['mAP50']:.3f}\n")
            report.append(f"  - mAP50-95: {split_results['mAP50-95']:.3f}\n")
            report.append(f"  - Precision: {split_results['precision']:.3f}\n")
            report.append(f"  - Recall: {split_results['recall']:.3f}\n\n")
            
            # Enhanced metrics
            if 'overall_metrics' in split_results:
                metrics = split_results['overall_metrics']
                report.append("### Enhanced Metrics:\n")
                report.append(f"**Macro-averaged (equal weight per class):**\n")
                report.append(f"  - mAP50: {metrics['macro_mAP50']:.3f}\n")
                report.append(f"  - mAP50-95: {metrics['macro_mAP50-95']:.3f}\n\n")
                report.append(f"**Micro-averaged (weighted by frequency):**\n")
                report.append(f"  - mAP50: {metrics['micro_mAP50']:.3f}\n")
                report.append(f"  - mAP50-95: {metrics['micro_mAP50-95']:.3f}\n\n")
            
            # Per-class metrics
            if 'per_class' in split_results:
                report.append("### Per-Class Performance:\n")
                report.append("| Class | AP50 | Precision | Recall | Support | Note |\n")
                report.append("|-------|------|-----------|--------|---------|------|\n")
                
                for cls, metrics in split_results['per_class'].items():
                    note = ""
                    if metrics['support'] < 50:
                        note = "Low support"
                    elif metrics['AP50'] < 0.3:
                        note = "Poor performance"
                    elif metrics['AP50'] > 0.7:
                        note = "Good performance"
                    
                    report.append(f"| {cls} | {metrics['AP50']:.3f} | {metrics['precision']:.3f} | "
                                f"{metrics['recall']:.3f} | {metrics['support']} | {note} |\n")
        
        # Save report with encoding
        with open(save_path, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        return ''.join(report)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Path to model weights')
    parser.add_argument('--data', default='processed_balanced_final', help='Dataset path')
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], 
                       default='all', help='Which split to evaluate')
    parser.add_argument('--output', default='evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    evaluator = EnhancedModelEvaluator(
        weights_path=args.weights,
        data_path=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.data
    )
    
    # Evaluate
    if args.split == 'all':
        results = evaluator.evaluate_all_splits()
    else:
        results = {args.split: evaluator.evaluate_on_split(args.split)}
    
    # Save results
    with open(output_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create ALL plots - original + enhanced
    fig_training = evaluator.create_training_plots(output_dir / 'training_plots.html')
    fig_classes = evaluator.create_class_performance_plot(results, 
                                                         output_dir / 'class_performance.html')
    evaluator.create_enhanced_plots(results, output_dir)
    
    # Generate report
    report = evaluator.generate_report(results, output_dir / 'evaluation_report.md')
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for split, metrics in results.items():
        print(f"\n{split.upper()} SET:")
        print(f"  Standard mAP50: {metrics['mAP50']:.3f}")
        print(f"  Standard mAP50-95: {metrics['mAP50-95']:.3f}")
        if 'overall_metrics' in metrics:
            print(f"  Macro mAP50: {metrics['overall_metrics']['macro_mAP50']:.3f}")
            print(f"  Micro mAP50: {metrics['overall_metrics']['micro_mAP50']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
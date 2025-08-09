"""
Evaluation module for YOLO vehicle detection project.

This module provides comprehensive evaluation functionality for YOLO models with
detailed metrics calculation, visualization, and reporting capabilities.

Usage:
    # Basic evaluation
    python evaluation.py --model best.pt --data data.yaml
    
    # Evaluation with custom thresholds
    python evaluation.py --model best.pt --data data.yaml --conf_thresh 0.5 --iou_thresh 0.6
    
    # Save detailed reports
    python evaluation.py --model best.pt --data data.yaml --save_plots --save_report
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import sys
import os

# Add current directory to Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
import torch

from config import Config, get_config
from utils import (
    setup_logging,
    NeptuneLogger,
    create_directories,
    save_json,
    format_time,
    print_system_info,
)


class YOLOEvaluator:
    """
    YOLO model evaluator with comprehensive metrics and analysis.
    
    This class handles model evaluation including:
    - Performance metrics calculation
    - Per-class analysis
    - Confusion matrix generation
    - Visualization and reporting
    - Neptune logging integration
    """
    
    def __init__(
        self,
        model_path: str,
        data_yaml: str,
        config: Optional[Config] = None,
        device: Union[str, int] = 'auto'
    ):
        """
        Initialize the YOLO evaluator.
        
        Args:
            model_path: Path to the trained YOLO model
            data_yaml: Path to dataset YAML configuration
            config: Configuration object. If None, uses default config
            device: Device for evaluation ('auto', 'cpu', 0, 1, etc.)
        """
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.config = config if config is not None else get_config()
        self.device = self._determine_device(device)
        
        # Setup logging
        self.logger = setup_logging(log_level="INFO")
        
        # Initialize Neptune logging if configured
        self.neptune_logger = None
        if hasattr(self.config.neptune, 'api_token') and self.config.neptune.api_token:
            self.neptune_logger = NeptuneLogger(self.config.neptune, "evaluation_run")
        
        # Initialize model and metrics
        self.model: Optional[YOLO] = None
        self.class_names: List[str] = self.config.data.class_names
        self.num_classes: int = self.config.data.num_classes
        
        # Evaluation results storage
        self.evaluation_results: Dict[str, Any] = {}
        self.confusion_matrix: Optional[np.ndarray] = None
        
        self.logger.info(f"YOLO Evaluator initialized with device: {self.device}")
    
    def _determine_device(self, device: Union[str, int]) -> Union[str, int]:
        """Determine the best device for evaluation."""
        if device == 'auto':
            if torch.cuda.is_available():
                return 0
            else:
                return 'cpu'
        return device
    
    def load_model(self) -> None:
        """Load the YOLO model from the specified path."""
        try:
            self.logger.info(f"Loading model from: {self.model_path}")
            
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            self.logger.info("Model loaded successfully")
            
            # Log model info to Neptune
            if self.neptune_logger and self.neptune_logger.is_active:
                model_info = {
                    "model_path": self.model_path,
                    "data_yaml": self.data_yaml,
                    "device": str(self.device),
                    "class_names": self.class_names,
                    "num_classes": self.num_classes
                }
                self.neptune_logger.log_hyperparameters({"evaluation": model_info})
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def validate_dataset(self) -> bool:
        """
        Validate the dataset configuration and paths.
        
        Returns:
            bool: True if dataset is valid, False otherwise
        """
        try:
            if not Path(self.data_yaml).exists():
                self.logger.error(f"Data YAML file not found: {self.data_yaml}")
                return False
            
            # Load and check YAML content
            import yaml
            with open(self.data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            
            required_keys = ['path', 'val', 'names']
            for key in required_keys:
                if key not in data_config:
                    self.logger.error(f"Missing required key in data.yaml: {key}")
                    return False
            
            # Check if validation path exists
            val_path = Path(data_config['path']) / data_config['val']
            if not val_path.exists():
                self.logger.error(f"Validation path not found: {val_path}")
                return False
            
            self.logger.info("Dataset validation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset validation failed: {e}")
            return False
    
    def run_evaluation(
        self,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        image_size: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive model evaluation.
        
        Args:
            confidence_threshold: Confidence threshold for evaluation
            iou_threshold: IoU threshold for evaluation
            image_size: Image size for evaluation
            batch_size: Batch size for evaluation
            
        Returns:
            Dict containing all evaluation results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Use config defaults if not specified
        conf_thresh = confidence_threshold or self.config.evaluation.confidence_threshold
        iou_thresh = iou_threshold or self.config.evaluation.iou_threshold
        img_size = image_size or self.config.evaluation.image_size
        batch_size = batch_size or self.config.evaluation.batch_size
        
        self.logger.info("Starting comprehensive model evaluation...")
        start_time = time.time()
        
        try:
            # Run YOLO validation
            self.logger.info("Running YOLO validation...")
            val_results = self.model.val(
                data=self.data_yaml,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=img_size,
                batch=batch_size,
                device=self.device,
                plots=self.config.evaluation.plots,
                save_json=self.config.evaluation.save_json,
                verbose=self.config.evaluation.verbose
            )
            
            # Extract key metrics
            metrics = self._extract_metrics(val_results)
            
            # Calculate additional metrics
            additional_metrics = self._calculate_additional_metrics(val_results)
            metrics.update(additional_metrics)
            
            # Generate confusion matrix
            confusion_matrix = self._generate_confusion_matrix(val_results)
            
            # Calculate per-class metrics
            per_class_metrics = self._calculate_per_class_metrics(val_results)
            
            evaluation_time = time.time() - start_time
            
            # Compile all results
            self.evaluation_results = {
                "model_path": self.model_path,
                "data_yaml": self.data_yaml,
                "evaluation_time": evaluation_time,
                "parameters": {
                    "confidence_threshold": conf_thresh,
                    "iou_threshold": iou_thresh,
                    "image_size": img_size,
                    "batch_size": batch_size
                },
                "overall_metrics": metrics,
                "per_class_metrics": per_class_metrics,
                "confusion_matrix": confusion_matrix.tolist() if confusion_matrix is not None else None,
                "class_names": self.class_names,
                "num_classes": self.num_classes
            }
            
            self.logger.info(f"Evaluation completed in {format_time(evaluation_time)}")
            self._log_summary()
            
            # Log to Neptune
            if self.neptune_logger and self.neptune_logger.is_active:
                self._log_to_neptune()
            
            return self.evaluation_results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise
    
    def _extract_metrics(self, val_results) -> Dict[str, float]:
        """Extract key metrics from YOLO validation results."""
        metrics = {}
        
        try:
            # Access metrics from results
            if hasattr(val_results, 'box'):
                box_results = val_results.box
                
                # mAP metrics
                if hasattr(box_results, 'map'):
                    metrics['mAP@0.5'] = float(box_results.map50)
                    metrics['mAP@0.5:0.95'] = float(box_results.map)
                
                # Precision and Recall
                if hasattr(box_results, 'mp'):
                    metrics['precision'] = float(box_results.mp)
                if hasattr(box_results, 'mr'):
                    metrics['recall'] = float(box_results.mr)
                
                # F1 Score
                if 'precision' in metrics and 'recall' in metrics:
                    p, r = metrics['precision'], metrics['recall']
                    metrics['f1_score'] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                
                # Per-class mAP
                if hasattr(box_results, 'ap_class_index') and hasattr(box_results, 'ap'):
                    for i, class_idx in enumerate(box_results.ap_class_index):
                        class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Class_{class_idx}"
                        metrics[f'mAP@0.5_{class_name}'] = float(box_results.ap50[i])
                        metrics[f'mAP@0.5:0.95_{class_name}'] = float(box_results.ap[i])
            
        except Exception as e:
            self.logger.warning(f"Could not extract some metrics: {e}")
        
        return metrics
    
    def _calculate_additional_metrics(self, val_results) -> Dict[str, Any]:
        """Calculate additional custom metrics."""
        additional_metrics = {}
        
        try:
            # Speed metrics
            if hasattr(val_results, 'speed'):
                speed = val_results.speed
                additional_metrics.update({
                    'preprocess_time_ms': speed.get('preprocess', 0),
                    'inference_time_ms': speed.get('inference', 0),
                    'postprocess_time_ms': speed.get('postprocess', 0),
                    'total_time_ms': sum(speed.values())
                })
            
            # Model complexity (if available)
            if self.model and hasattr(self.model.model, 'model'):
                try:
                    # Count parameters
                    total_params = sum(p.numel() for p in self.model.model.parameters())
                    trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
                    
                    additional_metrics.update({
                        'total_parameters': int(total_params),
                        'trainable_parameters': int(trainable_params),
                        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
                    })
                except Exception:
                    pass
            
        except Exception as e:
            self.logger.warning(f"Could not calculate additional metrics: {e}")
        
        return additional_metrics
    
    def _generate_confusion_matrix(self, val_results) -> Optional[np.ndarray]:
        """Generate confusion matrix from validation results."""
        try:
            if hasattr(val_results, 'confusion_matrix') and val_results.confusion_matrix is not None:
                # YOLO provides confusion matrix
                self.confusion_matrix = val_results.confusion_matrix.matrix
                return self.confusion_matrix
            else:
                self.logger.warning("Confusion matrix not available from validation results")
                return None
                
        except Exception as e:
            self.logger.warning(f"Could not generate confusion matrix: {e}")
            return None
    
    def _calculate_per_class_metrics(self, val_results) -> Dict[str, Dict[str, float]]:
        """Calculate detailed per-class metrics."""
        per_class_metrics = {}
        
        try:
            if hasattr(val_results, 'box') and hasattr(val_results.box, 'ap_class_index'):
                box_results = val_results.box
                
                for i, class_idx in enumerate(box_results.ap_class_index):
                    class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Class_{class_idx}"
                    
                    class_metrics = {
                        'class_id': int(class_idx),
                        'mAP@0.5': float(box_results.ap50[i]) if i < len(box_results.ap50) else 0.0,
                        'mAP@0.5:0.95': float(box_results.ap[i]) if i < len(box_results.ap) else 0.0,
                    }
                    
                    # Add precision and recall if available
                    if hasattr(box_results, 'p') and i < len(box_results.p):
                        class_metrics['precision'] = float(box_results.p[i])
                    if hasattr(box_results, 'r') and i < len(box_results.r):
                        class_metrics['recall'] = float(box_results.r[i])
                    
                    # Calculate F1 score
                    if 'precision' in class_metrics and 'recall' in class_metrics:
                        p, r = class_metrics['precision'], class_metrics['recall']
                        class_metrics['f1_score'] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                    
                    per_class_metrics[class_name] = class_metrics
            
        except Exception as e:
            self.logger.warning(f"Could not calculate per-class metrics: {e}")
        
        return per_class_metrics
    
    def generate_evaluation_report(self, output_dir: Union[str, Path] = "evaluation_results") -> Path:
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report file
        """
        output_dir = Path(output_dir)
        create_directories([output_dir])
        
        report_file = output_dir / "evaluation_report.json"
        
        # Create detailed report
        report = {
            "evaluation_summary": {
                "model_path": self.model_path,
                "data_yaml": self.data_yaml,
                "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_time": self.evaluation_results.get("evaluation_time", 0),
                "parameters": self.evaluation_results.get("parameters", {})
            },
            "overall_performance": self.evaluation_results.get("overall_metrics", {}),
            "per_class_performance": self.evaluation_results.get("per_class_metrics", {}),
            "class_names": self.class_names,
            "recommendations": self._generate_recommendations()
        }
        
        # Save report
        save_json(report, report_file, indent=2)
        
        self.logger.info(f"Evaluation report saved to: {report_file}")
        return report_file
    
    def plot_confusion_matrix(
        self,
        output_dir: Union[str, Path] = "evaluation_results",
        normalize: bool = True,
        figsize: Tuple[int, int] = (10, 8)
    ) -> Optional[Path]:
        """
        Plot and save confusion matrix.
        
        Args:
            output_dir: Directory to save the plot
            normalize: Whether to normalize the confusion matrix
            figsize: Figure size for the plot
            
        Returns:
            Path to the saved plot file or None if matrix not available
        """
        if self.confusion_matrix is None:
            self.logger.warning("Confusion matrix not available")
            return None
        
        output_dir = Path(output_dir)
        create_directories([output_dir])
        
        # Prepare confusion matrix
        cm = self.confusion_matrix.copy()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)
        
        # Create plot
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names + ['Background'],
            yticklabels=self.class_names + ['Background'],
            cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
        )
        
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / f"confusion_matrix{'_normalized' if normalize else ''}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrix plot saved to: {plot_file}")
        
        # Log to Neptune
        if self.neptune_logger and self.neptune_logger.is_active:
            self.neptune_logger.log_image(str(plot_file), f"confusion_matrix{'_normalized' if normalize else ''}")
        
        return plot_file
    
    def plot_per_class_metrics(
        self,
        output_dir: Union[str, Path] = "evaluation_results",
        figsize: Tuple[int, int] = (12, 8)
    ) -> Optional[Path]:
        """
        Plot per-class performance metrics.
        
        Args:
            output_dir: Directory to save the plot
            figsize: Figure size for the plot
            
        Returns:
            Path to the saved plot file
        """
        if not self.evaluation_results.get("per_class_metrics"):
            self.logger.warning("Per-class metrics not available")
            return None
        
        output_dir = Path(output_dir)
        create_directories([output_dir])
        
        per_class_metrics = self.evaluation_results["per_class_metrics"]
        
        # Prepare data for plotting
        classes = list(per_class_metrics.keys())
        metrics_to_plot = ['precision', 'recall', 'f1_score', 'mAP@0.5', 'mAP@0.5:0.95']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            values = [per_class_metrics[cls].get(metric, 0) for cls in classes]
            
            ax = axes[i]
            bars = ax.bar(classes, values, alpha=0.7)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Rotate x-axis labels if needed
            if len(max(classes, key=len)) > 8:
                ax.tick_params(axis='x', rotation=45)
        
        # Remove empty subplot
        if len(metrics_to_plot) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / "per_class_metrics.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Per-class metrics plot saved to: {plot_file}")
        
        # Log to Neptune
        if self.neptune_logger and self.neptune_logger.is_active:
            self.neptune_logger.log_image(str(plot_file), "per_class_metrics")
        
        return plot_file
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        overall_metrics = self.evaluation_results.get("overall_metrics", {})
        per_class_metrics = self.evaluation_results.get("per_class_metrics", {})
        
        # Overall performance recommendations
        map_50 = overall_metrics.get("mAP@0.5", 0)
        map_50_95 = overall_metrics.get("mAP@0.5:0.95", 0)
        precision = overall_metrics.get("precision", 0)
        recall = overall_metrics.get("recall", 0)
        
        if map_50 < 0.5:
            recommendations.append("Overall mAP@0.5 is below 50%. Consider increasing training epochs or adjusting hyperparameters.")
        
        if map_50_95 < 0.3:
            recommendations.append("mAP@0.5:0.95 is low. Model struggles with precise localization. Consider using higher resolution images.")
        
        if precision < 0.6:
            recommendations.append("Low precision detected. Model produces many false positives. Consider increasing confidence threshold.")
        
        if recall < 0.6:
            recommendations.append("Low recall detected. Model misses many objects. Consider data augmentation or collecting more training data.")
        
        # Per-class recommendations
        if per_class_metrics:
            worst_class = min(per_class_metrics.items(), key=lambda x: x[1].get('mAP@0.5', 0))
            best_class = max(per_class_metrics.items(), key=lambda x: x[1].get('mAP@0.5', 0))
            
            worst_map = worst_class[1].get('mAP@0.5', 0)
            best_map = best_class[1].get('mAP@0.5', 0)
            
            if best_map - worst_map > 0.3:
                recommendations.append(f"Large performance gap between classes. {worst_class[0]} (mAP: {worst_map:.3f}) performs much worse than {best_class[0]} (mAP: {best_map:.3f}). Consider collecting more data for {worst_class[0]}.")
        
        if not recommendations:
            recommendations.append("Model performance looks good overall. Consider fine-tuning for specific use cases.")
        
        return recommendations
    
    def _log_summary(self) -> None:
        """Log evaluation summary to console."""
        overall_metrics = self.evaluation_results.get("overall_metrics", {})
        
        self.logger.info("\n" + "="*50)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("="*50)
        
        if "mAP@0.5" in overall_metrics:
            self.logger.info(f"mAP@0.5: {overall_metrics['mAP@0.5']:.4f}")
        if "mAP@0.5:0.95" in overall_metrics:
            self.logger.info(f"mAP@0.5:0.95: {overall_metrics['mAP@0.5:0.95']:.4f}")
        if "precision" in overall_metrics:
            self.logger.info(f"Precision: {overall_metrics['precision']:.4f}")
        if "recall" in overall_metrics:
            self.logger.info(f"Recall: {overall_metrics['recall']:.4f}")
        if "f1_score" in overall_metrics:
            self.logger.info(f"F1-Score: {overall_metrics['f1_score']:.4f}")
        
        self.logger.info("="*50)
    
    def _log_to_neptune(self) -> None:
        """Log evaluation results to Neptune."""
        try:
            # Log overall metrics
            overall_metrics = self.evaluation_results.get("overall_metrics", {})
            self.neptune_logger.log_metrics(overall_metrics)
            
            # Log per-class metrics
            per_class_metrics = self.evaluation_results.get("per_class_metrics", {})
            for class_name, metrics in per_class_metrics.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.neptune_logger.log_metrics({f"per_class/{class_name}/{metric_name}": value})
            
            # Log evaluation parameters
            parameters = self.evaluation_results.get("parameters", {})
            self.neptune_logger.log_hyperparameters({"evaluation_parameters": parameters})
            
            self.logger.info("Evaluation results logged to Neptune")
            
        except Exception as e:
            self.logger.warning(f"Failed to log to Neptune: {e}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.neptune_logger:
            self.neptune_logger.stop()
        self.logger.info("Evaluation cleanup completed")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLO Vehicle Detection Model Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="runs/train/vehicle_detection/weights/best.pt",
        help="Path to trained model"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="data.yaml",
        help="Path to dataset YAML file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.001,
        help="Confidence threshold for evaluation"
    )
    
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.6,
        help="IoU threshold for NMS"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for evaluation"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for evaluation (auto, cpu, 0, 1, etc.)"
    )
    
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save evaluation plots"
    )
    
    parser.add_argument(
        "--save_report",
        action="store_true",
        help="Save detailed evaluation report"
    )
    
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate YOLO validation plots"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main evaluation function."""
    args = parse_arguments()
    
    try:
        print_system_info()
        
        # Load configuration
        config = get_config()
        
        # Override config with command line arguments
        config.evaluation.confidence_threshold = args.conf_thresh
        config.evaluation.iou_threshold = args.iou_thresh
        config.evaluation.image_size = args.imgsz
        config.evaluation.batch_size = args.batch_size
        config.evaluation.plots = args.plots
        
        print(f"Evaluation configuration:")
        print(f"  Model: {args.model}")
        print(f"  Data: {args.data}")
        print(f"  Confidence threshold: {args.conf_thresh}")
        print(f"  IoU threshold: {args.iou_thresh}")
        print(f"  Image size: {args.imgsz}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Device: {args.device}")
        
        # Initialize evaluator
        evaluator = YOLOEvaluator(args.model, args.data, config, args.device)
        
        try:
            # Load model and validate dataset
            evaluator.load_model()
            
            if not evaluator.validate_dataset():
                print("Dataset validation failed. Exiting.")
                return
            
            # Run evaluation
            print("Starting evaluation...")
            results = evaluator.run_evaluation(
                confidence_threshold=args.conf_thresh,
                iou_threshold=args.iou_thresh,
                image_size=args.imgsz,
                batch_size=args.batch_size
            )
            
            # Generate outputs
            output_dir = Path(args.output)
            create_directories([output_dir])
            
            # Save results
            results_file = output_dir / "evaluation_results.json"
            save_json(results, results_file, indent=2)
            print(f"Results saved to: {results_file}")
            
            # Generate report
            if args.save_report:
                report_file = evaluator.generate_evaluation_report(output_dir)
                print(f"Detailed report saved to: {report_file}")
            
            # Generate plots
            if args.save_plots:
                print("Generating evaluation plots...")
                
                # Confusion matrix
                cm_file = evaluator.plot_confusion_matrix(output_dir)
                if cm_file:
                    print(f"Confusion matrix saved to: {cm_file}")
                
                # Per-class metrics
                pcm_file = evaluator.plot_per_class_metrics(output_dir)
                if pcm_file:
                    print(f"Per-class metrics plot saved to: {pcm_file}")
            
            # Print summary
            overall_metrics = results.get("overall_metrics", {})
            print("\n" + "="*50)
            print("EVALUATION COMPLETED")
            print("="*50)
            print(f"Evaluation time: {format_time(results.get('evaluation_time', 0))}")
            if "mAP@0.5" in overall_metrics:
                print(f"mAP@0.5: {overall_metrics['mAP@0.5']:.4f}")
            if "mAP@0.5:0.95" in overall_metrics:
                print(f"mAP@0.5:0.95: {overall_metrics['mAP@0.5:0.95']:.4f}")
            print("="*50)
            
        finally:
            evaluator.cleanup()
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 
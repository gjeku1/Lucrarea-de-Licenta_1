"""
"""
Modul de evaluare pentru proiectul de detecție a vehiculelor cu YOLO.

Acest modul oferă funcționalități complete de evaluare pentru modelele YOLO, 
incluzând calculul metricilor, vizualizări și capabilități de raportare detaliate.

Utilizare:
    # Evaluare de bază
    python evaluation.py --model best.pt --data data.yaml

    # Evaluare cu praguri personalizate
    python evaluation.py --model best.pt --data data.yaml --conf_thresh 0.5 --iou_thresh 0.6

    # Salvare rapoarte detaliate
    python evaluation.py --model best.pt --data data.yaml --save_plots --save_report
"""

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
    def __init__(
        self,
        model_path: str,
        data_yaml: str,
        config: Optional[Config] = None,
        device: Union[str, int] = 'auto'
    ):
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.config = config if config is not None else get_config()
        self.device = self._determine_device(device)
        self.logger = setup_logging(log_level="INFO")
        self.neptune_logger = None
        if hasattr(self.config.neptune, 'api_token') and self.config.neptune.api_token:
            self.neptune_logger = NeptuneLogger(self.config.neptune, "evaluation_run")
        self.model: Optional[YOLO] = None
        self.class_names: List[str] = self.config.data.class_names
        self.num_classes: int = self.config.data.num_classes
        self.evaluation_results: Dict[str, Any] = {}
        self.confusion_matrix: Optional[np.ndarray] = None
        self.logger.info(f"YOLO Evaluator initialized with device: {self.device}")
    
    def _determine_device(self, device: Union[str, int]) -> Union[str, int]:
        if device == 'auto':
            if torch.cuda.is_available():
                return 0
            else:
                return 'cpu'
        return device
    
    def load_model(self) -> None:
        try:
            self.logger.info(f"Loading model from: {self.model_path}")
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.logger.info("Model loaded successfully")
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
        try:
            if not Path(self.data_yaml).exists():
                self.logger.error(f"Data YAML file not found: {self.data_yaml}")
                return False
            import yaml
            with open(self.data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            required_keys = ['path', 'val', 'names']
            for key in required_keys:
                if key not in data_config:
                    self.logger.error(f"Missing required key in data.yaml: {key}")
                    return False
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
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        conf_thresh = confidence_threshold or self.config.evaluation.confidence_threshold
        iou_thresh = iou_threshold or self.config.evaluation.iou_threshold
        img_size = image_size or self.config.evaluation.image_size
        batch_size = batch_size or self.config.evaluation.batch_size
        self.logger.info("Starting comprehensive model evaluation...")
        start_time = time.time()
        try:
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
            metrics = self._extract_metrics(val_results)
            additional_metrics = self._calculate_additional_metrics(val_results)
            metrics.update(additional_metrics)
            confusion_matrix = self._generate_confusion_matrix(val_results)
            per_class_metrics = self._calculate_per_class_metrics(val_results)
            evaluation_time = time.time() - start_time
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
            if self.neptune_logger and self.neptune_logger.is_active:
                self._log_to_neptune()
            return self.evaluation_results
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise
    
    def _extract_metrics(self, val_results) -> Dict[str, float]:
        metrics = {}
        try:
            if hasattr(val_results, 'box'):
                box_results = val_results.box
                if hasattr(box_results, 'map'):
                    metrics['mAP@0.5'] = float(box_results.map50)
                    metrics['mAP@0.5:0.95'] = float(box_results.map)
                if hasattr(box_results, 'mp'):
                    metrics['precision'] = float(box_results.mp)
                if hasattr(box_results, 'mr'):
                    metrics['recall'] = float(box_results.mr)
                if 'precision' in metrics and 'recall' in metrics:
                    p, r = metrics['precision'], metrics['recall']
                    metrics['f1_score'] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                if hasattr(box_results, 'ap_class_index') and hasattr(box_results, 'ap'):
                    for i, class_idx in enumerate(box_results.ap_class_index):
                        class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Class_{class_idx}"
                        metrics[f'mAP@0.5_{class_name}'] = float(box_results.ap50[i])
                        metrics[f'mAP@0.5:0.95_{class_name}'] = float(box_results.ap[i])
        except Exception as e:
            self.logger.warning(f"Could not extract some metrics: {e}")
        return metrics
    
    def _calculate_additional_metrics(self, val_results) -> Dict[str, Any]:
        additional_metrics = {}
        try:
            if hasattr(val_results, 'speed'):
                speed = val_results.speed
                additional_metrics.update({
                    'preprocess_time_ms': speed.get('preprocess', 0),
                    'inference_time_ms': speed.get('inference', 0),
                    'postprocess_time_ms': speed.get('postprocess', 0),
                    'total_time_ms': sum(speed.values())
                })
            if self.model and hasattr(self.model.model, 'model'):
                try:
                    total_params = sum(p.numel() for p in self.model.model.parameters())
                    trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
                    additional_metrics.update({
                        'total_parameters': int(total_params),
                        'trainable_parameters': int(trainable_params),
                        'model_size_mb': total_params * 4 / (1024 * 1024)
                    })
                except Exception:
                    pass
        except Exception as e:
            self.logger.warning(f"Could not calculate additional metrics: {e}")
        return additional_metrics
    
    def _generate_confusion_matrix(self, val_results) -> Optional[np.ndarray]:
        try:
            if hasattr(val_results, 'confusion_matrix') and val_results.confusion_matrix is not None:
                self.confusion_matrix = val_results.confusion_matrix.matrix
                return self.confusion_matrix
            else:
                self.logger.warning("Confusion matrix not available from validation results")
                return None
        except Exception as e:
            self.logger.warning(f"Could not generate confusion matrix: {e}")
            return None
    
    def _calculate_per_class_metrics(self, val_results) -> Dict[str, Dict[str, float]]:
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
                    if hasattr(box_results, 'p') and i < len(box_results.p):
                        class_metrics['precision'] = float(box_results.p[i])
                    if hasattr(box_results, 'r') and i < len(box_results.r):
                        class_metrics['recall'] = float(box_results.r[i])
                    if 'precision' in class_metrics and 'recall' in class_metrics:
                        p, r = class_metrics['precision'], class_metrics['recall']
                        class_metrics['f1_score'] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                    per_class_metrics[class_name] = class_metrics
        except Exception as e:
            self.logger.warning(f"Could not calculate per-class metrics: {e}")
        return per_class_metrics
    def _log_summary(self) -> None:
        if not self.evaluation_results:
            return
        overall = self.evaluation_results.get("overall_metrics", {})
        self.logger.info("Evaluation Summary:")
        for key, value in overall.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def _log_to_neptune(self) -> None:
        if not (self.neptune_logger and self.neptune_logger.is_active):
            return
        try:
            self.neptune_logger.log_metrics(self.evaluation_results.get("overall_metrics", {}))
            self.neptune_logger.log_metrics(self.evaluation_results.get("per_class_metrics", {}))
            self.neptune_logger.log_metrics({"evaluation_time": self.evaluation_results.get("evaluation_time", 0)})
            if self.confusion_matrix is not None:
                self._plot_confusion_matrix(log_to_neptune=True)
            if self.config.evaluation.plots:
                try:
                    self._plot_per_class_metrics(log_to_neptune=True)
                except Exception as e:
                    self.logger.warning(f"Failed to plot per-class metrics: {e}")
        except Exception as e:
            self.logger.warning(f"Failed to log to Neptune: {e}")
    
    def save_results(self, output_path: str) -> None:
        try:
            save_json(self.evaluation_results, output_path)
            self.logger.info(f"Evaluation results saved to: {output_path}")
            if self.neptune_logger and self.neptune_logger.is_active:
                self.neptune_logger.log_artifact(output_path, destination="evaluation_results.json")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def _plot_confusion_matrix(self, log_to_neptune: bool = False) -> None:
        if self.confusion_matrix is None:
            return
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(self.confusion_matrix, annot=True, fmt=".2f", cmap='Blues',
                        xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            plt.tight_layout()
            plot_path = Path(self.config.evaluation.output_dir) / "confusion_matrix.png"
            plt.savefig(plot_path)
            plt.close(fig)
            self.logger.info(f"Confusion matrix plot saved to: {plot_path}")
            if log_to_neptune and self.neptune_logger and self.neptune_logger.is_active:
                self.neptune_logger.log_image(str(plot_path), "confusion_matrix")
        except Exception as e:
            self.logger.warning(f"Failed to plot confusion matrix: {e}")
    
    def _plot_per_class_metrics(self, log_to_neptune: bool = False) -> None:
        if not self.evaluation_results.get("per_class_metrics"):
            return
        try:
            per_class_metrics = self.evaluation_results["per_class_metrics"]
            metrics_to_plot = ["precision", "recall", "f1_score", "mAP@0.5", "mAP@0.5:0.95"]
            for metric in metrics_to_plot:
                values = [per_class_metrics[cls].get(metric, 0) for cls in self.class_names]
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(x=self.class_names, y=values, ax=ax)
                ax.set_ylabel(metric)
                ax.set_xlabel("Class")
                ax.set_title(f"Per-Class {metric}")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_path = Path(self.config.evaluation.output_dir) / f"{metric}_per_class.png"
                plt.savefig(plot_path)
                plt.close(fig)
                self.logger.info(f"Per-class {metric} plot saved to: {plot_path}")
                if log_to_neptune and self.neptune_logger and self.neptune_logger.is_active:
                    self.neptune_logger.log_image(str(plot_path), f"per_class_{metric}")
        except Exception as e:
            self.logger.warning(f"Failed to plot per-class metrics: {e}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO Model Evaluation Script")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model file")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--device", type=str, default='auto', help="Computation device")
    parser.add_argument("--conf-thresh", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=None, help="IoU threshold")
    parser.add_argument("--img-size", type=int, default=None, help="Image size")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file path")
    parser.add_argument("--plots", action="store_true", help="Generate evaluation plots")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = get_config()
    config.evaluation.plots = args.plots
    config.evaluation.verbose = args.verbose
    create_directories([config.evaluation.output_dir])
    print_system_info()
    evaluator = YOLOEvaluator(
        model_path=args.model,
        data_yaml=args.data,
        config=config,
        device=args.device
    )
    evaluator.load_model()
    if not evaluator.validate_dataset():
        raise SystemExit("Dataset validation failed.")
    results = evaluator.run_evaluation(
        confidence_threshold=args.conf_thresh,
        iou_threshold=args.iou_thresh,
        image_size=args.img_size,
        batch_size=args.batch_size
    )
    evaluator.save_results(args.output)
    if config.evaluation.plots:
        evaluator._plot_confusion_matrix()
        evaluator._plot_per_class_metrics()


if __name__ == "__main__":
    main()


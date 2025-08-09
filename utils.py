"""
Utility functions for YOLO vehicle detection project.
"""

import os
import json
import yaml
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import torch

try:
    import neptune as neptune_client
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False
    print("Neptune not available. Install with: pip install neptune")

from config import Config, NeptuneConfig


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, logs only to console
        log_format: Custom log format string
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console handler
        ]
    )
    
    logger = logging.getLogger("vehicle_detection")
    
    # Add file handler if log_file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


class NeptuneLogger:
    """Neptune logging wrapper for ML experiment tracking."""
    
    def __init__(self, config: NeptuneConfig, run_name: Optional[str] = None):
        """
        Initialize Neptune logger.
        
        Args:
            config: Neptune configuration object
            run_name: Custom run name. If None, uses timestamp
        """
        self.config = config
        self.run = None
        self.is_active = False
        
        if not NEPTUNE_AVAILABLE:
            print("Warning: Neptune not available. Logging will be skipped.")
            return
        
        try:
            # Generate run name if not provided
            if run_name is None:
                run_name = f"vehicle_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize Neptune run
            self.run = neptune_client.init_run(
                project=config.project_name,
                api_token=config.api_token,
                mode=config.mode,
                name=run_name,
                tags=["yolo", "vehicle_detection", "computer_vision"]
            )
            
            self.is_active = True
            print(f"Neptune run initialized: {self.run['sys/id'].fetch()}")
            
        except Exception as e:
            print(f"Failed to initialize Neptune: {e}")
            self.is_active = False
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to Neptune."""
        if not self.is_active:
            return
        
        try:
            self.run["hyperparameters"] = params
            print("Hyperparameters logged to Neptune")
        except Exception as e:
            print(f"Failed to log hyperparameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to Neptune."""
        if not self.is_active:
            return
        
        try:
            for key, value in metrics.items():
                if step is not None:
                    self.run[f"metrics/{key}"].append(value, step=step)
                else:
                    self.run[f"metrics/{key}"] = value
        except Exception as e:
            print(f"Failed to log metrics: {e}")
    
    def log_image(self, image: Union[str, np.ndarray], name: str, description: str = "") -> None:
        """Log image to Neptune."""
        if not self.is_active:
            return
        
        try:
            if isinstance(image, str):
                self.run[f"images/{name}"].upload(image)
            else:
                # Convert numpy array to image
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 8))
                plt.imshow(image)
                plt.title(description)
                plt.axis('off')
                self.run[f"images/{name}"].upload(plt.gcf())
                plt.close()
        except Exception as e:
            print(f"Failed to log image {name}: {e}")
    
    def log_model(self, model_path: str, name: str = "model") -> None:
        """Log model checkpoint to Neptune."""
        if not self.is_active:
            return
        
        try:
            self.run[f"models/{name}"].upload(model_path)
            print(f"Model {name} logged to Neptune")
        except Exception as e:
            print(f"Failed to log model: {e}")
    
    def log_file(self, file_path: str, name: Optional[str] = None) -> None:
        """Log file to Neptune."""
        if not self.is_active:
            return
        
        try:
            if name is None:
                name = Path(file_path).name
            self.run[f"files/{name}"].upload(file_path)
        except Exception as e:
            print(f"Failed to log file {file_path}: {e}")
    
    def stop(self) -> None:
        """Stop Neptune run."""
        if self.is_active and self.run:
            try:
                self.run.stop()
                print("Neptune run stopped")
            except Exception as e:
                print(f"Failed to stop Neptune run: {e}")


def create_directories(paths: List[Union[str, Path]]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dict containing YAML content
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save data to YAML file.
    
    Args:
        data: Data to save
        file_path: Output file path
    """
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dict containing JSON content
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation level
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def visualize_predictions(
    image: np.ndarray,
    predictions: List[Dict[str, Any]],
    class_names: List[str],
    colors: Optional[List[Tuple[int, int, int]]] = None,
    confidence_threshold: float = 0.5,
    thickness: int = 2
) -> np.ndarray:
    """
    Visualize predictions on image.
    
    Args:
        image: Input image as numpy array
        predictions: List of prediction dictionaries with keys: 'bbox', 'confidence', 'class_id'
        class_names: List of class names
        colors: List of colors for each class
        confidence_threshold: Minimum confidence to display
        thickness: Line thickness for bounding boxes
        
    Returns:
        np.ndarray: Image with visualized predictions
    """
    if colors is None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    # Create a copy of the image
    vis_image = image.copy()
    
    for pred in predictions:
        confidence = pred['confidence']
        if confidence < confidence_threshold:
            continue
        
        # Extract bounding box coordinates
        bbox = pred['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get class information
        class_id = pred['class_id']
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            vis_image, 
            (x1, y1 - label_size[1] - 10), 
            (x1 + label_size[0], y1), 
            color, 
            -1
        )
        cv2.putText(
            vis_image, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
    
    return vis_image


def calculate_metrics(
    predictions: List[Dict[str, Any]], 
    ground_truth: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate detection metrics (mAP, precision, recall).
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        iou_threshold: IoU threshold for matching predictions to ground truth
        confidence_threshold: Confidence threshold for filtering predictions
        
    Returns:
        Dict containing calculated metrics
    """
    # Filter predictions by confidence
    filtered_preds = [p for p in predictions if p['confidence'] >= confidence_threshold]
    
    # Calculate IoU for all prediction-GT pairs
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # Match predictions to ground truth
    tp = 0  # True positives
    fp = 0  # False positives
    fn = len(ground_truth)  # False negatives (initially all GT)
    
    matched_gt = set()
    
    for pred in filtered_preds:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            
            if pred['class_id'] == gt['class_id']:
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
            fn -= 1
        else:
            fp += 1
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def get_device() -> torch.device:
    """
    Get the best available device for training/inference.
    
    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def print_system_info() -> None:
    """Print system and environment information."""
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    
    # Python and PyTorch info
    import sys
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Device info
    device = get_device()
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    print("=" * 50)


def validate_paths(config: Config) -> bool:
    """
    Validate that all required paths exist.
    
    Args:
        config: Configuration object
        
    Returns:
        bool: True if all paths are valid, False otherwise
    """
    required_paths = [
        config.data.dataset_root,
        config.data.dataset_root / config.data.train_images,
        config.data.dataset_root / config.data.val_images,
    ]
    
    for path in required_paths:
        if not Path(path).exists():
            print(f"Error: Required path does not exist: {path}")
            return False
    
    return True 
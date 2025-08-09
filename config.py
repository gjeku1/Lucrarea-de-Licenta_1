"""
Configuration module for YOLO vehicle detection project.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import os


@dataclass
class DataConfig:
    """Configuration for dataset paths and parameters."""
    
    # Dataset paths
    dataset_root: Path = Path("dataset")
    train_images: str = "images/train"
    val_images: str = "images/valid" 
    test_images: str = "images/test"
    train_labels: str = "labels/train"
    val_labels: str = "labels/valid"
    test_labels: str = "labels/test"
    
    # Dataset metadata
    class_names: List[str] = None
    num_classes: int = 4
    
    def __post_init__(self):
        """Initialize default class names if not provided."""
        if self.class_names is None:
            self.class_names = ['Ambulance', 'car', 'BUS', 'Truck']
        
        # Ensure dataset root is Path object
        if isinstance(self.dataset_root, str):
            self.dataset_root = Path(self.dataset_root)


@dataclass
class TrainingConfig:
    """Configuration for model training parameters."""
    
    # Model configuration
    model_name: str = "yolov3u.pt"  # or "yolov3.yaml" for training from scratch
    model_size: str = "n"  # n, s, m, l, x
    
    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    image_size: int = 640
    device: Union[str, int] = 0  # 0 for first GPU, 'cpu' for CPU
    workers: int = 8
    
    # Training settings
    patience: int = 50  # Early stopping patience
    save_period: int = 10  # Save checkpoint every N epochs
    cache: bool = True  # Cache images for faster training
    
    # Data augmentation
    hsv_h: float = 0.015  # HSV-Hue augmentation
    hsv_s: float = 0.7    # HSV-Saturation augmentation
    hsv_v: float = 0.4    # HSV-Value augmentation
    degrees: float = 0.0  # Rotation degrees
    translate: float = 0.1  # Translation
    scale: float = 0.5    # Scaling
    shear: float = 0.0    # Shearing
    perspective: float = 0.0  # Perspective transformation
    flipud: float = 0.0   # Flip up-down probability
    fliplr: float = 0.5   # Flip left-right probability
    mosaic: float = 1.0   # Mosaic augmentation probability
    mixup: float = 0.0    # Mixup augmentation probability
    
    # Output configuration
    project_name: str = "runs/train"
    experiment_name: str = "vehicle_detection"
    save_json: bool = True
    save_hybrid: bool = False


@dataclass
class InferenceConfig:
    """Configuration for model inference parameters."""
    
    # Model configuration
    model_path: str = "runs/train/vehicle_detection/weights/best.pt"
    
    # Inference parameters
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 1000
    image_size: int = 640
    device: Union[str, int] = 0
    
    # Output configuration
    save_images: bool = True
    save_txt: bool = False
    save_conf: bool = True
    save_crop: bool = False
    show_labels: bool = True
    show_conf: bool = True
    line_thickness: int = 3
    
    # Visualization
    colors: Optional[List[tuple]] = None
    
    def __post_init__(self):
        """Initialize default colors if not provided."""
        if self.colors is None:
            self.colors = [
                (255, 0, 0),    # Red for Ambulance
                (0, 255, 0),    # Green for car
                (0, 0, 255),    # Blue for BUS
                (255, 255, 0),  # Yellow for Truck
            ]


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation parameters."""
    
    # Model and data paths
    model_path: str = "runs/train/vehicle_detection/weights/best.pt"
    data_yaml: str = "data.yaml"
    
    # Evaluation parameters
    image_size: int = 640
    device: Union[str, int] = 0
    batch_size: int = 32
    confidence_threshold: float = 0.001  # Lower for evaluation
    iou_threshold: float = 0.6  # Higher for evaluation
    
    # Output configuration
    save_json: bool = True
    save_hybrid: bool = False
    plots: bool = True
    verbose: bool = True


@dataclass
class NeptuneConfig:
    """Configuration for Neptune ML logging."""
    
    # Neptune project configuration
    project_name: str = "vehicle-detection/yolo"
    api_token: Optional[str] = None
    mode: str = "async"  # "async", "sync", "offline", "debug"
    
    # Logging configuration
    log_model_checkpoints: bool = True
    log_images: bool = True
    log_predictions: bool = True
    log_metrics: bool = True
    log_hyperparameters: bool = True
    
    # Upload configuration
    upload_source_files: List[str] = None
    
    def __post_init__(self):
        """Initialize default source files and API token."""
        if self.upload_source_files is None:
            self.upload_source_files = [
                "train.py",
                "inference.py", 
                "evaluation.py",
                "config.py",
                "utils.py",
                "data.yaml"
            ]
        
        # Get API token from environment if not provided
        if self.api_token is None:
            self.api_token = os.getenv("NEPTUNE_API_TOKEN")


@dataclass
class Config:
    """Main configuration class that combines all sub-configurations."""
    
    data: DataConfig = None
    training: TrainingConfig = None
    inference: InferenceConfig = None
    evaluation: EvaluationConfig = None
    neptune: NeptuneConfig = None
    
    def __post_init__(self):
        """Initialize all sub-configurations with defaults."""
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.inference is None:
            self.inference = InferenceConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.neptune is None:
            self.neptune = NeptuneConfig()
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for logging."""
        return {
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "inference": self.inference.__dict__,
            "evaluation": self.evaluation.__dict__,
            "neptune": self.neptune.__dict__,
        }
    
    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        import yaml
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML file."""
        import yaml
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            inference=InferenceConfig(**config_dict.get('inference', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            neptune=NeptuneConfig(**config_dict.get('neptune', {})),
        )


# Default configuration instance
default_config = Config()


def get_config() -> Config:
    """
    Get the default configuration instance.
    
    Returns:
        Config: The default configuration object
    """
    return default_config


def create_data_yaml(config: Config, output_path: str = "data.yaml") -> None:
    """
    Create YOLO data.yaml file from configuration.
    
    Args:
        config: Configuration object containing dataset information
        output_path: Path where to save the data.yaml file
    """
    import yaml
    
    data_yaml = {
        'path': str(config.data.dataset_root),
        'train': config.data.train_images,
        'val': config.data.val_images,
        'test': config.data.test_images,
        'names': config.data.class_names,
        'nc': config.data.num_classes
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False, default_flow_style=False) 
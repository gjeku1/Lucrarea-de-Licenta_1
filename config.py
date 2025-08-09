from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import os

@dataclass
class DataConfig:
    dataset_root: Path = Path("dataset")
    train_images: str = "images/train"
    val_images: str = "images/valid"
    test_images: str = "images/test"
    train_labels: str = "labels/train"
    val_labels: str = "labels/valid"
    test_labels: str = "labels/test"
    class_names: List[str] = None
    num_classes: int = 4

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ['Ambulance', 'car', 'BUS', 'Truck']
        if isinstance(self.dataset_root, str):
            self.dataset_root = Path(self.dataset_root)

@dataclass
class TrainingConfig:
    model_name: str = "yolov3u.pt"
    model_size: str = "n"
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    image_size: int = 640
    device: Union[str, int] = 0
    workers: int = 8
    patience: int = 50
    save_period: int = 10
    cache: bool = True
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    project_name: str = "runs/train"
    experiment_name: str = "vehicle_detection"
    save_json: bool = True
    save_hybrid: bool = False

@dataclass
class InferenceConfig:
    model_path: str = "runs/train/vehicle_detection/weights/best.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 1000
    image_size: int = 640
    device: Union[str, int] = 0
    save_images: bool = True
    save_txt: bool = False
    save_conf: bool = True
    save_crop: bool = False
    show_labels: bool = True
    show_conf: bool = True
    line_thickness: int = 3
    colors: Optional[List[tuple]] = None

    def __post_init__(self):
        if self.colors is None:
            self.colors = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
            ]

@dataclass
class EvaluationConfig:
    model_path: str = "runs/train/vehicle_detection/weights/best.pt"
    data_yaml: str = "data.yaml"
    image_size: int = 640
    device: Union[str, int] = 0
    batch_size: int = 32
    confidence_threshold: float = 0.001
    iou_threshold: float = 0.6
    save_json: bool = True
    save_hybrid: bool = False
    plots: bool = True
    verbose: bool = True

@dataclass
class NeptuneConfig:
    project_name: str = "vehicle-detection/yolo"
    api_token: Optional[str] = None
    mode: str = "async"
    log_model_checkpoints: bool = True
    log_images: bool = True
    log_predictions: bool = True
    log_metrics: bool = True
    log_hyperparameters: bool = True
    upload_source_files: List[str] = None

    def __post_init__(self):
        if self.upload_source_files is None:
            self.upload_source_files = [
                "train.py",
                "inference.py",
                "evaluation.py",
                "config.py",
                "utils.py",
                "data.yaml"
            ]
        if self.api_token is None:
            self.api_token = os.getenv("NEPTUNE_API_TOKEN")

@dataclass
class Config:
    data: DataConfig = None
    training: TrainingConfig = None
    inference: InferenceConfig = None
    evaluation: EvaluationConfig = None
    neptune: NeptuneConfig = None

    def __post_init__(self):
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
        return {
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "inference": self.inference.__dict__,
            "evaluation": self.evaluation.__dict__,
            "neptune": self.neptune.__dict__,
        }

    def save_yaml(self, path: Union[str, Path]) -> None:
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'Config':
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

default_config = Config()

def get_config() -> Config:
    return default_config

def create_data_yaml(config: Config, output_path: str = "data.yaml") -> None:
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

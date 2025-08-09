"""
modul de antrenare

Usage:
    python train.py --config config.yaml --epochs 100 --batch_size 16
"""

import argparse
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union
import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
import torch
import yaml

from config import Config, get_config, create_data_yaml
from utils import (
    setup_logging, 
    NeptuneLogger, 
    create_directories,
    print_system_info,
    validate_paths,
    format_time
)


class YOLOTrainer:
    """
Antrenor pentru modelul YOLO cu înregistrare și monitorizare completă.

Această clasă gestionează procesul intreg de antrenament, inclusiv:

Inițializarea și configurarea modelului

Pregătirea și validarea datelor

Executarea antrenamentului cu monitorizare

Salvarea punctelor de control și a modelului

Integrarea cu Neptune pentru înregistrarea datelor
    """
    
    def __init__(self, config: Config, run_name: Optional[str] = None):
        """
        Initialize the YOLO trainer.
        
        Args:
            config: Configuration object containing all training parameters
            run_name: Custom name for the training run
        """
        self.config = config
        
        # Create directories first (using basic logging)
        dirs_to_create = [
            self.config.training.project_name,
            f"{self.config.training.project_name}/{self.config.training.experiment_name}",
            f"{self.config.training.project_name}/{self.config.training.experiment_name}/weights",
            f"{self.config.training.project_name}/{self.config.training.experiment_name}/logs",
        ]
        create_directories(dirs_to_create)
        print(f"Created directories: {dirs_to_create}")
        
  
        self.logger = setup_logging(
            log_level="INFO",
            log_file=f"{config.training.project_name}/train.log"
        )
       
        self.neptune_logger = None
        if hasattr(config.neptune, 'api_token') and config.neptune.api_token:
            self.neptune_logger = NeptuneLogger(config.neptune, run_name)
        
        self.model: Optional[YOLO] = None
        self.start_time: Optional[float] = None
        
        self.logger.info("YOLO Trainer initialized")
    
    def setup_directories(self) -> None:
        """Create necessary directories for training outputs."""
        dirs_to_create = [
            self.config.training.project_name,
            f"{self.config.training.project_name}/{self.config.training.experiment_name}",
            f"{self.config.training.project_name}/{self.config.training.experiment_name}/weights",
            f"{self.config.training.project_name}/{self.config.training.experiment_name}/logs",
        ]
        
        create_directories(dirs_to_create)
        if hasattr(self, 'logger'):
            self.logger.info(f"Created directories: {dirs_to_create}")
        else:
            print(f"Created directories: {dirs_to_create}")
    
    def validate_environment(self) -> bool:
        """
        Validate the training environment and requirements.
        
        Returns:
            bool: True if environment is valid, False otherwise
        """
        self.logger.info("Validating training environment...")
        
      
        if isinstance(self.config.training.device, int) and self.config.training.device >= 0:
            if not torch.cuda.is_available():
                self.logger.error("CUDA not available but GPU device requested")
                return False
            
            if self.config.training.device >= torch.cuda.device_count():
                self.logger.error(f"GPU device {self.config.training.device} not available")
                return False
       
        if not validate_paths(self.config):
            self.logger.error("Dataset path validation failed")
            return False
        
      
        model_path = Path(self.config.training.model_name)
        if model_path.suffix == '.pt' and not model_path.exists():
            self.logger.warning(f"Pretrained model {model_path} not found. Will try to download.")
        
        self.logger.info("Environment validation completed successfully")
        return True
    
    def prepare_data(self) -> str:
        """
        Prepare dataset configuration for training.
        
        Returns:
            str: Path to the data configuration file
        """
        self.logger.info("Preparing dataset configuration...")
       
        data_yaml_path = "data.yaml"
        create_data_yaml(self.config, data_yaml_path)
        
        self.logger.info(f"Dataset configuration saved to {data_yaml_path}")
        self.logger.info(f"Classes: {self.config.data.class_names}")
        self.logger.info(f"Number of classes: {self.config.data.num_classes}")
        
        return data_yaml_path
    
    def initialize_model(self) -> None:
        """Initialize the YOLO model with specified configuration."""
        try:
            self.logger.info(f"Initializing YOLO model: {self.config.training.model_name}")
            
            # Load model (pretrained or from scratch)
            self.model = YOLO(self.config.training.model_name)
            
            self.logger.info("Model initialized successfully")
            
            if self.neptune_logger and self.neptune_logger.is_active:
                try:
                    model_info = {
                        "model_name": self.config.training.model_name,
                        "model_type": "YOLOv3" if "v3" in self.config.training.model_name else "YOLO",
                        "pretrained": self.config.training.model_name.endswith('.pt')
                    }
                    self.neptune_logger.log_hyperparameters({"model": model_info})
                except Exception as e:
                    self.logger.warning(f"Failed to log model info to Neptune: {e}")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise
    
    def log_hyperparameters(self) -> None:
        """Log training hyperparameters to Neptune."""
        if not (self.neptune_logger and self.neptune_logger.is_active):
            return
        
        try:
            
            hyperparameters = {
                "training": {
                    "epochs": self.config.training.epochs,
                    "batch_size": self.config.training.batch_size,
                    "learning_rate": self.config.training.learning_rate,
                    "image_size": self.config.training.image_size,
                    "device": str(self.config.training.device),
                    "patience": self.config.training.patience,
                    "workers": self.config.training.workers,
                },
                "augmentation": {
                    "hsv_h": self.config.training.hsv_h,
                    "hsv_s": self.config.training.hsv_s,
                    "hsv_v": self.config.training.hsv_v,
                    "degrees": self.config.training.degrees,
                    "translate": self.config.training.translate,
                    "scale": self.config.training.scale,
                    "flipud": self.config.training.flipud,
                    "fliplr": self.config.training.fliplr,
                    "mosaic": self.config.training.mosaic,
                    "mixup": self.config.training.mixup,
                },
                "dataset": {
                    "num_classes": self.config.data.num_classes,
                    "class_names": self.config.data.class_names,
                }
            }
            
            self.neptune_logger.log_hyperparameters(hyperparameters)
            self.logger.info("Hyperparameters logged to Neptune")
            
        except Exception as e:
            self.logger.warning(f"Failed to log hyperparameters: {e}")
    
    def train(self, data_yaml_path: str) -> Dict[str, Any]:
        """
        Execute the training process.
        
        Args:
            data_yaml_path: Path to the dataset configuration file
            
        Returns:
            Dict containing training results and metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        self.logger.info("Starting training process...")
        self.start_time = time.time()
        
        try:
            
            train_args = {
                'data': data_yaml_path,
                'epochs': self.config.training.epochs,
                'imgsz': self.config.training.image_size,
                'batch': self.config.training.batch_size,
                'lr0': self.config.training.learning_rate,
                'device': self.config.training.device,
                'workers': self.config.training.workers,
                'project': self.config.training.project_name,
                'name': self.config.training.experiment_name,
                'patience': self.config.training.patience,
                'save_period': self.config.training.save_period,
                'cache': self.config.training.cache,
                'save_json': self.config.training.save_json,
                'save_hybrid': self.config.training.save_hybrid,
                
                
                'hsv_h': self.config.training.hsv_h,
                'hsv_s': self.config.training.hsv_s,
                'hsv_v': self.config.training.hsv_v,
                'degrees': self.config.training.degrees,
                'translate': self.config.training.translate,
                'scale': self.config.training.scale,
                'shear': self.config.training.shear,
                'perspective': self.config.training.perspective,
                'flipud': self.config.training.flipud,
                'fliplr': self.config.training.fliplr,
                'mosaic': self.config.training.mosaic,
                'mixup': self.config.training.mixup,
            }
            
            self.logger.info(f"Training arguments: {train_args}")
            
            
            results = self.model.train(**train_args)
            
            
            training_time = time.time() - self.start_time
            formatted_time = format_time(training_time)
            
            self.logger.info(f"Training completed successfully in {formatted_time}")
            
            if self.neptune_logger and self.neptune_logger.is_active:
                try:
                    final_metrics = {
                        "training_time_seconds": training_time,
                        "training_time_formatted": formatted_time,
                        "final_epoch": self.config.training.epochs,
                    }
                    
                    
                    if hasattr(results, 'results_dict'):
                        final_metrics.update(results.results_dict)
                    
                    self.neptune_logger.log_metrics(final_metrics)
                    
                  
                    best_weights_path = f"{self.config.training.project_name}/{self.config.training.experiment_name}/weights/best.pt"
                    if Path(best_weights_path).exists():
                        self.neptune_logger.log_model(best_weights_path, "best_model")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to log final results to Neptune: {e}")
            
            return {
                "success": True,
                "training_time": training_time,
                "results": results,
                "best_weights": f"{self.config.training.project_name}/{self.config.training.experiment_name}/weights/best.pt",
                "last_weights": f"{self.config.training.project_name}/{self.config.training.experiment_name}/weights/last.pt"
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.logger.error(traceback.format_exc())
            
            if self.neptune_logger and self.neptune_logger.is_active:
                self.neptune_logger.log_metrics({"training_failed": True, "error": str(e)})
            
            raise
    
    def cleanup(self) -> None:
        """Clean up resources and stop logging."""
        if self.neptune_logger:
            self.neptune_logger.stop()
        
        self.logger.info("Training cleanup completed")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Train YOLO model for vehicle detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolov3u.pt",
        help="Model name or path to model file"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device for training (0, 1, 2, ... for GPU, 'cpu' for CPU)"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Project directory for saving results"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="vehicle_detection",
        help="Experiment name"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--neptune_project",
        type=str,
        default=None,
        help="Neptune project name for logging"
    )
    
    parser.add_argument(
        "--neptune_token",
        type=str,
        default=None,
        help="Neptune API token"
    )
    
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Custom name for the training run"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_arguments()
    
    try:
        
        print_system_info()
        
       
        if args.config:
            config = Config.from_yaml(args.config)
            print(f"Configuration loaded from {args.config}")
        else:
            config = get_config()
            print("Using default configuration")
        
        if args.model:
            config.training.model_name = args.model
        if args.epochs:
            config.training.epochs = args.epochs
        if args.batch_size:
            config.training.batch_size = args.batch_size
        if args.imgsz:
            config.training.image_size = args.imgsz
        if args.device:
            try:
                config.training.device = int(args.device)
            except ValueError:
                config.training.device = args.device
        if args.project:
            config.training.project_name = args.project
        if args.name:
            config.training.experiment_name = args.name
        if args.neptune_project:
            config.neptune.project_name = args.neptune_project
        if args.neptune_token:
            config.neptune.api_token = args.neptune_token
        
        print(f"Training configuration:")
        print(f"  Model: {config.training.model_name}")
        print(f"  Epochs: {config.training.epochs}")
        print(f"  Batch size: {config.training.batch_size}")
        print(f"  Image size: {config.training.image_size}")
        print(f"  Device: {config.training.device}")
        print(f"  Project: {config.training.project_name}")
        print(f"  Experiment: {config.training.experiment_name}")
        
          
        run_name = f"{config.training.experiment_name}_{int(time.time())}"
        trainer = YOLOTrainer(config, run_name)
        
        try:
          
            if not trainer.validate_environment():
                print("Environment validation failed. Exiting.")
                return
            
          
            data_yaml_path = trainer.prepare_data()
            trainer.initialize_model()
            trainer.log_hyperparameters()
            
            if args.resume:
                print(f"Resuming training from {args.resume}")
                # Load checkpoint logic would go here
            
          
            print("Starting training...")
            results = trainer.train(data_yaml_path)
            
          
            print("\n" + "="*50)
            print("TRAINING COMPLETED SUCCESSFULLY")
            print("="*50)
            print(f"Training time: {format_time(results['training_time'])}")
            print(f"Best weights: {results['best_weights']}")
            print(f"Last weights: {results['last_weights']}")
            print("="*50)
            
        finally:
            # Cleanup
            trainer.cleanup()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()


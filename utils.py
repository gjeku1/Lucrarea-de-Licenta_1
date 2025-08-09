"""
Funcții utilitare pentru proiectul de detectare vehicule YOLO.
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
    print("Neptune nu este disponibil. Instalează cu: pip install neptune")

from config import Config, NeptuneConfig


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configurează sistemul de înregistrare a jurnalului pentru proiect.
    
    Args:
        log_level: Nivelul de logare (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Calea către fișierul de log. Dacă None, loghează doar în consolă
        log_format: Format personalizat pentru mesajele de log
        
    Returns:
        logging.Logger: Instanța configurată a logger-ului
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configurează logging-ul
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Handler pentru consolă
        ]
    )
    
    logger = logging.getLogger("vehicle_detection")
    
    # Adaugă handler pentru fișier dacă log_file este specificat
    if log_file:
        # Creează directorul dacă nu există
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


class NeptuneLogger:
    """Wrapper pentru înregistrarea experimentelor în Neptune."""
    
    def __init__(self, config: NeptuneConfig, run_name: Optional[str] = None):
        """
        Inițializează logger-ul Neptune.
        
        Args:
            config: Obiect de configurare Neptune
            run_name: Nume personalizat pentru rulare. Dacă None, folosește timestamp
        """
        self.config = config
        self.run = None
        self.is_active = False
        
        if not NEPTUNE_AVAILABLE:
            print("Avertisment: Neptune nu este disponibil. Înregistrarea va fi omisă.")
            return
        
        try:
            # Generează nume pentru rulare dacă nu este specificat
            if run_name is None:
                run_name = f"vehicle_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Inițializează rularea Neptune
            self.run = neptune_client.init_run(
                project=config.project_name,
                api_token=config.api_token,
                mode=config.mode,
                name=run_name,
                tags=["yolo", "vehicle_detection", "computer_vision"]
            )
            
            self.is_active = True
            print(f"Rulare Neptune inițializată: {self.run['sys/id'].fetch()}")
            
        except Exception as e:
            print(f"Eșec la inițializarea Neptune: {e}")
            self.is_active = False
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Înregistrează hiperparametrii în Neptune."""
        if not self.is_active:
            return
        
        try:
            self.run["hyperparameters"] = params
            print("Hiperparametri înregistrați în Neptune")
        except Exception as e:
            print(f"Eșec la înregistrarea hiperparametrilor: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Înregistrează metrici în Neptune."""
        if not self.is_active:
            return
        
        try:
            for key, value in metrics.items():
                if step is not None:
                    self.run[f"metrics/{key}"].append(value, step=step)
                else:
                    self.run[f"metrics/{key}"] = value
        except Exception as e:
            print(f"Eșec la înregistrarea metricilor: {e}")
    
    def log_image(self, image: Union[str, np.ndarray], name: str, description: str = "") -> None:
        """Înregistrează imagine în Neptune."""
        if not self.is_active:
            return
        
        try:
            if isinstance(image, str):
                self.run[f"images/{name}"].upload(image)
            else:
                # Converteste numpy array în imagine
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 8))
                plt.imshow(image)
                plt.title(description)
                plt.axis('off')
                self.run[f"images/{name}"].upload(plt.gcf())
                plt.close()
        except Exception as e:
            print(f"Eșec la înregistrarea imaginii {name}: {e}")
    
    def log_model(self, model_path: str, name: str = "model") -> None:
        """Înregistrează model în Neptune."""
        if not self.is_active:
            return
        
        try:
            self.run[f"models/{name}"].upload(model_path)
            print(f"Model {name} înregistrat în Neptune")
        except Exception as e:
            print(f"Eșec la înregistrarea modelului: {e}")
    
    def log_file(self, file_path: str, name: Optional[str] = None) -> None:
        """Înregistrează fișier în Neptune."""
        if not self.is_active:
            return
        
        try:
            if name is None:
                name = Path(file_path).name
            self.run[f"files/{name}"].upload(file_path)
        except Exception as e:
            print(f"Eșec la înregistrarea fișierului {file_path}: {e}")
    
    def stop(self) -> None:
        """Oprește rularea Neptune."""
        if self.is_active and self.run:
            try:
                self.run.stop()
                print("Rulare Neptune oprită")
            except Exception as e:
                print(f"Eșec la oprirea rulării Neptune: {e}")


def create_directories(paths: List[Union[str, Path]]) -> None:
    """
    Creează directoarele dacă nu există.
    
    Args:
        paths: Listă de căi pentru directoarele de creat
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Încarcă fișier YAML.
    
    Args:
        file_path: Calea către fișierul YAML
        
    Returns:
        Dict conținând datele din YAML
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Salvează date în fișier YAML.
    
    Args:
        data: Datele de salvat
        file_path: Calea către fișierul de ieșire
    """
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Încarcă fișier JSON.
    
    Args:
        file_path: Calea către fișierul JSON
        
    Returns:
        Dict conținând datele din JSON
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Salvează date în fișier JSON.
    
    Args:
        data: Datele de salvat
        file_path: Calea către fișierul de ieșire
        indent: Nivelul de indentare JSON
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
    Vizualizează predicțiile pe imagine.
    
    Args:
        image: Imaginea de intrare ca numpy array
        predictions: Listă de dicționare cu predicții (chei: 'bbox', 'confidence', 'class_id')
        class_names: Lista de nume de clase
        colors: Lista de culori pentru fiecare clasă
        confidence_threshold: Pragma de încredere minimă pentru afișare
        thickness: Grosimea liniei pentru bounding boxes
        
    Returns:
        np.ndarray: Imaginea cu predicții vizualizate
    """
    if colors is None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    # Creează o copie a imaginii
    vis_image = image.copy()
    
    for pred in predictions:
        confidence = pred['confidence']
        if confidence < confidence_threshold:
            continue
        
        # Extrage coordonatele bounding box-ului
        bbox = pred['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Obține informațiile despre clasă
        class_id = pred['class_id']
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        color = colors[class_id % len(colors)]
        
        # Desenează bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
        
        # Desenează etichetă
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
    Calculează metrici de detectare (mAP, precizie, recall).
    
    Args:
        predictions: Lista de predicții
        ground_truth: Lista de adevăruri teren (ground truth)
        iou_threshold: Pragul IoU pentru potrivirea predicțiilor cu ground truth
        confidence_threshold: Pragul de încredere pentru filtrarea predicțiilor
        
    Returns:
        Dict conținând metricile calculate
    """
    # Filtrează predicțiile după încredere
    filtered_preds = [p for p in predictions if p['confidence'] >= confidence_threshold]
    
    # Calculează IoU pentru toate perechile predicție-ground truth
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculează IoU între două bounding box-uri."""
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
    
    # Potrivește predicțiile cu ground truth
    tp = 0  # True positives
    fp = 0  # False positives
    fn = len(ground_truth)  # False negatives (inițial toate GT)
    
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
    
    # Calculează metrici
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
    Returnează cel mai bun dispozitiv disponibil pentru antrenare/inferență.
    
    Returns:
        torch.device: Cel mai bun dispozitiv disponibil
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def format_time(seconds: float) -> str:
    """
    Formatează timpul în secunde într-un șir lizibil.
    
    Args:
        seconds: Timpul în secunde
        
    Returns:
        str: Șir de timp formatat
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def print_system_info() -> None:
    """Afișează informații despre sistem și mediu."""
    print("=" * 50)
    print("INFORMAȚII SISTEM")
    print("=" * 50)
    
    # Informații Python și PyTorch
    import sys
    print(f"Versiune Python: {sys.version}")
    print(f"Versiune PyTorch: {torch.__version__}")
    
    # Informații dispozitiv
    device = get_device()
    print(f"Dispozitiv: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA disponibil: {torch.cuda.is_available()}")
        print(f"Număr dispozitive CUDA: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"Nume dispozitiv CUDA: {torch.cuda.get_device_name(0)}")
    
    print("=" * 50)


def validate_paths(config: Config) -> bool:
    """
    Verifică dacă toate căile necesare există.
    
    Args:
        config: Obiect de configurare
        
    Returns:
        bool: True dacă toate căile sunt valide, False altfel
    """
    required_paths = [
        config.data.dataset_root,
        config.data.dataset_root / config.data.train_images,
        config.data.dataset_root / config.data.val_images,
    ]
    
    for path in required_paths:
        if not Path(path).exists():
            print(f"Eroare: Calea necesară nu există: {path}")
            return False
    
    return True 

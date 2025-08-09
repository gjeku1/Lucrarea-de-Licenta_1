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
            sel

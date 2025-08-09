"""
Inference module for YOLO vehicle detection project.

Utilizare:
    # Inferență pe o singură imagine
    python inference.py --source image.jpg --model best.pt

    # Inferență pe video
    python inference.py --source video.mp4 --model best.pt --save_video

    # Inferență pe mai multe imagini (batch)
    python inference.py --source images/ --model best.pt --save_results
"""


import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
import torch

from config import Config, get_config
from utils import (
    setup_logging,
    NeptuneLogger,
    create_directories,
    visualize_predictions,
    save_json,
    format_time,
    print_system_info
)


class YOLOInference:
    """
    YOLO model inference class with comprehensive functionality.
    
    This class handles various types of inference including:
    - Single image inference
    - Batch image processing
    - Video processing
    - Result visualization and export
    """
    
    def __init__(
        self, 
        model_path: str,
        config: Optional[Config] = None,
        device: Union[str, int] = 'auto'
    ):
        """
        Initialize the YOLO inference engine.
        
        Args:
            model_path: Path to the trained YOLO model
            config: Configuration object. If None, uses default config
            device: Device for inference ('auto', 'cpu', 0, 1, etc.)
        """
        self.model_path = model_path
        self.config = config if config is not None else get_config()
        self.device = self._determine_device(device)
        
        
        self.logger = setup_logging(log_level="INFO")
        
        
        self.neptune_logger = None
        if hasattr(self.config.neptune, 'api_token') and self.config.neptune.api_token:
            self.neptune_logger = NeptuneLogger(self.config.neptune, "inference_run")
        
        self.model: Optional[YOLO] = None
        self.class_names: List[str] = self.config.data.class_names
        self.colors: List[Tuple[int, int, int]] = self.config.inference.colors
        
        self.logger.info(f"YOLO Inference initialized with device: {self.device}")
    
    def _determine_device(self, device: Union[str, int]) -> Union[str, int]:
        """Determine the best device for inference."""
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
            
           
            if self.neptune_logger and self.neptune_logger.is_active:
                model_info = {
                    "model_path": self.model_path,
                    "device": str(self.device),
                    "class_names": self.class_names,
                    "num_classes": len(self.class_names)
                }
                self.neptune_logger.log_hyperparameters({"inference": model_info})
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_image(
        self,
        image_path: Union[str, Path],
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        image_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform inference on a single image.
        
        Args:
            image_path: Path to the input image
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            image_size: Input image size for inference
            
        Returns:
            Dict containing predictions and metadata
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Use config defaults if not specified
        conf_thresh = confidence_threshold or self.config.inference.confidence_threshold
        iou_thresh = iou_threshold or self.config.inference.iou_threshold
        img_size = image_size or self.config.inference.image_size
        
        try:
            start_time = time.time()
            
            
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
          
            results = self.model.predict(
                source=str(image_path),
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=img_size,
                device=self.device,
                verbose=False
            )
            
            inference_time = time.time() - start_time
            
        
            predictions = self._process_results(results[0])
            
            
            original_image = cv2.imread(str(image_path))
            if original_image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            result_dict = {
                "image_path": str(image_path),
                "predictions": predictions,
                "inference_time": inference_time,
                "num_detections": len(predictions),
                "image_shape": original_image.shape,
                "parameters": {
                    "confidence_threshold": conf_thresh,
                    "iou_threshold": iou_thresh,
                    "image_size": img_size
                }
            }
            
            self.logger.info(
                f"Inference completed for {image_path}: "
                f"{len(predictions)} detections in {inference_time:.3f}s"
            )
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Inference failed for {image_path}: {e}")
            raise
    
    def predict_batch(
        self,
        image_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_visualizations: bool = True,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform batch inference on multiple images.
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save results. If None, creates 'inference_results'
            save_visualizations: Whether to save visualized images
            save_results: Whether to save prediction results as JSON
            
        Returns:
            List of prediction dictionaries for all images
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        
        if output_dir is None:
            output_dir = Path("inference_results")
        else:
            output_dir = Path(output_dir)
        
        if save_visualizations or save_results:
            create_directories([output_dir])
            if save_visualizations:
                create_directories([output_dir / "visualizations"])
            if save_results:
                create_directories([output_dir / "predictions"])
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in image_dir.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")
        
        self.logger.info(f"Processing {len(image_files)} images from {image_dir}")
        
        all_results = []
        start_time = time.time()
        
        for i, image_path in enumerate(image_files):
            try:
               
                result = self.predict_image(image_path)
                all_results.append(result)
                
                
                if save_results:
                    result_file = output_dir / "predictions" / f"{image_path.stem}_predictions.json"
                    save_json(result, result_file)
                
              
                if save_visualizations:
                    vis_image = self.create_visualization(image_path, result["predictions"])
                    vis_file = output_dir / "visualizations" / f"{image_path.stem}_visualization.jpg"
                    cv2.imwrite(str(vis_file), vis_image)
                
                
                if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                    self.logger.info(f"Processed {i + 1}/{len(image_files)} images")
                    
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                continue
        
        total_time = time.time() - start_time
        avg_time_per_image = total_time / len(all_results) if all_results else 0
        
       
        if save_results:
            batch_summary = {
                "total_images": len(image_files),
                "successful_predictions": len(all_results),
                "total_time": total_time,
                "average_time_per_image": avg_time_per_image,
                "total_detections": sum(len(r["predictions"]) for r in all_results),
                "parameters": {
                    "confidence_threshold": self.config.inference.confidence_threshold,
                    "iou_threshold": self.config.inference.iou_threshold,
                    "image_size": self.config.inference.image_size
                }
            }
            
            summary_file = output_dir / "batch_summary.json"
            save_json(batch_summary, summary_file)
            
            if self.neptune_logger and self.neptune_logger.is_active:
                self.neptune_logger.log_metrics(batch_summary)
        
        self.logger.info(
            f"Batch inference completed: {len(all_results)}/{len(image_files)} images "
            f"processed in {format_time(total_time)}"
        )
        
        return all_results
    
    def predict_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save_video: bool = True,
        show_video: bool = False,
        skip_frames: int = 1
    ) -> Dict[str, Any]:
        """
        Perform inference on a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path for output video. If None, auto-generated
            save_video: Whether to save the annotated video
            show_video: Whether to display video during processing
            skip_frames: Process every Nth frame (1 = all frames)
            
        Returns:
            Dict containing video processing results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
    
        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_annotated{video_path.suffix}"
        else:
            output_path = Path(output_path)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
       
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        if save_video:
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        all_predictions = []
        frame_count = 0
        processed_frames = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
               
                if frame_count % skip_frames != 0:
                    if save_video:
                        out.write(frame)
                    continue
                
                frame_start = time.time()
                
                results = self.model.predict(
                    source=frame,
                    conf=self.config.inference.confidence_threshold,
                    iou=self.config.inference.iou_threshold,
                    imgsz=self.config.inference.image_size,
                    device=self.device,
                    verbose=False
                )
                
                predictions = self._process_results(results[0])
                frame_inference_time = time.time() - frame_start
                
                # Visualize predictions on frame
                annotated_frame = self._visualize_frame(frame, predictions)
                
                
                frame_result = {
                    "frame_number": frame_count,
                    "timestamp": frame_count / fps,
                    "predictions": predictions,
                    "inference_time": frame_inference_time,
                    "num_detections": len(predictions)
                }
                all_predictions.append(frame_result)
                
              
                if save_video and out is not None:
                    out.write(annotated_frame)
                
           
                if show_video:
                    cv2.imshow('YOLO Video Inference', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                processed_frames += 1
                
                # Log progress
                if processed_frames % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_current = processed_frames / elapsed
                    self.logger.info(f"Processed {processed_frames} frames, {fps_current:.1f} FPS")
        
        finally:
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        
     
        video_results = {
            "video_path": str(video_path),
            "output_path": str(output_path) if save_video else None,
            "video_info": {
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames,
                "processed_frames": processed_frames
            },
            "processing_stats": {
                "total_time": total_time,
                "average_fps": processed_frames / total_time if total_time > 0 else 0,
                "total_detections": sum(len(f["predictions"]) for f in all_predictions)
            },
            "frame_predictions": all_predictions
        }
        
        self.logger.info(
            f"Video processing completed: {processed_frames} frames in {format_time(total_time)}"
        )
        
        # Log to Neptune
        if self.neptune_logger and self.neptune_logger.is_active:
            self.neptune_logger.log_metrics(video_results["processing_stats"])
        
        return video_results
    
    def create_visualization(
        self,
        image_path: Union[str, Path],
        predictions: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Create visualization of predictions on image.
        
        Args:
            image_path: Path to the original image
            predictions: List of prediction dictionaries
            
        Returns:
            np.ndarray: Annotated image
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return visualize_predictions(
            image=image,
            predictions=predictions,
            class_names=self.class_names,
            colors=self.colors,
            confidence_threshold=self.config.inference.confidence_threshold,
            thickness=self.config.inference.line_thickness
        )
    
    def _visualize_frame(
        self,
        frame: np.ndarray,
        predictions: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Visualize predictions on a video frame."""
        return visualize_predictions(
            image=frame,
            predictions=predictions,
            class_names=self.class_names,
            colors=self.colors,
            confidence_threshold=self.config.inference.confidence_threshold,
            thickness=self.config.inference.line_thickness
        )
    
    def _process_results(self, result) -> List[Dict[str, Any]]:
        """
        Process YOLO results into standardized format.
        
        Args:
            result: YOLO result object
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                prediction = {
                    "bbox": boxes[i].tolist(),  # [x1, y1, x2, y2]
                    "confidence": float(scores[i]),
                    "class_id": int(classes[i]),
                    "class_name": self.class_names[int(classes[i])] if int(classes[i]) < len(self.class_names) else f"Class_{int(classes[i])}"
                }
                predictions.append(prediction)
        
        return predictions
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.neptune_logger:
            self.neptune_logger.stop()
        self.logger.info("Inference cleanup completed")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLO Vehicle Detection Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input source (image file, image directory, video file)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="runs/train/vehicle_detection/weights/best.pt",
        help="Path to trained model"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory or file path"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference (auto, cpu, 0, 1, etc.)"
    )
    
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save annotated video (for video input)"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results in real-time"
    )
    
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save prediction results as JSON"
    )
    
    parser.add_argument(
        "--save_viz",
        action="store_true",
        help="Save visualization images"
    )
    
    parser.add_argument(
        "--skip_frames",
        type=int,
        default=1,
        help="Process every Nth frame (video only)"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main inference function."""
    args = parse_arguments()
    source = r"C:\Users\User\Downloads\Cars Moving On Road Stock Footage - Free Download.mp4"
    model_path = "runs/train/vehicle_detection/weights/best.pt"
    try:
        print_system_info()
        
        
        config = get_config()
        
      
        config.inference.confidence_threshold = args.conf
        config.inference.iou_threshold = args.iou
        config.inference.image_size = args.imgsz
        
        print(f"Inference configuration:")
        print(f"  Model: {args.model}")
        print(f"  Source: {args.source}")
        print(f"  Confidence: {args.conf}")
        print(f"  IoU: {args.iou}")
        print(f"  Image size: {args.imgsz}")
        print(f"  Device: {args.device}")
        
  
        inference = YOLOInference(args.model, config, args.device)
        inference.load_model()
        
        try:
            source_path = Path(args.source)
            
            if source_path.is_file():
                # Check if it's an image or video
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
                
                if source_path.suffix.lower() in image_extensions:
                    # Single image inference
                    print(f"Processing single image: {args.source}")
                    result = inference.predict_image(args.source)
                    
                    if args.save_results:
                        output_dir = Path(args.output) if args.output else Path("inference_results")
                        create_directories([output_dir])
                        result_file = output_dir / f"{source_path.stem}_predictions.json"
                        save_json(result, result_file)
                        print(f"Results saved to: {result_file}")
                    
                    if args.save_viz:
                        output_dir = Path(args.output) if args.output else Path("inference_results")
                        create_directories([output_dir])
                        vis_image = inference.create_visualization(args.source, result["predictions"])
                        vis_file = output_dir / f"{source_path.stem}_visualization.jpg"
                        cv2.imwrite(str(vis_file), vis_image)
                        print(f"Visualization saved to: {vis_file}")
                    
                    print(f"Found {len(result['predictions'])} detections in {result['inference_time']:.3f}s")
                    
                elif source_path.suffix.lower() in video_extensions:
                    # Video inference
                    print(f"Processing video: {args.source}")
                    result = inference.predict_video(
                        video_path=args.source,
                        output_path=args.output,
                        save_video=args.save_video,
                        show_video=args.show,
                        skip_frames=args.skip_frames
                    )
                    
                    if args.save_results:
                        output_dir = Path(args.output).parent if args.output else Path("inference_results")
                        create_directories([output_dir])
                        result_file = output_dir / f"{source_path.stem}_video_results.json"
                        save_json(result, result_file)
                        print(f"Video results saved to: {result_file}")
                    
                    stats = result["processing_stats"]
                    print(f"Processed {result['video_info']['processed_frames']} frames")
                    print(f"Total detections: {stats['total_detections']}")
                    print(f"Average FPS: {stats['average_fps']:.1f}")
                
                else:
                    raise ValueError(f"Unsupported file format: {source_path.suffix}")
            
            elif source_path.is_dir():
                # Batch image inference
                print(f"Processing image directory: {args.source}")
                results = inference.predict_batch(
                    image_dir=args.source,
                    output_dir=args.output,
                    save_visualizations=args.save_viz,
                    save_results=args.save_results
                )
                
                total_detections = sum(len(r["predictions"]) for r in results)
                print(f"Processed {len(results)} images with {total_detections} total detections")
            
            else:
                raise ValueError(f"Invalid source: {args.source}")
        
        finally:
            inference.cleanup()
    
    except KeyboardInterrupt:
        print("\nInference interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nInference failed: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 

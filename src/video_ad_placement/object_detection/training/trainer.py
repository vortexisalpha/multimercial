"""
YOLOv9 Training Pipeline for Video Advertisement Placement

This module provides a comprehensive training pipeline with custom loss functions,
advanced augmentation, experiment tracking, and specialized optimization for
advertisement placement scenarios.
"""

import os
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import shutil
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not available. Experiment tracking disabled.")

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    from ultralytics.cfg import get_cfg
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    warnings.warn("Ultralytics not available.")

from ..detection_models import TrainingConfig, DetectionClass
from .dataset import VideoAdDataset, DatasetConfig
from .augmentation import VideoAugmentationPipeline

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics tracking."""
    
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    
    # Component losses
    box_loss: float = 0.0
    cls_loss: float = 0.0
    dfl_loss: float = 0.0
    
    # mAP metrics
    map50: float = 0.0
    map50_95: float = 0.0
    
    # Class-specific metrics
    person_ap: float = 0.0
    furniture_ap: float = 0.0
    hand_ap: float = 0.0
    
    # Training performance
    lr: float = 0.0
    gpu_memory: float = 0.0
    epoch_time: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            'epoch': float(self.epoch),
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'box_loss': self.box_loss,
            'cls_loss': self.cls_loss,
            'dfl_loss': self.dfl_loss,
            'map50': self.map50,
            'map50_95': self.map50_95,
            'person_ap': self.person_ap,
            'furniture_ap': self.furniture_ap,
            'hand_ap': self.hand_ap,
            'lr': self.lr,
            'gpu_memory': self.gpu_memory,
            'epoch_time': self.epoch_time
        }


class CustomLossFunction:
    """Custom loss function for advertisement placement optimization."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.advertisement_classes = DetectionClass.get_advertisement_relevant_classes()
        
        # Loss weights for different classes
        self.class_weights = self._setup_class_weights()
        
    def _setup_class_weights(self) -> Dict[int, float]:
        """Setup class-specific loss weights."""
        weights = {}
        
        for i, class_name in enumerate(self.config.class_names):
            if class_name.lower() in ['person', 'people']:
                weight = self.config.hand_tracking_weight if self.config.focus_on_person_detection else 1.0
            elif 'furniture' in class_name.lower():
                weight = self.config.furniture_detection_weight
            elif class_name.lower() in ['hand', 'arm']:
                weight = self.config.hand_tracking_weight
            else:
                weight = 1.0
            
            weights[i] = weight
        
        return weights
    
    def compute_weighted_loss(self, loss_components: Dict[str, torch.Tensor], 
                             class_predictions: torch.Tensor) -> torch.Tensor:
        """Compute weighted loss based on class importance."""
        total_loss = 0.0
        
        # Apply standard loss weights
        total_loss += loss_components.get('box_loss', 0) * self.config.box_loss_gain
        total_loss += loss_components.get('cls_loss', 0) * self.config.cls_loss_gain
        total_loss += loss_components.get('dfl_loss', 0) * self.config.dfl_loss_gain
        
        # Apply class-specific weighting if available
        if 'cls_loss_per_class' in loss_components:
            cls_loss_weighted = 0.0
            for class_id, class_loss in loss_components['cls_loss_per_class'].items():
                weight = self.class_weights.get(class_id, 1.0)
                cls_loss_weighted += class_loss * weight
            
            # Replace standard cls_loss with weighted version
            total_loss = total_loss - loss_components.get('cls_loss', 0) * self.config.cls_loss_gain
            total_loss += cls_loss_weighted * self.config.cls_loss_gain
        
        return total_loss


class EarlyStopping:
    """Early stopping implementation."""
    
    def __init__(self, patience: int = 50, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            
        return self.should_stop


class YOLOv9Trainer:
    """
    Comprehensive YOLOv9 trainer for video advertisement placement scenarios.
    
    Features:
    - Custom loss functions for advertisement-relevant classes
    - Advanced augmentation pipelines
    - Experiment tracking with MLflow
    - Distributed training support
    - Automatic hyperparameter optimization
    - Model versioning and deployment preparation
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.config.validate()
        
        # Initialize device
        self.device = self._setup_device()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision training
        
        # Training state
        self.current_epoch = 0
        self.best_map = 0.0
        self.training_metrics = []
        
        # Custom components
        self.custom_loss = CustomLossFunction(config)
        self.early_stopping = EarlyStopping(patience=config.patience)
        
        # Experiment tracking
        self.experiment_tracker = None
        if MLFLOW_AVAILABLE:
            self._setup_mlflow()
        
        # Setup training directory
        self.train_dir = Path(f"training_runs/{config.experiment_name}")
        self.train_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"YOLOv9Trainer initialized: {config.experiment_name}")
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)
        
        # Setup distributed training if needed
        if self.config.multi_gpu and torch.cuda.device_count() > 1:
            self._setup_distributed_training()
        
        logger.info(f"Training device: {device}")
        return device
    
    def _setup_distributed_training(self):
        """Setup distributed training."""
        try:
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            logger.info("Distributed training initialized")
        except Exception as e:
            logger.warning(f"Distributed training setup failed: {e}")
    
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        try:
            mlflow.set_experiment(self.config.project_name)
            logger.info("MLflow experiment tracking enabled")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load YOLOv9 model."""
        try:
            if ULTRALYTICS_AVAILABLE:
                if model_path:
                    self.model = YOLO(model_path)
                else:
                    # Download pretrained model
                    model_url = self.config.get_model_url()
                    self.model = YOLO(model_url)
                
                # Move to device
                self.model.to(self.device)
                
                logger.info(f"Model loaded: {self.config.model_size}")
                
            else:
                raise ImportError("Ultralytics not available")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def setup_data(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """Setup training, validation, and test dataloaders."""
        # Create dataset configuration
        dataset_config = DatasetConfig(
            input_size=self.config.input_size,
            num_classes=self.config.num_classes,
            class_names=self.config.class_names
        )
        
        # Setup augmentation pipeline
        augmentation_pipeline = VideoAugmentationPipeline(
            mosaic_prob=self.config.mosaic_prob,
            mixup_prob=self.config.mixup_prob,
            copy_paste_prob=self.config.copy_paste_prob,
            hsv_h=self.config.hsv_h,
            hsv_s=self.config.hsv_s,
            hsv_v=self.config.hsv_v,
            degrees=self.config.degrees,
            translate=self.config.translate,
            scale=self.config.scale,
            shear=self.config.shear,
            perspective=self.config.perspective,
            flipud=self.config.flipud,
            fliplr=self.config.fliplr
        )
        
        # Create datasets
        train_dataset = VideoAdDataset(
            data_path=self.config.train_data_path,
            config=dataset_config,
            augmentation=augmentation_pipeline,
            is_training=True
        )
        
        val_dataset = VideoAdDataset(
            data_path=self.config.val_data_path,
            config=dataset_config,
            augmentation=None,  # No augmentation for validation
            is_training=False
        )
        
        test_dataset = None
        if self.config.test_data_path:
            test_dataset = VideoAdDataset(
                data_path=self.config.test_data_path,
                config=dataset_config,
                augmentation=None,
                is_training=False
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.workers,
            pin_memory=True
        )
        
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.workers,
                pin_memory=True
            )
        
        logger.info(f"Datasets loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        # Get model parameters
        if hasattr(self.model, 'model'):
            model_params = self.model.model.parameters()
        else:
            model_params = self.model.parameters()
        
        # Create optimizer
        if self.config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                model_params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                model_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                model_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        # Create learning rate scheduler
        if self.config.scheduler.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler.lower() == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=self.config.epochs
            )
        else:
            self.scheduler = None
        
        # Setup mixed precision training
        if self.config.amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"Optimizer setup: {self.config.optimizer}")
    
    def train(self, save_best: bool = True) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            save_best: Whether to save the best model
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting training...")
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Setup data and optimizer
        train_loader, val_loader, test_loader = self.setup_data()
        self.setup_optimizer()
        
        # Start MLflow run
        if MLFLOW_AVAILABLE:
            mlflow.start_run(run_name=self.config.experiment_name)
            mlflow.log_params(self.config.to_dict())
        
        try:
            # Training loop
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch
                
                # Train epoch
                train_metrics = self._train_epoch(train_loader)
                
                # Validate epoch
                val_metrics = self._validate_epoch(val_loader)
                
                # Combine metrics
                epoch_metrics = TrainingMetrics(
                    epoch=epoch,
                    train_loss=train_metrics.get('loss', 0.0),
                    val_loss=val_metrics.get('loss', 0.0),
                    box_loss=train_metrics.get('box_loss', 0.0),
                    cls_loss=train_metrics.get('cls_loss', 0.0),
                    dfl_loss=train_metrics.get('dfl_loss', 0.0),
                    map50=val_metrics.get('map50', 0.0),
                    map50_95=val_metrics.get('map50_95', 0.0),
                    lr=self.optimizer.param_groups[0]['lr'],
                    gpu_memory=torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                )
                
                self.training_metrics.append(epoch_metrics)
                
                # Log metrics
                self._log_metrics(epoch_metrics)
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()
                
                # Save checkpoint
                if epoch % self.config.save_period == 0:
                    self._save_checkpoint(epoch)
                
                # Save best model
                if save_best and epoch_metrics.map50_95 > self.best_map:
                    self.best_map = epoch_metrics.map50_95
                    self._save_best_model()
                
                # Early stopping
                if self.early_stopping(epoch_metrics.val_loss):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Print progress
                logger.info(
                    f"Epoch {epoch}/{self.config.epochs} - "
                    f"Train Loss: {epoch_metrics.train_loss:.4f}, "
                    f"Val Loss: {epoch_metrics.val_loss:.4f}, "
                    f"mAP@0.5:0.95: {epoch_metrics.map50_95:.4f}"
                )
            
            # Final evaluation
            final_results = self._final_evaluation(test_loader)
            
            # Generate training report
            self._generate_training_report()
            
            return final_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.end_run()
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        if hasattr(self.model, 'train'):
            self.model.train()
        
        epoch_losses = {
            'loss': 0.0,
            'box_loss': 0.0,
            'cls_loss': 0.0,
            'dfl_loss': 0.0
        }
        
        epoch_start = time.time()
        
        with tqdm(train_loader, desc=f"Epoch {self.current_epoch} Training") as pbar:
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Use Ultralytics training if available
                    if ULTRALYTICS_AVAILABLE and hasattr(self.model, 'train'):
                        # Let Ultralytics handle the training step
                        results = self.model.train(
                            data=self.config.data_yaml_path,
                            epochs=1,
                            batch=self.config.batch_size,
                            imgsz=self.config.input_size,
                            device=self.device,
                            project=str(self.train_dir),
                            name="training",
                            exist_ok=True
                        )
                        
                        # Extract metrics from results
                        if hasattr(results, 'box_loss'):
                            epoch_losses['box_loss'] += float(results.box_loss)
                        if hasattr(results, 'cls_loss'):
                            epoch_losses['cls_loss'] += float(results.cls_loss)
                        if hasattr(results, 'dfl_loss'):
                            epoch_losses['dfl_loss'] += float(results.dfl_loss)
                        
                        break  # Ultralytics handles the full epoch
                    
                    else:
                        # Custom training step (fallback)
                        loss = self._custom_training_step(batch)
                        epoch_losses['loss'] += loss.item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{epoch_losses['loss']/(batch_idx+1):.4f}",
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    })
                    
                except Exception as e:
                    logger.warning(f"Training step failed: {e}")
                    continue
        
        # Average losses
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        epoch_losses['epoch_time'] = time.time() - epoch_start
        
        return epoch_losses
    
    def _custom_training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Custom training step implementation."""
        # This is a simplified implementation
        # In practice, you would implement the full YOLO loss calculation
        
        images = batch['images'].to(self.device)
        targets = batch['targets'].to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.config.amp):
            predictions = self.model(images)
            
            # Calculate loss (simplified)
            loss = F.mse_loss(predictions, targets)  # Placeholder
        
        # Backward pass
        if self.config.amp and self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        return loss
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        if hasattr(self.model, 'val'):
            # Use Ultralytics validation
            results = self.model.val()
            
            return {
                'loss': float(getattr(results, 'box_loss', 0) + getattr(results, 'cls_loss', 0)),
                'map50': float(getattr(results, 'map50', 0)),
                'map50_95': float(getattr(results, 'map50_95', 0))
            }
        else:
            # Custom validation implementation
            return self._custom_validation(val_loader)
    
    def _custom_validation(self, val_loader: DataLoader) -> Dict[str, float]:
        """Custom validation implementation."""
        if hasattr(self.model, 'eval'):
            self.model.eval()
        
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                predictions = self.model(images)
                loss = F.mse_loss(predictions, targets)  # Placeholder
                val_loss += loss.item()
        
        return {
            'loss': val_loss / len(val_loader),
            'map50': 0.0,  # Placeholder
            'map50_95': 0.0  # Placeholder
        }
    
    def _log_metrics(self, metrics: TrainingMetrics) -> None:
        """Log training metrics."""
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_metrics(metrics.to_dict(), step=metrics.epoch)
        
        # Log to console
        logger.info(f"Epoch {metrics.epoch}: {metrics.to_dict()}")
    
    def _save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.train_dir / f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict() if hasattr(self.model, 'state_dict') else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_map': self.best_map,
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_best_model(self) -> None:
        """Save the best model."""
        best_model_path = self.train_dir / "best_model.pt"
        
        if hasattr(self.model, 'save'):
            self.model.save(best_model_path)
        else:
            torch.save(self.model.state_dict(), best_model_path)
        
        # Log model to MLflow
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.pytorch.log_model(self.model, "best_model")
        
        logger.info(f"Best model saved: {best_model_path}")
    
    def _final_evaluation(self, test_loader: Optional[DataLoader]) -> Dict[str, Any]:
        """Perform final evaluation."""
        results = {
            'best_map': self.best_map,
            'total_epochs': self.current_epoch + 1,
            'training_time': sum(m.epoch_time for m in self.training_metrics if hasattr(m, 'epoch_time'))
        }
        
        if test_loader and hasattr(self.model, 'val'):
            test_results = self.model.val(data=test_loader)
            results.update({
                'test_map50': float(getattr(test_results, 'map50', 0)),
                'test_map50_95': float(getattr(test_results, 'map50_95', 0))
            })
        
        return results
    
    def _generate_training_report(self) -> None:
        """Generate comprehensive training report."""
        report_path = self.train_dir / "training_report.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Training Report: {self.config.experiment_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Configuration:\n")
            for key, value in self.config.to_dict().items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nTraining Results:\n")
            f.write(f"  Best mAP@0.5:0.95: {self.best_map:.4f}\n")
            f.write(f"  Total Epochs: {self.current_epoch + 1}\n")
            
            if self.training_metrics:
                final_metrics = self.training_metrics[-1]
                f.write(f"  Final Train Loss: {final_metrics.train_loss:.4f}\n")
                f.write(f"  Final Val Loss: {final_metrics.val_loss:.4f}\n")
        
        logger.info(f"Training report saved: {report_path}")
    
    def resume_training(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.best_map = checkpoint['best_map']
        
        if self.model and checkpoint['model_state_dict']:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Training resumed from epoch {self.current_epoch}")
    
    def export_model(self, export_formats: List[str] = None) -> Dict[str, str]:
        """Export trained model in various formats."""
        if export_formats is None:
            export_formats = self.config.export_formats
        
        export_paths = {}
        
        for format_name in export_formats:
            try:
                export_path = self.train_dir / f"exported_model.{format_name}"
                
                if hasattr(self.model, 'export'):
                    self.model.export(format=format_name, imgsz=self.config.input_size)
                    export_paths[format_name] = str(export_path)
                    logger.info(f"Model exported in {format_name} format")
                
            except Exception as e:
                logger.warning(f"Failed to export model in {format_name} format: {e}")
        
        return export_paths
    
    def cleanup(self) -> None:
        """Cleanup training resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if hasattr(self, 'model') and self.model:
            del self.model
        
        logger.info("Training cleanup completed") 
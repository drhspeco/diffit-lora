"""
Training Callbacks for DiffiT

PyTorch Lightning callbacks for monitoring and checkpointing.
"""

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Dict, Any, List
import os


def setup_callbacks(config: Dict[str, Any]) -> List:
    """
    Setup PyTorch Lightning callbacks based on configuration
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        List of configured callbacks
    """
    training_config = config.get('training', {})
    
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=training_config.get('output_dir', './checkpoints/'),
        filename="{epoch:02d}-{" + training_config.get('monitor_metric', 'train_loss') + ":.2f}",
        monitor=training_config.get('monitor_metric', 'train_loss'),
        mode="min",
        save_top_k=training_config.get('save_top_k', 3),
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping (optional)
    if training_config.get('early_stopping', {}).get('enabled', False):
        early_stopping = EarlyStopping(
            monitor=training_config.get('monitor_metric', 'train_loss'),
            patience=training_config.get('early_stopping', {}).get('patience', 10),
            mode="min",
            verbose=True
        )
        callbacks.append(early_stopping)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    return callbacks


def setup_logger(config: Dict[str, Any]) -> TensorBoardLogger:
    """
    Setup PyTorch Lightning logger
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Configured TensorBoard logger
    """
    training_config = config.get('training', {})
    
    log_dir = os.path.join(training_config.get('output_dir', './logs/'), 'logs')
    experiment_name = training_config.get('experiment_name', 'diffit_experiment')
    
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=experiment_name,
        version=None
    )
    
    return logger

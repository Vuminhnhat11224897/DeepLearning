"""
Logging configuration using loguru
"""
import os
import sys
from loguru import logger
from datetime import datetime

def setup_logger(log_dir: str = None, log_name: str = None, level: str = "INFO"):
    """
    Setup logger with file and console output
    
    Args:
        log_dir: Directory to save log files
        log_name: Name of the log file (without extension)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()
    
    # Console format with colors
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # File format without colors
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    
    # Add console handler
    logger.add(
        sys.stdout,
        format=console_format,
        level=level,
        colorize=True
    )
    
    # Add file handler if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
        if log_name is None:
            log_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_file = os.path.join(log_dir, f"{log_name}.log")
        
        logger.add(
            log_file,
            format=file_format,
            level=level,
            rotation="100 MB",  # Rotate when file reaches 100 MB
            retention="7 days",  # Keep logs for 7 days
            compression="zip",  # Compress rotated files
            enqueue=True  # Thread-safe logging
        )
        
        logger.info(f"Log file: {log_file}")
    
    return logger


def get_training_logger(log_dir: str):
    """Get logger specifically for training"""
    return setup_logger(
        log_dir=log_dir,
        log_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        level="INFO"
    )


def get_optuna_logger(log_dir: str):
    """Get logger specifically for Optuna optimization"""
    return setup_logger(
        log_dir=log_dir,
        log_name=f"optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        level="INFO"
    )


class TrainingLogger:
    """Class to handle training logs with structured output"""
    
    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir
        self.logger = setup_logger(log_dir, "training", "INFO") if log_dir else logger
        self.epoch_logs = []
        
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  lr: float, epoch_time: float, extra: dict = None):
        """Log epoch results"""
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': lr,
            'epoch_time': epoch_time
        }
        if extra:
            log_entry.update(extra)
        
        self.epoch_logs.append(log_entry)
        
        self.logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
    
    def log_best_model(self, epoch: int, val_loss: float, path: str):
        """Log when best model is saved"""
        self.logger.success(f"New best model at epoch {epoch} with val_loss={val_loss:.6f}")
        self.logger.info(f"Model saved to: {path}")
    
    def log_early_stopping(self, epoch: int, patience: int):
        """Log early stopping"""
        self.logger.warning(f"Early stopping triggered at epoch {epoch} (patience={patience})")
    
    def log_training_complete(self, total_time: float, best_epoch: int, best_loss: float):
        """Log training completion"""
        self.logger.success(
            f"Training completed in {total_time/60:.1f} minutes | "
            f"Best epoch: {best_epoch} | "
            f"Best val_loss: {best_loss:.6f}"
        )
    
    def log_hyperparams(self, params: dict):
        """Log hyperparameters"""
        self.logger.info("=" * 60)
        self.logger.info("Hyperparameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 60)


class OptunaLogger:
    """Class to handle Optuna optimization logs"""
    
    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir
        self.logger = setup_logger(log_dir, "optuna", "INFO") if log_dir else logger
        
    def log_trial_start(self, trial_number: int, params: dict):
        """Log trial start"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Trial {trial_number} started")
        self.logger.info("Parameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_trial_complete(self, trial_number: int, value: float, best_value: float):
        """Log trial completion"""
        is_best = value <= best_value
        if is_best:
            self.logger.success(f"Trial {trial_number} completed with value: {value:.6f} (NEW BEST)")
        else:
            self.logger.info(f"Trial {trial_number} completed with value: {value:.6f}")
    
    def log_trial_pruned(self, trial_number: int, step: int):
        """Log pruned trial"""
        self.logger.warning(f"Trial {trial_number} pruned at step {step}")
    
    def log_optimization_complete(self, best_trial, n_trials: int, duration: float):
        """Log optimization completion"""
        self.logger.success(f"\n{'='*60}")
        self.logger.success("Optimization completed!")
        self.logger.success(f"Total trials: {n_trials}")
        self.logger.success(f"Duration: {duration/60:.1f} minutes")
        self.logger.success(f"Best trial: {best_trial.number}")
        self.logger.success(f"Best value: {best_trial.value:.6f}")
        self.logger.success("Best parameters:")
        for key, value in best_trial.params.items():
            self.logger.success(f"  {key}: {value}")

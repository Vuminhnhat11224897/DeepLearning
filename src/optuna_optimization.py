"""
Optuna Hyperparameter Optimization
"""
import os
import time
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from typing import Dict, Optional, Callable

from .model import Encoder, Decoder, Seq2Seq
from .train import EarlyStopping, train_one_epoch, validate
from .logger import OptunaLogger


def create_model_from_params(trial_params: dict, 
                              input_size: int,
                              output_seq_len: int,
                              device: torch.device) -> nn.Module:
    """Create model from trial parameters"""
    hidden_size = trial_params['hidden_size']
    num_layers = trial_params['num_layers']
    dropout = trial_params['dropout']
    bidirectional = trial_params.get('bidirectional', True)
    
    encoder = Encoder(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout if num_layers > 1 else 0,
        bidirectional=bidirectional
    )
    
    decoder_input_size = hidden_size * 2 if bidirectional else hidden_size
    decoder = Decoder(
        input_size=1,
        hidden_size=decoder_input_size,
        output_size=1,
        num_layers=num_layers,
        dropout=dropout if num_layers > 1 else 0
    )
    
    model = Seq2Seq(encoder, decoder, output_seq_len, device)
    return model.to(device)


class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimizer for Seq2Seq model
    """
    
    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 output_seq_len: int,
                 search_space: Dict,
                 device: torch.device = None,
                 log_dir: str = None,
                 study_name: str = "seq2seq_optimization",
                 storage: str = None):
        """
        Initialize optimizer
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            output_seq_len: Number of output timesteps
            search_space: Dictionary defining hyperparameter search space
            device: torch device
            log_dir: Directory for logging
            study_name: Name of Optuna study
            storage: SQLite database path for persistent storage
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_size = X_train.shape[2]
        self.output_seq_len = output_seq_len
        self.search_space = search_space
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_dir = log_dir
        self.study_name = study_name
        self.storage = storage
        
        # Setup logger
        self.optuna_logger = OptunaLogger(log_dir)
        
        # Best results
        self.best_params = None
        self.best_value = float('inf')
        
    def _create_dataloaders(self, batch_size: int):
        """Create train and validation dataloaders"""
        from .dataset import TimeSeriesDataset
        
        train_dataset = TimeSeriesDataset(self.X_train, self.y_train)
        val_dataset = TimeSeriesDataset(self.X_val, self.y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, val_loader
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation loss (to be minimized)
        """
        # Sample hyperparameters
        params = {}
        
        # Hidden size
        if 'hidden_size' in self.search_space:
            if isinstance(self.search_space['hidden_size'], list):
                params['hidden_size'] = trial.suggest_categorical('hidden_size', self.search_space['hidden_size'])
            else:
                params['hidden_size'] = trial.suggest_int('hidden_size', *self.search_space['hidden_size'])
        else:
            params['hidden_size'] = 128
            
        # Num layers
        if 'num_layers' in self.search_space:
            if isinstance(self.search_space['num_layers'], list):
                params['num_layers'] = trial.suggest_categorical('num_layers', self.search_space['num_layers'])
            else:
                params['num_layers'] = trial.suggest_int('num_layers', *self.search_space['num_layers'])
        else:
            params['num_layers'] = 2
            
        # Dropout
        if 'dropout' in self.search_space:
            params['dropout'] = trial.suggest_float('dropout', *self.search_space['dropout'])
        else:
            params['dropout'] = 0.2
            
        # Learning rate
        if 'learning_rate' in self.search_space:
            params['learning_rate'] = trial.suggest_float('learning_rate', *self.search_space['learning_rate'], log=True)
        else:
            params['learning_rate'] = 0.001
            
        # Batch size
        if 'batch_size' in self.search_space:
            if isinstance(self.search_space['batch_size'], list):
                params['batch_size'] = trial.suggest_categorical('batch_size', self.search_space['batch_size'])
            else:
                params['batch_size'] = trial.suggest_int('batch_size', *self.search_space['batch_size'])
        else:
            params['batch_size'] = 64
            
        # Weight decay
        if 'weight_decay' in self.search_space:
            params['weight_decay'] = trial.suggest_float('weight_decay', *self.search_space['weight_decay'], log=True)
        else:
            params['weight_decay'] = 1e-5
            
        # Teacher forcing ratio
        if 'teacher_forcing_ratio' in self.search_space:
            params['teacher_forcing_ratio'] = trial.suggest_float('teacher_forcing_ratio', *self.search_space['teacher_forcing_ratio'])
        else:
            params['teacher_forcing_ratio'] = 0.5
        
        # Log trial start
        self.optuna_logger.log_trial_start(trial.number, params)
        
        try:
            # Create data loaders
            train_loader, val_loader = self._create_dataloaders(params['batch_size'])
            
            # Create model
            model = create_model_from_params(
                params, 
                self.input_size, 
                self.output_seq_len, 
                self.device
            )
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = AdamW(
                model.parameters(), 
                lr=params['learning_rate'], 
                weight_decay=params['weight_decay']
            )
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7)
            
            # Early stopping
            early_stopping = EarlyStopping(patience=10, mode='min')
            
            # Training loop
            n_epochs = 50  # Reduced epochs for faster optimization
            best_val_loss = float('inf')
            
            for epoch in range(n_epochs):
                # Decay teacher forcing
                current_tf = params['teacher_forcing_ratio'] * (1 - epoch / n_epochs)
                
                # Train and validate
                train_loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, 
                    self.device, current_tf, gradient_clip=1.0
                )
                val_loss = validate(model, val_loader, criterion, self.device)
                
                # Update scheduler
                scheduler.step(val_loss)
                
                # Track best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                # Report intermediate value for pruning
                trial.report(val_loss, epoch)
                
                # Handle pruning
                if trial.should_prune():
                    self.optuna_logger.log_trial_pruned(trial.number, epoch)
                    raise optuna.TrialPruned()
                
                # Early stopping
                if early_stopping(val_loss):
                    break
            
            # Log trial completion
            self.optuna_logger.log_trial_complete(trial.number, best_val_loss, self.best_value)
            
            # Update best
            if best_val_loss < self.best_value:
                self.best_value = best_val_loss
                self.best_params = params.copy()
            
            # Clean up
            del model, optimizer, train_loader, val_loader
            torch.cuda.empty_cache()
            
            return best_val_loss
            
        except Exception as e:
            self.optuna_logger.logger.error(f"Trial {trial.number} failed: {str(e)}")
            raise optuna.TrialPruned()
    
    def optimize(self, 
                 n_trials: int = 50, 
                 timeout: int = None,
                 show_progress_bar: bool = True) -> Dict:
        """
        Run hyperparameter optimization
        
        Args:
            n_trials: Number of trials
            timeout: Maximum time in seconds
            show_progress_bar: Show progress bar
            
        Returns:
            Dictionary with best parameters and study results
        """
        self.optuna_logger.logger.info(f"Starting Optuna optimization with {n_trials} trials")
        self.optuna_logger.logger.info(f"Device: {self.device}")
        self.optuna_logger.logger.info(f"Input size: {self.input_size}")
        self.optuna_logger.logger.info(f"Output sequence length: {self.output_seq_len}")
        
        start_time = time.time()
        
        # Create study with pruner
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        if self.storage:
            os.makedirs(os.path.dirname(self.storage), exist_ok=True)
            storage_url = f"sqlite:///{self.storage}"
        else:
            storage_url = None
        
        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage_url,
            load_if_exists=True,
            direction="minimize",
            pruner=pruner
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar,
            gc_after_trial=True
        )
        
        duration = time.time() - start_time
        
        # Get results
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
        
        # Log completion
        self.optuna_logger.log_optimization_complete(study.best_trial, len(completed_trials), duration)
        
        results = {
            'best_params': study.best_trial.params,
            'best_value': study.best_trial.value,
            'best_trial_number': study.best_trial.number,
            'n_completed_trials': len(completed_trials),
            'n_pruned_trials': len(pruned_trials),
            'duration_minutes': duration / 60,
            'study': study
        }
        
        return results
    
    def get_best_params(self) -> Dict:
        """Get best hyperparameters found"""
        return self.best_params
    

def run_optuna_optimization(X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: np.ndarray,
                            y_val: np.ndarray,
                            output_seq_len: int,
                            search_space: Dict,
                            n_trials: int = 50,
                            timeout: int = None,
                            device: torch.device = None,
                            log_dir: str = None,
                            db_path: str = None) -> Dict:
    """
    Convenience function to run Optuna optimization
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        output_seq_len: Number of output timesteps
        search_space: Hyperparameter search space
        n_trials: Number of trials
        timeout: Maximum time in seconds
        device: torch device
        log_dir: Directory for logs
        db_path: Path to SQLite database for study storage
        
    Returns:
        Dictionary with optimization results
    """
    optimizer = OptunaOptimizer(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        output_seq_len=output_seq_len,
        search_space=search_space,
        device=device,
        log_dir=log_dir,
        storage=db_path
    )
    
    results = optimizer.optimize(n_trials=n_trials, timeout=timeout)
    
    return results

"""
Optuna Hyperparameter Optimization for Seq2Seq Model
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.trial import TrialState
from datetime import datetime

from .config import DEVICE, RANDOM_SEED, set_seed
from .model import build_model


class Seq2SeqObjective:
    """Objective function for Optuna optimization"""
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        output_seq_len: int,
        search_space: dict,
        device: str = 'cuda',
        n_epochs: int = 30,
        patience: int = 5
    ):
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.FloatTensor(y_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_val = torch.FloatTensor(y_val)
        self.input_size = X_train.shape[2]
        self.output_seq_len = output_seq_len
        self.search_space = search_space
        self.device = device
        self.n_epochs = n_epochs
        self.patience = patience
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Single trial objective function"""
        
        # Sample hyperparameters
        hidden_size = trial.suggest_categorical('hidden_size', self.search_space['hidden_size'])
        num_layers = trial.suggest_categorical('num_layers', self.search_space['num_layers'])
        dropout = trial.suggest_float('dropout', *self.search_space['dropout'])
        lr = trial.suggest_float('learning_rate', *self.search_space['learning_rate'], log=True)
        batch_size = trial.suggest_categorical('batch_size', self.search_space['batch_size'])
        weight_decay = trial.suggest_float('weight_decay', *self.search_space['weight_decay'], log=True)
        teacher_forcing = trial.suggest_float('teacher_forcing_ratio', *self.search_space['teacher_forcing_ratio'])
        
        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model using build_model function
        model = build_model(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            output_seq_len=self.output_seq_len,
            device=torch.device(self.device)
        )
        
        # Optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                output = model(X_batch, y_batch, teacher_forcing_ratio=teacher_forcing)
                loss = criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    output = model(X_batch, y_batch, teacher_forcing_ratio=0.0)
                    val_loss += criterion(output, y_batch).item()
            
            val_loss /= len(val_loader)
            
            # Report to Optuna for pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        return best_val_loss


def run_optuna_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_seq_len: int,
    search_space: dict,
    n_trials: int = 50,
    timeout: int = 3600,
    device: str = 'cuda',
    log_dir: str = 'logs',
    db_path: str = None
) -> dict:
    """
    Run Optuna hyperparameter optimization
    
    Returns:
        dict with best_params, best_value, study, etc.
    """
    
    set_seed(RANDOM_SEED)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create objective
    objective = Seq2SeqObjective(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        output_seq_len=output_seq_len,
        search_space=search_space,
        device=device
    )
    
    # Create study - use in-memory storage to avoid database issues
    study_name = f"seq2seq_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if db_path:
        # Delete old database if exists to avoid corruption
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Removed old database: {db_path}")
        storage = f"sqlite:///{db_path}"
    else:
        storage = None  # In-memory
    
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=False,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Run optimization
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Starting Optuna Optimization")
    print(f"{'='*60}")
    print(f"Trials: {n_trials}")
    print(f"Timeout: {timeout/3600:.1f} hours")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    duration = time.time() - start_time
    
    # Get results
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    
    results = {
        'best_params': study.best_trial.params,
        'best_value': study.best_trial.value,
        'best_trial_number': study.best_trial.number,
        'n_completed_trials': len(completed_trials),
        'n_pruned_trials': len(pruned_trials),
        'duration_minutes': duration / 60,
        'study': study
    }
    
    # Save best params
    best_params_path = os.path.join(log_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(results['best_params'], f, indent=2)
    print(f"\nBest params saved to: {best_params_path}")
    
    return results
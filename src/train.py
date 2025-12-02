"""
Training Functions
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict
import numpy as np

from .logger import TrainingLogger

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_one_epoch(model: nn.Module,
                    train_loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    teacher_forcing_ratio: float = 0.5,
                    gradient_clip: float = 1.0) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch_x, batch_y in tqdm(train_loader, desc="Training", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_x, batch_y, teacher_forcing_ratio)
        
        # Calculate loss
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model: nn.Module,
             val_loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> float:
    """Validate model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(val_loader, desc="Validating", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass (no teacher forcing during validation)
            outputs = model(batch_x, None, 0.0)
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          num_epochs: int = 100,
          learning_rate: float = 0.001,
          weight_decay: float = 1e-5,
          teacher_forcing_ratio: float = 0.5,
          gradient_clip: float = 1.0,
          early_stopping_patience: int = 15,
          lr_scheduler_type: str = 'ReduceLROnPlateau',
          checkpoint_dir: str = None,
          best_model_path: str = None,
          device: torch.device = None,
          log_dir: str = None) -> Dict:
    """
    Complete training loop with logging
    
    Returns:
        Dictionary containing training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup logger
    training_logger = TrainingLogger(log_dir)
    
    training_logger.logger.info(f"Training on {device}")
    training_logger.log_hyperparams({
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'teacher_forcing_ratio': teacher_forcing_ratio,
        'gradient_clip': gradient_clip,
        'early_stopping_patience': early_stopping_patience,
        'lr_scheduler': lr_scheduler_type
    })
    
    model = model.to(device)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    if lr_scheduler_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
    elif lr_scheduler_type == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    elif lr_scheduler_type == 'StepLR':
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        scheduler = None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode='min')
    
    # History
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Decay teacher forcing ratio
        current_tf_ratio = teacher_forcing_ratio * (1 - epoch / num_epochs)
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            current_tf_ratio, gradient_clip
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update scheduler
        if scheduler is not None:
            if lr_scheduler_type == 'ReduceLROnPlateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        
        # Save best model
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            
            if best_model_path:
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, best_model_path)
                training_logger.log_best_model(epoch + 1, val_loss, best_model_path)
        
        # Save checkpoint
        if checkpoint_dir and (epoch + 1) % 10 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
        
        # Log progress
        epoch_time = time.time() - epoch_start
        training_logger.log_epoch(epoch + 1, train_loss, val_loss, current_lr, epoch_time)
        
        # Early stopping
        if early_stopping(val_loss):
            training_logger.log_early_stopping(epoch + 1, early_stopping_patience)
            break
    
    total_time = time.time() - start_time
    training_logger.log_training_complete(total_time, history['best_epoch'] + 1, history['best_val_loss'])
    
    # Load best model
    if best_model_path and os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        training_logger.logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    return history

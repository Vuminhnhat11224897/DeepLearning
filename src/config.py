"""
Configuration file for hyperparameters and paths
"""
import os

# ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'Metro_Interstate_Traffic_Volume.csv')
CLEANED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'cleaned_data.csv')
FEATURED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'featured_data.csv')
SELECTED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'selected_features.csv')
SEQUENCES_DIR = os.path.join(DATA_DIR, 'sequences')

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, 'checkpoints')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pth')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# Results paths
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
EDA_FIGURES_DIR = os.path.join(FIGURES_DIR, 'eda')
TRAINING_FIGURES_DIR = os.path.join(FIGURES_DIR, 'training')
EVAL_FIGURES_DIR = os.path.join(FIGURES_DIR, 'evaluation')
METRICS_PATH = os.path.join(RESULTS_DIR, 'metrics.json')
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, 'predictions.csv')

# Logs paths
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
OPTUNA_DB_PATH = os.path.join(LOGS_DIR, 'optuna_study.db')

# ==================== DATA CONFIG ====================
TARGET_COLUMN = 'traffic_volume'
DATE_COLUMN = 'date_time'

# ==================== SEQUENCE CONFIG ====================
INPUT_SEQ_LEN = 24      # Use 24 hours of history
OUTPUT_SEQ_LEN = 5      # Predict next 5 hours

# ==================== TRAIN/VAL/TEST SPLIT ====================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==================== MODEL HYPERPARAMETERS ====================
# Encoder
ENCODER_INPUT_SIZE = None  # Will be set based on number of features
ENCODER_HIDDEN_SIZE = 128
ENCODER_NUM_LAYERS = 2
ENCODER_DROPOUT = 0.2
ENCODER_BIDIRECTIONAL = True

# Decoder
DECODER_HIDDEN_SIZE = 128
DECODER_NUM_LAYERS = 2
DECODER_DROPOUT = 0.2

# ==================== TRAINING HYPERPARAMETERS ====================
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
GRADIENT_CLIP = 1.0
TEACHER_FORCING_RATIO = 0.5

# Learning rate scheduler
LR_SCHEDULER = 'ReduceLROnPlateau'  # or 'CosineAnnealingLR', 'StepLR'
LR_PATIENCE = 5
LR_FACTOR = 0.5
LR_MIN = 1e-7

# ==================== OPTUNA HYPERPARAMETER SEARCH SPACE ====================
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = 3600 * 2  # 2 hours

OPTUNA_SEARCH_SPACE = {
    'hidden_size': [64, 128, 256],
    'num_layers': [1, 2, 3],
    'dropout': (0.1, 0.5),
    'learning_rate': (1e-4, 1e-2),
    'batch_size': [32, 64, 128],
    'weight_decay': (1e-6, 1e-3),
    'teacher_forcing_ratio': (0.3, 0.7),
}

# ==================== DEVICE ====================
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== RANDOM SEED ====================
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

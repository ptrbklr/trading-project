from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml

@dataclass
class DataConfig:
    dir: str
    interval_minutes: int
    date_col: Optional[str]
    close_col: Optional[str]
    col_names: str
    add_features: bool
    train_split: float
    symbol: Optional[str] = None
    symbols: Optional[list] = None
    target_symbol: Optional[str] = None
    reciprocal_source: Optional[str] = None
    lookback_hours: Optional[float] = None
    predict_returns: bool = False
    futures_path: Optional[str] = None
    futures_columns: Optional[list] = None

@dataclass
class ModelConfig:
    type: str
    hidden_size: int
    num_layers: int
    dropout: float
    seq_len: int
    attention: bool
    classification_head: bool = False

@dataclass
class SchedulerConfig:
    type: str
    factor: float
    patience: int

@dataclass
class EarlyStoppingConfig:
    enabled: bool
    patience: int
    min_delta: float

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    lr: float
    grad_clip: float
    optimizer: str
    scheduler: SchedulerConfig
    early_stopping: EarlyStoppingConfig
    seed: int = 42
    use_xgboost: bool = False
    ensemble_weight: float = 0.3
    xgboost: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LossConfig:
    type: str
    mse_weight: float
    mae_weight: float
    directional_weight: float
    directional_alpha: float
    classification_weight: float = 0.0
    dir_deadband: float = 0.0

@dataclass
class ArtifactsConfig:
    base_dir: str
    save_best_only: bool
    save_every_n_epochs: int

@dataclass
class Config:
    # All required fields (no defaults) FIRST
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    artifacts: ArtifactsConfig
    # Optional fields (with defaults) LAST
    experiment_name: str = "default_experiment"  # ← MOVED TO THE END

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    
    # Get XGBoost config if it exists
    xgboost_config = raw.get('xgboost', {})
    
    # Update training config to include XGBoost settings
    training_raw = raw['training'].copy()
    training_raw['use_xgboost'] = raw.get('use_xgboost', False)
    training_raw['ensemble_weight'] = raw.get('ensemble_weight', 0.3)
    training_raw['xgboost'] = xgboost_config
    
    # Get experiment name with default if missing
    experiment_name = raw.get('experiment_name', 'default_experiment')
    
    return Config(
        experiment_name=experiment_name,
        data=DataConfig(**raw['data']),
        model=ModelConfig(**raw['model']),
        training=TrainingConfig(
            **{k: v for k, v in training_raw.items() 
               if k not in ['scheduler', 'early_stopping']},
            scheduler=SchedulerConfig(**raw['training']['scheduler']),
            early_stopping=EarlyStoppingConfig(**raw['training']['early_stopping']),
        ),
        loss=LossConfig(**raw['loss']),
        artifacts=ArtifactsConfig(**raw['artifacts']),
    )
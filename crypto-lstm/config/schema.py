from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class DataConfig:
    dir: str
    symbol: str
    interval_minutes: int
    date_col: Optional[str]
    close_col: Optional[str]
    col_names: str
    add_features: bool
    train_split: float
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
    experiment_name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    artifacts: ArtifactsConfig

def load_config(path: str) -> Config:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    data_raw = dict(raw['data'])
    data_raw.setdefault('lookback_hours', None)

    return Config(
        experiment_name=raw['experiment_name'],
        data=DataConfig(**data_raw),
        model=ModelConfig(**raw['model']),
        training=TrainingConfig(
            **{k: v for k, v in raw['training'].items() if k not in ['scheduler', 'early_stopping']},
            scheduler=SchedulerConfig(**raw['training']['scheduler']),
            early_stopping=EarlyStoppingConfig(**raw['training']['early_stopping']),
        ),
        loss=LossConfig(**raw['loss']),
        artifacts=ArtifactsConfig(**raw['artifacts']),
    )

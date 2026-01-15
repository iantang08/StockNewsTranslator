"""Shared constants for the stock_news_translator package."""
from typing import Dict
from dataclasses import dataclass


# Sentiment label mapping (used across training, testing, and prediction)
LABEL_MAP: Dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

REVERSE_LABEL_MAP: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}


@dataclass
class ModelConfig:
    """Model hyperparameters configuration."""
    vocab_size: int = 25000
    embedding_dim: int = 256
    gru_units: int = 256
    num_classes: int = 3
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 8
    validation_split: float = 0.2
    train_size: float = 0.7


@dataclass
class Paths:
    """Default file paths."""
    model: str = "trained_model/model.keras"
    tokenizer: str = "trained_model/tokenizer.pickle"
    training_data_dir: str = "training_data"

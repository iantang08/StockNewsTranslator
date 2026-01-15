"""Lazy data loading utilities for training and testing datasets."""
from typing import Dict, Optional
from pathlib import Path

import pandas as pd

from ..constants import Paths


class DataLoader:
    """Lazy loader for training and testing datasets with caching."""

    def __init__(self, data_dir: str = Paths.training_data_dir) -> None:
        """Initialize the data loader.

        Args:
            data_dir: Directory containing CSV datasets.
        """
        self.data_dir = Path(data_dir)
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_dataset(self, filename: str) -> pd.DataFrame:
        """Load a CSV dataset with caching.

        Args:
            filename: Name of the CSV file to load.

        Returns:
            DataFrame with Label and Text columns.
        """
        if filename not in self._cache:
            self._cache[filename] = pd.read_csv(
                self.data_dir / filename,
                names=["Label", "Text"],
                encoding="latin-1"
            )
        return self._cache[filename]

    def clear_cache(self) -> None:
        """Clear the dataset cache."""
        self._cache.clear()

    @property
    def train_dataset(self) -> pd.DataFrame:
        """Load the main training dataset."""
        return self.load_dataset("train_dataset.csv")

    @property
    def oov_train_dataset(self) -> pd.DataFrame:
        """Load the out-of-vocabulary training dataset."""
        return self.load_dataset("oov_train_dataset.csv")

    @property
    def test_dataset_1(self) -> pd.DataFrame:
        """Load test dataset 1."""
        return self.load_dataset("test_dataset_1.csv")

    @property
    def test_dataset_2(self) -> pd.DataFrame:
        """Load test dataset 2."""
        return self.load_dataset("test_dataset_2.csv")

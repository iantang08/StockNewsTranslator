"""Shared preprocessing utilities for text sequences."""
from typing import Tuple, Any

import pandas as pd
import numpy as np
from numpy.typing import NDArray

# Handle different Keras versions - prefer tf_keras for legacy compatibility
try:
    from tf_keras.preprocessing.sequence import pad_sequences
    from tf_keras.preprocessing.text import Tokenizer
except ImportError:
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import Tokenizer
    except ImportError:
        from keras.preprocessing.sequence import pad_sequences
        from keras.preprocessing.text import Tokenizer

from ..constants import LABEL_MAP

# Type alias for Tokenizer (works with both keras versions)
TokenizerType = Any


def tokenize_texts(
    texts: pd.Series,
    tokenizer: TokenizerType,
    max_seq: int
) -> NDArray[np.int32]:
    """Convert text series to padded sequences.

    Args:
        texts: Series of text strings to tokenize.
        tokenizer: Fitted Keras Tokenizer.
        max_seq: Maximum sequence length for padding.

    Returns:
        Padded sequences as numpy array.
    """
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_seq, padding="post")


def prepare_prediction_input(
    dataframe: pd.DataFrame,
    tokenizer: TokenizerType,
    max_seq: int
) -> NDArray[np.int32]:
    """Prepare dataframe for prediction.

    Args:
        dataframe: DataFrame with 'Text' column.
        tokenizer: Fitted Keras Tokenizer.
        max_seq: Maximum sequence length.

    Returns:
        Padded sequences ready for model prediction.
    """
    return tokenize_texts(dataframe["Text"], tokenizer, max_seq)


def prepare_labeled_data(
    dataframe: pd.DataFrame,
    tokenizer: TokenizerType,
    max_seq: int
) -> Tuple[NDArray[np.int32], pd.Series]:
    """Prepare labeled data for training or testing.

    Args:
        dataframe: DataFrame with 'Text' and 'Label' columns.
        tokenizer: Fitted Keras Tokenizer.
        max_seq: Maximum sequence length.

    Returns:
        Tuple of (padded sequences, numeric labels).
    """
    sequences = tokenize_texts(dataframe["Text"], tokenizer, max_seq)
    labels = dataframe["Label"].replace(LABEL_MAP)
    return sequences, labels

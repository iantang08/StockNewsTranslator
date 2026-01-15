"""Sentiment model wrapper for financial text analysis."""
from typing import Optional
import pickle
import sys

import pandas as pd
import numpy as np
from numpy.typing import NDArray

# Handle different Keras versions - prefer tf_keras for loading legacy models
try:
    import tf_keras as keras
    # Create module aliases for unpickling legacy tokenizers
    import tf_keras.src.preprocessing.text
    sys.modules['keras.src.preprocessing'] = tf_keras.src.preprocessing
    sys.modules['keras.src.preprocessing.text'] = tf_keras.src.preprocessing.text
except ImportError:
    try:
        import tensorflow.keras as keras
    except ImportError:
        import keras

from ..constants import Paths
from .preprocessing import prepare_prediction_input


def _build_model_from_config(input_length: int = 71, vocab_size: int = 25000) -> keras.Model:
    """Rebuild the model architecture for loading legacy weights."""
    # Use generic names to match the H5 weight file structure
    inputs = keras.Input(shape=(input_length,), name="input_layer")
    x = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=256,
        input_length=input_length,
        name="embedding"
    )(inputs)
    x = keras.layers.GRU(256, return_sequences=True, activation="tanh", name="gru")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    outputs = keras.layers.Dense(3, activation="softmax", name="dense")(x)

    # Use underscore in name to avoid TensorFlow scope name errors
    model = keras.Model(inputs=inputs, outputs=outputs, name="stock_news_sentiment")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


class StockNewsTranslatorModel:
    """Wrapper around the trained GRU sentiment model for financial text."""

    def __init__(
        self,
        model_path: str = Paths.model,
        tokenizer_path: str = Paths.tokenizer,
        verbose: bool = True
    ) -> None:
        """Initialize the sentiment model.

        Args:
            model_path: Path to the trained Keras model file.
            tokenizer_path: Path to the pickled tokenizer.
            verbose: Whether to print model summary and loading info.
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        # Try loading with different options for compatibility
        self.model = self._load_model_compat(self.model_path)
        if verbose:
            self.model.summary()

        with open(self.tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        self.max_seq: int = self.model.layers[0].input_shape[0][1]

        if verbose:
            print(f"Vocabulary Length: {len(self.tokenizer.word_index) + 1}")
            print("\n_________________________________________________________________\n\n")
            print("Model and Tokenizer Loaded.")
            print("\n\n_________________________________________________________________")

    def _load_model_compat(self, model_path: str) -> keras.Model:
        """Load model with compatibility for different Keras versions."""
        import zipfile
        import tempfile
        import os
        import h5py

        # First try direct loading
        try:
            return keras.models.load_model(model_path)
        except (ValueError, TypeError, OSError) as e:
            print(f"Direct load failed: {e}, trying alternative method...")

        # Try extracting and loading weights manually
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(model_path, 'r') as zf:
                    zf.extractall(tmpdir)

                weights_path = os.path.join(tmpdir, 'model.weights.h5')
                if os.path.exists(weights_path):
                    # Rebuild architecture
                    model = _build_model_from_config()

                    # Manually load weights from H5 file (handles Windows backslash paths)
                    with h5py.File(weights_path, 'r') as f:
                        # Load embedding weights
                        emb_weights = f['layers\\embedding/vars/0'][:]
                        model.get_layer('embedding').set_weights([emb_weights])

                        # Load GRU weights
                        gru_kernel = f['layers\\gru\\cell/vars/0'][:]
                        gru_recurrent = f['layers\\gru\\cell/vars/1'][:]
                        gru_bias = f['layers\\gru\\cell/vars/2'][:]
                        model.get_layer('gru').set_weights([gru_kernel, gru_recurrent, gru_bias])

                        # Load Dense weights
                        dense_kernel = f['layers\\dense/vars/0'][:]
                        dense_bias = f['layers\\dense/vars/1'][:]
                        model.get_layer('dense').set_weights([dense_kernel, dense_bias])

                    return model
        except Exception as e:
            print(f"Alternative load also failed: {e}")
            raise RuntimeError(f"Could not load model from {model_path}") from e

    def _get_prediction_input(self, dataframe: pd.DataFrame) -> NDArray[np.int32]:
        """Preprocess dataframe for prediction.

        Args:
            dataframe: DataFrame with 'Text' column.

        Returns:
            Padded sequences for model input.
        """
        return prepare_prediction_input(dataframe, self.tokenizer, self.max_seq)

    def model_predict(
        self,
        data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Run sentiment prediction on text data.

        Args:
            data: DataFrame with 'Text' column containing text to analyze.
            output_path: Optional path to save predictions as CSV.

        Returns:
            DataFrame with sentiment probabilities and final prediction.
        """
        predict_sequence = self._get_prediction_input(data)
        predictions = self.model.predict(predict_sequence)

        negative = predictions[:, 0].tolist()
        neutral = predictions[:, 1].tolist()
        positive = predictions[:, 2].tolist()

        table = {
            "Positive": positive,
            "Negative": negative,
            "Neutral": neutral
        }

        df = pd.DataFrame(table).round(5)
        prediction_df = df.join(data)

        prediction_df["prediction"] = prediction_df.apply(
            lambda x: (
                "neutral" if float(x["Neutral"]) >= max(float(x["Positive"]), float(x["Negative"]))
                else "positive" if float(x["Positive"]) > max(float(x["Neutral"]), float(x["Negative"]))
                else "negative"
            ),
            axis=1
        )

        if output_path:
            prediction_df.to_csv(output_path, index=False)

        return prediction_df

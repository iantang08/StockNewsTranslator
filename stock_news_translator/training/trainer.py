"""Model training and retraining utilities."""
from typing import Optional, Tuple, Any
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Handle different Keras versions - prefer tf_keras for legacy compatibility
try:
    import tf_keras as keras
    from tf_keras.preprocessing.text import Tokenizer
    from tf_keras.preprocessing.sequence import pad_sequences
except ImportError:
    try:
        import tensorflow.keras as keras
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except ImportError:
        import keras
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences

from ..constants import ModelConfig, Paths, LABEL_MAP
from .data_loader import DataLoader

# Type alias for Tokenizer
TokenizerType = Any


class ModelTrainer:
    """Handles model training and retraining with lazy data loading."""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        data_dir: str = Paths.training_data_dir
    ) -> None:
        """Initialize the trainer.

        Args:
            config: Model configuration. Uses defaults if not provided.
            data_dir: Directory containing training data.
        """
        self.config = config or ModelConfig()
        self.data_loader = DataLoader(data_dir)
        self._tokenizer: Optional[Tokenizer] = None

    def _create_tokenizer_and_sequences(
        self,
        texts: pd.Series
    ) -> Tuple[Tokenizer, np.ndarray, int]:
        """Create tokenizer and convert texts to padded sequences.

        Args:
            texts: Series of text strings.

        Returns:
            Tuple of (tokenizer, padded_sequences, max_sequence_length).
        """
        tokenizer = Tokenizer(oov_token=1)
        tokenizer.fit_on_texts(texts)

        sequences = tokenizer.texts_to_sequences(texts)
        print(f"Vocab Length: {len(tokenizer.word_index) + 1}")

        max_seq = int(np.max([len(s) for s in sequences]))
        padded = pad_sequences(sequences, maxlen=max_seq, padding="post")

        self._tokenizer = tokenizer
        return tokenizer, padded, max_seq

    def _prepare_training_data(
        self,
        dataframe: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """Prepare data for training with train/test split.

        Args:
            dataframe: DataFrame with Label and Text columns.

        Returns:
            Tuple of (train_sequences, test_sequences, y_train, y_test).
        """
        _, sequences, _ = self._create_tokenizer_and_sequences(dataframe["Text"])
        labels = dataframe["Label"].replace(LABEL_MAP)

        return train_test_split(
            sequences,
            labels,
            train_size=self.config.train_size,
            shuffle=True,
            random_state=1
        )

    def _prepare_retrain_data(
        self,
        dataframe: pd.DataFrame,
        tokenizer: TokenizerType,
        max_seq: int
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """Prepare out-of-vocabulary data for retraining.

        Args:
            dataframe: DataFrame with Label and Text columns.
            tokenizer: Existing fitted tokenizer.
            max_seq: Maximum sequence length from original model.

        Returns:
            Tuple of (train_sequences, test_sequences, y_train, y_test).
        """
        sequences = tokenizer.texts_to_sequences(dataframe["Text"])
        sequences = pad_sequences(sequences, maxlen=max_seq, padding="post")
        labels = dataframe["Label"].replace(LABEL_MAP)

        return train_test_split(
            sequences,
            labels,
            train_size=self.config.train_size,
            shuffle=True,
            random_state=1
        )

    def _build_model(self, input_length: int) -> keras.Model:
        """Build the GRU sentiment model architecture.

        Args:
            input_length: Length of input sequences.

        Returns:
            Compiled Keras model.
        """
        inputs = keras.Input(shape=(input_length,))
        x = keras.layers.Embedding(
            input_dim=self.config.vocab_size,
            output_dim=self.config.embedding_dim,
            input_length=input_length
        )(inputs)
        x = keras.layers.GRU(
            self.config.gru_units,
            return_sequences=True,
            activation="tanh"
        )(x)
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(
            self.config.num_classes,
            activation="softmax"
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def create_model(
        self,
        min_acc_save: float = 0.75,
        output_dir: Optional[str] = None
    ) -> Tuple[keras.Model, float]:
        """Create and train a new sentiment model.

        Args:
            min_acc_save: Minimum accuracy threshold to save the model.
            output_dir: Directory to save model and tokenizer. Uses current dir if None.

        Returns:
            Tuple of (trained_model, test_accuracy).
        """
        train_data = self.data_loader.train_dataset
        train_seq, test_seq, y_train, y_test = self._prepare_training_data(train_data)

        model = self._build_model(train_seq.shape[1])

        model.fit(
            train_seq,
            y_train,
            validation_split=self.config.validation_split,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                )
            ]
        )

        loss, accuracy = model.evaluate(test_seq, y_test)
        print(f"New model created, accuracy: {100 * accuracy:.2f}%, loss: {100 * loss:.2f}")
        print(y_test.value_counts())

        if accuracy > min_acc_save:
            output_path = Path(output_dir) if output_dir else Path(".")
            model_name = f"modelGRU({round(accuracy * 1000)}).keras"
            model.save(output_path / model_name)

            tokenizer_path = output_path / "tokenizer.pickle"
            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(self._tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return model, accuracy

    def retrain_model(
        self,
        model_path: str,
        min_acc_save: float = 0.75,
        output_dir: Optional[str] = None,
        tokenizer_path: str = "tokenizer.pickle"
    ) -> Tuple[keras.Model, float]:
        """Retrain an existing model with out-of-vocabulary data.

        Args:
            model_path: Path to the model to retrain.
            min_acc_save: Minimum accuracy threshold to save the model.
            output_dir: Directory to save retrained model.
            tokenizer_path: Path to the tokenizer pickle file.

        Returns:
            Tuple of (retrained_model, test_accuracy).
        """
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)

        model = keras.models.load_model(model_path)
        model.summary()

        max_seq = model.layers[0].input_shape[0][1]
        retrain_data = self.data_loader.oov_train_dataset
        train_seq, test_seq, y_train, y_test = self._prepare_retrain_data(
            retrain_data, tokenizer, max_seq
        )

        model.fit(
            train_seq,
            y_train,
            validation_split=self.config.validation_split,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                )
            ]
        )

        loss, accuracy = model.evaluate(test_seq, y_test)
        print(f"Model retrained, accuracy: {100 * accuracy:.2f}%, loss: {100 * loss:.2f}")

        if accuracy > min_acc_save:
            output_path = Path(output_dir) if output_dir else Path(".")
            model_name = f"modelRetrained{round(accuracy * 1000)}.keras"
            model.save(output_path / model_name)

        return model, accuracy


# Backward-compatible functions
def create_new_model(min_acc_save: float) -> float:
    """Create a new model (backward-compatible wrapper).

    Args:
        min_acc_save: Minimum accuracy threshold to save the model.

    Returns:
        Test accuracy of the trained model.
    """
    trainer = ModelTrainer()
    _, accuracy = trainer.create_model(min_acc_save)
    return accuracy


def retrain_model(model_path: str, min_acc_save: float) -> float:
    """Retrain an existing model (backward-compatible wrapper).

    Args:
        model_path: Path to the model to retrain.
        min_acc_save: Minimum accuracy threshold to save.

    Returns:
        Test accuracy of the retrained model.
    """
    trainer = ModelTrainer()
    _, accuracy = trainer.retrain_model(model_path, min_acc_save)
    return accuracy

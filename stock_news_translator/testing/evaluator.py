"""Model evaluation utilities."""
from typing import Dict, Tuple, Any

# Handle different Keras versions - prefer tf_keras for legacy compatibility
try:
    import tf_keras as keras
except ImportError:
    try:
        import tensorflow.keras as keras
    except ImportError:
        import keras

from ..constants import Paths
from ..model.preprocessing import prepare_labeled_data
from ..training.data_loader import DataLoader

# Type alias for Tokenizer
TokenizerType = Any


class ModelEvaluator:
    """Handles model evaluation on test datasets."""

    def __init__(self, data_dir: str = Paths.training_data_dir) -> None:
        """Initialize the evaluator.

        Args:
            data_dir: Directory containing test datasets.
        """
        self.data_loader = DataLoader(data_dir)

    def evaluate(
        self,
        model_path: str,
        tokenizer: TokenizerType,
        dataset_name: str = "test_dataset_1"
    ) -> Tuple[float, float]:
        """Evaluate model on a specific test dataset.

        Args:
            model_path: Path to the trained model.
            tokenizer: Fitted tokenizer.
            dataset_name: Name of the test dataset (without .csv extension).

        Returns:
            Tuple of (accuracy, loss).
        """
        model = keras.models.load_model(model_path)
        model.summary()

        dataset = self.data_loader.load_dataset(f"{dataset_name}.csv")
        max_seq = model.layers[0].input_shape[0][1]
        sequences, labels = prepare_labeled_data(dataset, tokenizer, max_seq)

        loss, accuracy = model.evaluate(sequences, labels)
        print(f"Testing on {dataset_name}: accuracy={100 * accuracy:.2f}%, loss={100 * loss:.2f}")

        return accuracy, loss

    def evaluate_all(
        self,
        model_path: str,
        tokenizer: TokenizerType
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model on all available test datasets.

        Args:
            model_path: Path to the trained model.
            tokenizer: Fitted tokenizer.

        Returns:
            Dictionary mapping dataset names to accuracy/loss results.
        """
        model = keras.models.load_model(model_path)
        model.summary()

        max_seq = model.layers[0].input_shape[0][1]
        results: Dict[str, Dict[str, float]] = {}

        for dataset_name in ["test_dataset_1", "test_dataset_2"]:
            dataset = self.data_loader.load_dataset(f"{dataset_name}.csv")
            sequences, labels = prepare_labeled_data(dataset, tokenizer, max_seq)

            loss, accuracy = model.evaluate(sequences, labels)
            results[dataset_name] = {"accuracy": accuracy, "loss": loss}
            print(f"Testing on {dataset_name}: accuracy={100 * accuracy:.2f}%, loss={100 * loss:.2f}")

        return results


def test_model(model_path: str, tokenizer: TokenizerType) -> None:
    """Test model on all datasets (backward-compatible wrapper).

    Args:
        model_path: Path to the model to test.
        tokenizer: Fitted tokenizer.
    """
    evaluator = ModelEvaluator()
    evaluator.evaluate_all(model_path, tokenizer)

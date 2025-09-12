import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import keras
import pickle


class StockNewsTranslatorModel:
    """Wrapper around the trained GRU sentiment model for financial text."""

    def __init__(self, model_path="trained_model/model.keras", tokenizer_path="trained_model/tokenizer.pickle"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        # Load model and tokenizer
        self.model = keras.models.load_model(self.model_path)
        self.model.summary()

        with open(self.tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        print(f"Vocabulary Length: {len(self.tokenizer.word_index) + 1}")
        self.max_seq = self.model.layers[0].input_shape[0][1]
        print("\n_________________________________________________________________\n\n")
        print("Model and Tokenizer Loaded.")
        print("\n\n_________________________________________________________________")

    # Preprocessing for prediction
    def _get_prediction_input(self, dataframe: pd.DataFrame):
        dataframe = dataframe.copy()
        sequences = self.tokenizer.texts_to_sequences(dataframe["Text"])
        sequences = pad_sequences(sequences, maxlen=self.max_seq, padding="post")
        return sequences

    # predict on data
    def model_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        predict_sequence = self._get_prediction_input(data)
        predictions = self.model.predict(predict_sequence)

        negative = predictions[:, 0].tolist()
        neutral = predictions[:, 1].tolist()
        positive = predictions[:, 2].tolist()

        table = {"Positive": positive,
                 "Negative": negative,
                 "Neutral": neutral}

        df = pd.DataFrame(table).round(5)
        prediction_df = df.join(data)

        prediction_df["prediction"] = prediction_df.apply(lambda x:
                                                          "neutral" if float(x["Neutral"]) >= max(float(x["Positive"]), float(x["Negative"]))
                                                          else "positive" if float(x["Positive"]) > max(float(x["Neutral"]), float(x["Negative"]))
                                                          else "negative", axis=1)
        # Not automatically writing a CSV inside library call – responsibility of caller.
        return prediction_df

from typing import List, Tuple, Optional
import re
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

import yfinance as yf
from newspaper import Article

from StockNewsTranslator import StockNewsTranslatorModel as SentimentModel

TICKER_REGEX = re.compile(r"\b[A-Z]{2,5}\b")

@dataclass
class StockSentiment:
    ticker: str
    positive: float
    negative: float
    neutral: float
    recommendation_score: float = field(init=False)
    likelihood_up: float = field(init=False)
    current_price: Optional[float] = None

    def __post_init__(self):
        # Heuristic: positive - negative maps to score; convert to 0-1 likelihood via logistic
        self.recommendation_score = self.positive - self.negative
        # Scale factor chosen empirically; tweak as needed
        self.likelihood_up = float(1 / (1 + np.exp(-5 * self.recommendation_score)))

class StockNewsRecommender:
    """Pipeline to scrape a news article, extract tickers, run sentiment analysis and recommend stocks."""

    def __init__(self,
                 sentiment_model: Optional[SentimentModel] = None,
                 positive_threshold: float = 0.5,
                 min_occurrences: int = 1):
        self.sentiment_model = sentiment_model or SentimentModel()
        self.positive_threshold = positive_threshold
        self.min_occurrences = min_occurrences

    @staticmethod
    def _extract_article_text(url_or_text: str) -> str:
        # If it looks like a URL, try to download article
        if url_or_text.lower().startswith("http"):
            article = Article(url_or_text)
            article.download()
            article.parse()
            return article.text
        return url_or_text

    @staticmethod
    def _find_candidate_tickers(text: str) -> List[str]:
        # Very naive ticker extraction. Filters out common English words by querying yfinance info.
        matches = set(TICKER_REGEX.findall(text))
        valid = []
        for symbol in matches:
            try:
                info = yf.Ticker(symbol).fast_info
                # fast_info returns None if symbol invalid in many cases
                if info and info.get("lastPrice") is not None:
                    valid.append(symbol)
            except Exception:
                continue
        return valid

    def _segment_sentences(self, text: str) -> List[str]:
        # Split by period for simplicity. Advanced NLP could be added.
        sentences = [s.strip() for s in re.split(r"[\.!?]", text) if s.strip()]
        return sentences

    def _sentiment_dataframe(self, sentences: List[str]) -> pd.DataFrame:
        df = pd.DataFrame({"Text": sentences})
        predictions = self.sentiment_model.model_predict(df)
        return predictions

    def recommend(self, url_or_text: str, top_k: int = 5) -> List[StockSentiment]:
        text = self._extract_article_text(url_or_text)
        tickers = self._find_candidate_tickers(text)
        if not tickers:
            return []

        sentences = self._segment_sentences(text)
        predictions = self._sentiment_dataframe(sentences)

        # Map sentiments to tickers
        ticker_scores = {t: [] for t in tickers}
        for _, row in predictions.iterrows():
            sentence = row["Text"]
            contained = [t for t in tickers if t in sentence]
            if not contained:
                continue
            for t in contained:
                ticker_scores[t].append((row["Positive"], row["Negative"], row["Neutral"]))

        recommendations: List[StockSentiment] = []
        for t, vals in ticker_scores.items():
            if len(vals) < self.min_occurrences:
                continue
            arr = np.array(vals)
            pos, neg, neu = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
            if pos - neg < self.positive_threshold:
                continue
            try:
                price = yf.Ticker(t).fast_info.get("lastPrice")
            except Exception:
                price = None
            recommendations.append(StockSentiment(ticker=t, positive=pos, negative=neg, neutral=neu, current_price=price))

        recommendations.sort(key=lambda x: x.recommendation_score, reverse=True)
        return recommendations[:top_k]

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Recommend stocks from a news article")
    parser.add_argument("input", help="URL to news article or raw text input")
    parser.add_argument("--top_k", type=int, default=5, help="Number of tickers to return")

    args = parser.parse_args()

    recommender = StockNewsRecommender()
    recs = recommender.recommend(args.input, top_k=args.top_k)
    print(json.dumps([rec.__dict__ for rec in recs], indent=2))

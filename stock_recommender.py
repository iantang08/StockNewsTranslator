from typing import List, Tuple, Optional, Dict
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
    def _find_candidate_tickers(text: str) -> Tuple[List[str], Dict[str, dict]]:
        # Extract uppercase tokens and validate via yfinance fast_info once per symbol.
        matches = set(TICKER_REGEX.findall(text))
        valid: List[str] = []
        ticker_info_map: Dict[str, dict] = {}
        for symbol in matches:
            try:
                info = yf.Ticker(symbol).fast_info
                if info and info.get("lastPrice") is not None:
                    valid.append(symbol)
                    ticker_info_map[symbol] = info
            except Exception:
                continue
        return valid, ticker_info_map

    @staticmethod
    def _compile_ticker_search(tickers: List[str]) -> Optional[re.Pattern]:
        if not tickers:
            return None
        # Compile a single regex to search all tickers as substrings (preserves original behavior)
        pattern = "|".join(re.escape(t) for t in tickers)
        return re.compile(pattern)

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
        tickers, ticker_info_map = self._find_candidate_tickers(text)
        if not tickers:
            return []

        sentences = self._segment_sentences(text)
        predictions = self._sentiment_dataframe(sentences)

        # Map sentiments to tickers
        ticker_scores = {t: [] for t in tickers}
        pattern = self._compile_ticker_search(tickers)
        for _, row in predictions.iterrows():
            sentence = row["Text"]
            if not pattern:
                continue
            matches = pattern.findall(sentence)
            if not matches:
                continue
            for t in set(matches):
                ticker_scores[t].append((row["Positive"], row["Negative"], row["Neutral"]))

        recommendations: List[StockSentiment] = []
        for t, vals in ticker_scores.items():
            if len(vals) < self.min_occurrences:
                continue
            count = len(vals)
            pos = sum(v[0] for v in vals) / count
            neg = sum(v[1] for v in vals) / count
            neu = sum(v[2] for v in vals) / count
            if pos - neg < self.positive_threshold:
                continue
            price = None
            try:
                price = ticker_info_map.get(t, {}).get("lastPrice")
                if price is None:
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

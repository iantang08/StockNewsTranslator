"""Stock News Translator - Backward compatibility module.

This module re-exports from the stock_news_translator package for
backward compatibility with existing code.
"""
from stock_news_translator.model.sentiment_model import StockNewsTranslatorModel

__all__ = ["StockNewsTranslatorModel"]

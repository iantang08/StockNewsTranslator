#!/usr/bin/env python3
"""Pipeline to fetch today's financial news and recommend stocks based on sentiment analysis."""
import argparse
import json
import sys
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

import yfinance as yf

from stock_recommender import StockSentiment

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",  # Tech
    "META", "TSLA", "JPM", "V", "MA",          # Tech/Finance
    "SPY", "QQQ"                                # ETFs
]


@dataclass
class NewsItem:
    """Represents a news article."""
    title: str
    link: str
    publisher: str
    ticker: str
    timestamp: Optional[int] = None


def fetch_news_for_ticker(ticker: str) -> List[NewsItem]:
    """Fetch recent news for a single ticker from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        List of NewsItem objects.
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news or []
        items = []
        for article in news:
            # Handle new yfinance API structure (nested under 'content')
            content = article.get("content", article)

            # Get title
            title = content.get("title", "")

            # Get link - try multiple possible locations
            link = ""
            if "canonicalUrl" in content and content["canonicalUrl"]:
                link = content["canonicalUrl"].get("url", "")
            elif "clickThroughUrl" in content and content["clickThroughUrl"]:
                link = content["clickThroughUrl"].get("url", "")
            elif "link" in article:
                link = article.get("link", "")

            # Get publisher
            publisher = ""
            if "provider" in content and content["provider"]:
                publisher = content["provider"].get("displayName", "")
            elif "publisher" in article:
                publisher = article.get("publisher", "")

            # Get timestamp
            timestamp = content.get("pubDate") or article.get("providerPublishTime")

            if title:  # Only add if we have a title
                items.append(NewsItem(
                    title=title,
                    link=link,
                    publisher=publisher,
                    ticker=ticker,
                    timestamp=timestamp
                ))
        return items
    except Exception as e:
        print(f"Warning: Failed to fetch news for {ticker}: {e}", file=sys.stderr)
        return []


def fetch_all_news(tickers: List[str]) -> List[NewsItem]:
    """Fetch news for all tickers and deduplicate by URL or title.

    Args:
        tickers: List of stock ticker symbols.

    Returns:
        Deduplicated list of NewsItem objects.
    """
    all_news: Dict[str, NewsItem] = {}  # Dedupe by URL or title

    for ticker in tickers:
        print(f"Fetching news for {ticker}...", file=sys.stderr)
        items = fetch_news_for_ticker(ticker)
        for item in items:
            # Use URL as key if available, otherwise use title
            key = item.link if item.link else item.title
            if key and key not in all_news:
                all_news[key] = item

    return list(all_news.values())


def analyze_news(
    news_items: List[NewsItem],
    sentiment_model,
    tickers: List[str]
) -> List[StockSentiment]:
    """Analyze news items and return stock recommendations.

    Args:
        news_items: List of news articles to analyze.
        sentiment_model: The sentiment model for predictions.
        tickers: List of tickers to get prices for.

    Returns:
        List of StockSentiment recommendations sorted by score.
    """
    import pandas as pd

    # Collect all news titles for sentiment analysis
    all_texts = [item.title for item in news_items]

    if not all_texts:
        return []

    # Run sentiment analysis on all titles at once
    df = pd.DataFrame({"Text": all_texts})
    print(f"Running sentiment analysis on {len(all_texts)} headlines...", file=sys.stderr)

    try:
        predictions = sentiment_model.model_predict(df)
    except Exception as e:
        print(f"Sentiment analysis failed: {e}", file=sys.stderr)
        return []

    # Calculate average sentiment across all news
    avg_positive = predictions["Positive"].mean()
    avg_negative = predictions["Negative"].mean()
    avg_neutral = predictions["Neutral"].mean()

    print(f"Average sentiment - Positive: {avg_positive:.2%}, Negative: {avg_negative:.2%}, Neutral: {avg_neutral:.2%}", file=sys.stderr)

    # Get current prices for requested tickers
    recommendations: List[StockSentiment] = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.fast_info
            price = info.get("lastPrice") if info else None

            recommendations.append(StockSentiment(
                ticker=ticker,
                positive=float(avg_positive),
                negative=float(avg_negative),
                neutral=float(avg_neutral),
                current_price=price
            ))
        except Exception as e:
            print(f"  Could not get price for {ticker}: {e}", file=sys.stderr)
            recommendations.append(StockSentiment(
                ticker=ticker,
                positive=float(avg_positive),
                negative=float(avg_negative),
                neutral=float(avg_neutral),
                current_price=None
            ))

    # Sort by recommendation score (highest first)
    recommendations.sort(key=lambda x: x.recommendation_score, reverse=True)
    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description="Fetch today's financial news and recommend stocks based on sentiment analysis"
    )
    parser.add_argument(
        "--tickers", "-t",
        nargs="+",
        default=DEFAULT_TICKERS,
        help=f"Stock tickers to monitor (default: {', '.join(DEFAULT_TICKERS[:5])}...)"
    )
    parser.add_argument(
        "--top_k", "-k",
        type=int,
        default=10,
        help="Number of top recommendations to return (default: 10)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path for JSON results (default: print to console)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    print(f"Stock News Recommender Pipeline", file=sys.stderr)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
    print(f"Tickers: {', '.join(args.tickers)}", file=sys.stderr)
    print("-" * 50, file=sys.stderr)

    # Step 1: Fetch news
    print("\n[1/3] Fetching news from Yahoo Finance...", file=sys.stderr)
    news_items = fetch_all_news(args.tickers)
    print(f"Found {len(news_items)} unique news articles", file=sys.stderr)

    if not news_items:
        print("No news articles found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Load sentiment model
    print("\n[2/3] Loading sentiment model...", file=sys.stderr)
    from stock_news_translator import StockNewsTranslatorModel
    sentiment_model = StockNewsTranslatorModel(verbose=False)

    print("\n[3/3] Analyzing news sentiment...", file=sys.stderr)
    recommendations = analyze_news(news_items, sentiment_model, args.tickers)

    # Step 3: Output results
    top_recs = recommendations[:args.top_k]

    result = {
        "date": datetime.now().isoformat(),
        "tickers_analyzed": args.tickers,
        "news_articles_processed": len(news_items),
        "recommendations": [
            {
                "ticker": r.ticker,
                "recommendation_score": round(r.recommendation_score, 4),
                "likelihood_up": round(r.likelihood_up, 4),
                "sentiment": {
                    "positive": round(r.positive, 4),
                    "negative": round(r.negative, 4),
                    "neutral": round(r.neutral, 4)
                },
                "current_price": r.current_price
            }
            for r in top_recs
        ]
    }

    output_json = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"\nResults saved to {args.output}", file=sys.stderr)
    else:
        print("\n" + "=" * 50, file=sys.stderr)
        print("RECOMMENDATIONS:", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
        print(output_json)


if __name__ == "__main__":
    main()

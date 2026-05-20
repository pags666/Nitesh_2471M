"""
This script fetches stock news articles, adds them to a database and computes sentiment scores for articles that don't have them yet.
"""
# Note: During multiprocessing, individual processes do not share memory. Hence, data is returned from each process and aggregated and not stored in a class variable directly.

import multiprocessing as mp
import sys

import pandas as pd
from loguru import logger
from tqdm import tqdm

import utils as utils
from database import DatabaseManager
from news_fetcher import TickerNewsObject
from datetime import datetime, timedelta
from database import DatabaseManager
from export_to_sheets import push_to_sheet

from utils import get_relative_date
from datetime import datetime, timedelta
import pytz

ist = pytz.timezone("Asia/Kolkata")
now = datetime.now(ist)

# Remove the default logger to prevent duplicate log entries.
logger.remove()
# Define the logging format.
fmt: str = '<white>{time:HH:mm:ss!UTC}({elapsed})</white> - <level> {level} - {message} </level>'
# Add a new logger configuration.
logger.add(sys.stderr, colorize=True, level='INFO', format=fmt, enqueue=True)


# Define the worker function outside the class for multiprocessing
def worker_collect_news(ticker_obj: TickerNewsObject) -> list[dict[str, str]]:
    """Collects news for a ticker object."""
    try:
        news: list[dict[str, str]] = ticker_obj.collect_news()
        return news
    except Exception as e:
        logger.error(f'Error collecting news for {ticker_obj.ticker}: {e}')
        return []  # Return empty list on error


def _log_source_summary(all_articles: list[dict[str, str]], total_tickers: int) -> None:
    """Log a per-source summary table showing article counts and data quality."""
    from collections import Counter

    source_counts: Counter[str] = Counter()
    for article in all_articles:
        source = article.get('source', 'Unknown')
        source_counts[source] += 1

    if not source_counts:
        logger.warning('No articles to summarize.')
        return

    # Print summary table
    logger.info('=' * 60)
    logger.info(f'{"SOURCE SUMMARY":^60}')
    logger.info('=' * 60)
    logger.info(f'{"Source":<30} {"Articles":>10} {"Avg/Ticker":>10}')
    logger.info('-' * 60)

    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        avg_per_ticker = count / max(total_tickers, 1)
        logger.info(f'{source:<30} {count:>10} {avg_per_ticker:>10.1f}')

    logger.info('-' * 60)
    logger.info(f'{"TOTAL":<30} {sum(source_counts.values()):>10}')
    logger.info('=' * 60)

    # Warn about underperforming sources
    expected_sources = [
        'Moneycontrol', 'Economic Times Markets', 'Business Standard',
        'CNBC TV18', 'Reuters',
    ]
    for source in expected_sources:
        count = source_counts.get(source, 0)
        if count == 0:
            logger.warning(
                f'⚠️  {source}: returned 0 articles across all tickers — '
                f'source may be down or blocked'
            )
        elif count < total_tickers * 0.1:
            logger.warning(
                f'⚠️  {source}: only {count} articles for {total_tickers} tickers — '
                f'data coverage may be low'
            )


def get_news(universe: str, multiprocess: bool) -> None:
    """
    Collect the news articles for a given universe of tickers and store them in the database.
    """
    dbm = DatabaseManager()

    # Fetch the tickers
    tickers_df: pd.Series[str] = dbm.get_index_constituents(universe).loc[:, 'ticker']
    ticker_objs: list[TickerNewsObject] = [
        TickerNewsObject(ticker) for ticker in tickers_df
    ]

    # Fetch and process news data for all tickers.
    logger.info(f'Start Processing {len(ticker_objs)} Tickers for {universe}')

    all_articles: list[dict[str, str]] = []

    if not multiprocess:
        # Process tickers sequentially.
        logger.info('Processing tickers sequentially.')
        for ticker_obj in tqdm(ticker_objs, desc='Processing Tickers'):
            try:
                news: list[dict[str, str]] = ticker_obj.collect_news()
                all_articles.extend(news)
            except Exception as e:
                logger.error(
                    f'Error collecting news for {ticker_obj.ticker} (sequential): {e}'
                )
    else:
        # Process tickers in parallel using multiprocessing.
        logger.info(f'Processing tickers in parallel using {mp.cpu_count()} processes.')
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # pool.map applies worker_collect_news to each item in ticker_objs
            # The result is a list of lists (one list of articles per ticker)
            results_list_of_lists: list[list[dict[str, str]]] = list(
                tqdm(
                    pool.map(
                        worker_collect_news, ticker_objs
                    ),  # Pass the worker function and the objects
                    total=len(ticker_objs),  # Use the correct list length
                    desc='Processing Tickers (Parallel)',
                )
            )
        # Flatten the list of lists into a single list of articles
        all_articles: list[dict[str, str]] = [
            article for sublist in results_list_of_lists for article in sublist
        ]

    # --- Aggregation and Processing ---
    logger.success(
        f'Collected {len(all_articles)} articles in total for {len(ticker_objs)} tickers'
    )

    # Log per-source summary
    _log_source_summary(all_articles, len(ticker_objs))

    # Check if any articles were collected
    if not all_articles:
        logger.warning(
            'No news articles found for any ticker after processing. Exiting!'
        )
        return

    # Create DataFrame directly from the flattened list
    articles_df = pd.DataFrame(all_articles)
    logger.info(
        f'Fetched {articles_df.shape[0]} articles from {len(ticker_objs)} tickers'
    )

    # Drop rows where essential info might be missing (e.g., headline)
    articles_df.dropna(subset=['headline'], inplace=True)

    try:
        dbm.insert_articles(articles_df, has_sentiment=False)
    except Exception as e:
        logger.error(f'Error inserting articles into database: {e}')


def compute_and_update_sentiment(n: int = 200):
    """
    Fetch the latest N articles without sentiment scores from the database and compute their sentiment scores.
    Then, update the database with the computed sentiment scores.
    """
    # get 200 latest articles without sentiment score from the database
    dbm: DatabaseManager = DatabaseManager()
    articles_df: pd.DataFrame = dbm.get_articles(n=n, has_sentiment=False, latest=True)
    if articles_df.empty:
        logger.warning('No articles without sentiment scores found in the database.')
        return
    logger.info(
        f'Fetched {articles_df.shape[0]} articles without sentiment scores from the database'
    )
    # perform sentiment analysis on them
    headlines: list[str] = articles_df['headline'].tolist()
    sentiment_scores = utils.analyse_sentiment(headlines)
    articles_df_with_sentiment = articles_df.merge(
        sentiment_scores, left_index=True, right_index=True, how='inner'
    )
    # update the database with the sentiment scores
    # we can use insert function since all the duplicate articles (ticker, headline) will be replaced
    # and the new sentiment scores will be added.
    dbm.insert_articles(articles_df_with_sentiment, has_sentiment=True)
    logger.success(
        f'Updated database with sentiment scores for {articles_df_with_sentiment.shape[0]} articles.'
    )

def aggregate_and_push():
    dbm = DatabaseManager() 
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)

    date_24h = (now - timedelta(days=1)).strftime('%Y-%m-%d')
    date_3d = (now - timedelta(days=3)).strftime('%Y-%m-%d')
    date_7d = (now - timedelta(days=7)).strftime('%Y-%m-%d')
    date_1m = (now - timedelta(days=30)).strftime('%Y-%m-%d')

    def get_agg_df(date):
        df = dbm.get_articles(n=10000, has_sentiment=True, after_date=date)

        if df.empty:
            return df

        agg_df = (
            df.groupby("ticker")["compound_sentiment"]   # ✅ MUST BE THIS
            .mean()
            .reset_index()
        )
    
        agg_df = agg_df.rename(columns={"compound_sentiment": "sentiment_score"})
    
        return agg_df

    df_24h = get_agg_df(date_24h)
    df_3d = get_agg_df(date_3d)
    df_7d = get_agg_df(date_7d)
    df_1m = get_agg_df(date_1m)

    # ✅ push only if data exists
    if not df_24h.empty:
        push_to_sheet(df_24h, "24H_Data")

    if not df_3d.empty:
        push_to_sheet(df_3d, "3D_Data")

    if not df_7d.empty:
        push_to_sheet(df_7d, "7D_Data")

    if not df_1m.empty:
        push_to_sheet(df_1m, "1M_Data")

    print("✅ Sheets updated!")

if __name__ == '__main__':
    # Example usage: Fetch data for Nifty 50.
    # Choose the universe: "nifty_50", "nifty_100", "nifty_200", "nifty_500"
    universe: str = 'nifty_50'
    multiprocess: bool = True
    # Call the function to fetch news
    get_news(universe, multiprocess)
    compute_and_update_sentiment()
    aggregate_and_push()  

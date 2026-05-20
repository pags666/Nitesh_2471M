from __future__ import annotations

import random
import threading
import time
from datetime import datetime, timedelta
from typing import Any

import httpx
import pandas as pd
from curl_cffi import requests
from dateutil.relativedelta import relativedelta
from loguru import logger
from nse import NSE
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CAPTCHA_INDICATORS,
    DB_UTILS,
    HEADER,
    SENTIMENT_MODEL_NAME,
    SOURCE_CONFIG,
    USER_AGENTS,
)
from database import DatabaseManager

# --- Reliability Infrastructure ---


class DomainRateLimiter:
    """Thread-safe per-domain rate limiter to avoid overwhelming any single server."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._domain_timestamps: dict[str, float] = {}
                    cls._instance._domain_lock = threading.Lock()
        return cls._instance

    def wait_for_domain(
        self, domain: str, min_delay: float = 0.3, max_delay: float = 1.0
    ) -> None:
        """Wait if needed to respect rate limits for a specific domain."""
        with self._domain_lock:
            now = time.time()
            last_request = self._domain_timestamps.get(domain, 0)
            elapsed = now - last_request
            required_delay = random.uniform(min_delay, max_delay)  # nosec B311
            if elapsed < required_delay:
                sleep_time = required_delay - elapsed
                time.sleep(sleep_time)
            self._domain_timestamps[domain] = time.time()


_rate_limiter = DomainRateLimiter()


def get_random_user_agent() -> str:
    """Return a random User-Agent string from the configured list."""
    return random.choice(USER_AGENTS)  # nosec B311


def _extract_domain(url: str) -> str:
    """Extract the domain from a URL for rate limiting purposes."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        return parsed.netloc or 'unknown'
    except Exception:
        return 'unknown'


def validate_response(content: str, url: str = '') -> bool:
    """Check if a response contains real content vs. CAPTCHA/block page."""
    if not content or len(content) < 200:
        logger.warning(f'Response too short ({len(content)} chars) for {url}')
        return False
    content_lower = content.lower()
    for indicator in CAPTCHA_INDICATORS:
        # Only flag if the indicator appears prominently (in the first 5000 chars)
        if indicator in content_lower[:5000]:
            logger.warning(
                f'Possible CAPTCHA/block detected for {url} (matched: {indicator})'
            )
            return False
    return True


def get_webpage_content(
    url: str,
    custom_header: bool = True,
    impersonate: bool = False,
    max_retries: int = 3,
    source_name: str = '',
) -> str:
    """
    Fetches the content of a webpage given its URL with exponential backoff for rate limiting.

    Args:
        url (str): The URL of the webpage to fetch.
        custom_header (bool): If True, uses a custom header for the request.
        impersonate (bool): If True, uses curl_cffi to impersonate a browser.
        max_retries (int): Maximum number of retry attempts for 429 responses.
        source_name (str): Name of the news source (for per-source rate limiting).

    Returns:
        str: The content of the webpage.
    """
    # Per-domain rate limiting
    domain = _extract_domain(url)
    source_cfg = SOURCE_CONFIG.get(source_name, {})
    min_delay = source_cfg.get('min_delay', 0.3)
    max_delay = source_cfg.get('max_delay', 0.8)
    timeout = source_cfg.get('timeout', 15)
    _rate_limiter.wait_for_domain(domain, min_delay, max_delay)

    # Build headers with rotating User-Agent
    headers = HEADER.copy()
    headers['User-Agent'] = get_random_user_agent()

    for attempt in range(max_retries + 1):
        try:
            if impersonate:
                response = requests.get(url, impersonate='chrome', timeout=timeout)
                response.raise_for_status()
                content = response.text
                if not validate_response(content, url):
                    logger.warning(f'Invalid response from {url} (impersonate mode)')
                    return ''
                return content

            response = (
                httpx.get(url, headers=headers, follow_redirects=True, timeout=timeout)
                if custom_header
                else httpx.get(url, follow_redirects=True, timeout=timeout)
            )

            # Handle rate limiting with retry
            if response.status_code == 429:
                if attempt < max_retries:
                    retry_after = int(
                        response.headers.get('Retry-After', 2 ** (attempt + 1))
                    )
                    wait_time = min(retry_after, 2 ** (attempt + 1) * 2)
                    logger.warning(
                        f'Rate limited (429) for {url}. Retrying in {wait_time}s '
                        f'(attempt {attempt + 1}/{max_retries})'
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        f'Rate limited (429) for {url}. Max retries ({max_retries}) exceeded'
                    )
                    return ''

            response.raise_for_status()
            content = response.text
            if not validate_response(content, url):
                logger.warning(f'Invalid response from {url}')
                return ''
            return content

        except Exception as e:
            # Handle curl_cffi and other exceptions with retry for 429
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                if e.response.status_code == 429 and attempt < max_retries:
                    retry_after = int(
                        getattr(e.response, 'headers', {}).get(
                            'Retry-After', 2 ** (attempt + 1)
                        )
                    )
                    wait_time = min(retry_after, 2 ** (attempt + 1) * 2)
                    logger.warning(
                        f'Rate limited (429) for {url}. Retrying in {wait_time}s '
                        f'(attempt {attempt + 1}/{max_retries})'
                    )
                    time.sleep(wait_time)
                    continue
                elif e.response.status_code == 429:
                    logger.error(
                        f'Rate limited (429) for {url}. Max retries ({max_retries}) exceeded'
                    )
                    return ''
                logger.warning(f'HTTP error {e.response.status_code} for URL: {url}')
                return ''
            logger.warning(f'Error fetching {url}: {e}')
            if attempt < max_retries:
                time.sleep(2**attempt)
                continue
            return ''

    return ''


def fetch_metadata(ticker: str):
    """
    Fetches metadata for a given ticker.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        list: A list containing metadata fields: [ticker, sector, industry, market_cap (in billions), company_name].
              Returns None for fields if data is unavailable.
    """
    # Fetch quote data from NSE
    logger.debug(f'Fetching metadata for {ticker}')
    with NSE('./') as nse:
        meta: dict[str, Any] = nse.quote(ticker)

    # Extract metadata fields
    try:
        sector: str = meta['industryInfo']['macro']
        industry: str = meta['industryInfo']['industry']
        previousClose: float = meta['priceInfo']['previousClose']
        issuedSize: int = meta['securityInfo']['issuedSize']
        # Calculate market capitalization in billions
        mCap: float = round(previousClose * issuedSize / 1e9, 2)
        companyName: str = meta['info']['companyName']

    except KeyError as e:
        logger.warning(f'KeyError: {e} for ticker {ticker}')
        return None

    return [ticker, sector, industry, mCap, companyName]


def parse_date(
    date_string: str, relative: bool = True, format: str | None = None
) -> str:
    """google news contains date info in relative format. This method parses the relative date and turns into absolute dates.

    Parameters
    ----------
    date_string : str
        article date in relative terms
    relative : bool
        indicates if the date_string is in relative format

    Returns
    -------
    datetime
        datetime object
    """
    now: datetime = datetime.now()
    if relative:
        parts: list[str] = date_string.split()

        datetime_object: datetime

        if len(parts) != 2 and len(parts) != 3:
            logger.warning(f'Invalid relative date format: {date_string}')
            return ''

        value: int = int(parts[0]) if parts[0] not in ['a', 'last'] else 1
        unit: str = parts[1]

        if unit.startswith('minute'):
            datetime_object = now - timedelta(minutes=value)
        elif unit.startswith('hour'):
            datetime_object = now - timedelta(hours=value)
        elif unit.startswith('day'):
            datetime_object = now - timedelta(days=value)
        elif unit.startswith('week'):
            datetime_object = now - timedelta(weeks=value)
        elif unit.startswith('month'):
            datetime_object = now - relativedelta(months=value)
        elif unit.startswith('year'):
            datetime_object = now - relativedelta(years=value)
        elif unit.startswith('yesterday'):
            datetime_object = now - timedelta(days=1)
        elif unit.startswith('today'):
            datetime_object = now
        else:
            logger.warning(f'Unknown date format: {date_string}')
            return ''
    else:
        if not format:
            logger.error('Format string is required for absolute date parsing.')
            return ''
        elif format.__contains__('%Y'):
            datetime_object: datetime = datetime.strptime(date_string, format)
        else:
            try:
                datetime_object = datetime.strptime(date_string, format).replace(
                    year=2025
                )
                datetime_object = (
                    datetime_object
                    if datetime_object < now
                    else datetime_object.replace(year=datetime_object.year - 1)
                )

            except Exception as e:
                logger.warning(f"Error parsing date '{date_string}': {e}")
                return ''

    if (now - datetime_object).days > 30:
        logger.warning(f'Stale date parsed: {date_string} -> {datetime_object}')

    # Format the datetime object to a string
    datetime_format: str = '%Y-%m-%d %H:%M:%S'
    formatted_date: str = datetime_object.strftime(datetime_format)
    return formatted_date


def get_relative_date(option: str) -> str:
    now = datetime.now()

    if option == 'Past 24 Hours':
        return (now - timedelta(days=1)).strftime('%Y-%m-%d')
    elif option == 'Past 3 Days':
        return (now - timedelta(days=3)).strftime('%Y-%m-%d')
    elif option == 'Past 7 Days':
        return (now - timedelta(days=7)).strftime('%Y-%m-%d')
    elif option == 'Past 1 Month':
        return (now - timedelta(days=30)).strftime('%Y-%m-%d')

    # default fallback
    return (now - timedelta(days=1)).strftime('%Y-%m-%d')


def analyse_sentiment(headlines: list[str]) -> pd.DataFrame:
    """
    Perform Sentiment Analysis using finBERT model. Create a dataframe from the results.

    Parameters
    ----------
    headline : list[str]
        list of article headlines

    Returns
    -------
    pd.DataFrame
        returns sentiment scores in a df with following columns:
        Positive, Negative, Neutral, compound
    """

    from transformers.models.bert import BertForSequenceClassification, BertTokenizer
    from transformers.pipelines import pipeline

    finbert_1: BertForSequenceClassification = (
        BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=SENTIMENT_MODEL_NAME,
            num_labels=3,
            use_safetensors=True,  # Use safe tensors
        )
    )

    tokenizer_1 = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=SENTIMENT_MODEL_NAME
    )

    # set top_k=1 to get the most likely label or top_k=None to get all labels
    # device=-1 means CPU
    nlp_1 = pipeline(
        'sentiment-analysis',
        model=finbert_1,
        tokenizer=tokenizer_1,
        device=-1,
        top_k=None,
        framework='pt',
    )

    try:
        results: list[list[dict[str, str | float]]] = nlp_1(
            headlines, batch_size=BATCH_SIZE
        )
    except Exception as e:
        logger.warning(f'Error: {e}')
        return pd.DataFrame()

    # Check if the results are empty or contain only one result
    if len(results) != len(headlines):
        logger.warning(
            f'Sentiment analysis returned {len(results)} results for {len(headlines)} headlines.'
        )
        return pd.DataFrame()

    logger.debug(f'Articles for which Sentiment Score is available: {len(results)}')

    # Initialize an empty list to hold the flattened data
    # we will transform a list of list of dictionaries into a list of dictionaries
    flattened_data: list[dict[str, float]] = []

    for news_item_sentiment_list in tqdm(iterable=results, desc='Processing Sentiment'):
        news_item_sentiment_dict = {}
        for individual_label_dict in news_item_sentiment_list:
            news_item_sentiment_dict[individual_label_dict['label']] = (
                individual_label_dict['score']
            )
        flattened_data.append(news_item_sentiment_dict)

    # Create the DataFrame
    df = pd.DataFrame(flattened_data)
    df.fillna(0, inplace=True)  # Fill NaN values with 0
    logger.debug(f'DataFrame: {df}')

    # Calculate the compound score
    df.loc[:, 'compound'] = (
        df.loc[:, 'Positive']
        .where(df['Positive'] > df['Negative'], -df['Negative'])
        .astype(float)
        .round(4)
    )
    df.loc[:, 'compound'] = df.loc[:, 'compound'].fillna(0)
    df.loc[:, 'compound'] = df.loc[:, 'compound'].clip(lower=-1, upper=1)
    return df


# ------------------ database utility functions ---------------------------


def query_duplicates(return_df: bool = False) -> pd.DataFrame | None:
    with DatabaseManager().get_connection() as conn:
        # Select duplicates
        duplicates_df: pd.DataFrame = conn.execute(
            DB_UTILS['query_duplicates']
        ).fetchdf()
        duplicate_rows_count: int = duplicates_df.shape[0]

        if duplicate_rows_count == 0:
            logger.info('No duplicates found in database')
            return None

        logger.info(f'{duplicate_rows_count} duplicates found in database')

        if return_df:
            return duplicates_df


def deduplicate_db() -> None:
    duplicates_df: pd.DataFrame | None = query_duplicates(return_df=True)

    if duplicates_df is None:
        logger.info('No duplicates found to delete.')
        return

    duplicates_count: int = duplicates_df.shape[0]

    if duplicates_count == 0:
        logger.info('No duplicates found to delete.')
        return

    with DatabaseManager().get_connection() as conn:
        conn.execute(DB_UTILS['delete_duplicates'])
        logger.success(f'Deleted {duplicates_count} duplicate rows from database.')


if __name__ == '__main__':
    # Example usage
    # url = "https://www.example.com"
    # content = get_webpage_content(url)
    # print(content)

    # test_headlines = [
    #     'ADANIENT Publishes news about quarterly results',
    #     'Equity markets are down',
    #     'Market volatility increases',
    # ]
    # sentiment_results = analyse_sentiment(test_headlines)
    # logger.debug(f'Sentiment Results: {sentiment_results}')
    query_duplicates(return_df=False)

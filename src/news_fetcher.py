from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import final
from urllib.parse import quote_plus, urljoin

from bs4 import BeautifulSoup, Tag
from loguru import logger
from typing_extensions import override

from config import (
    GENERIC_HEADLINE_PATTERNS,
    MAX_ARTICLE_AGE_DAYS,
    MIN_HEADLINE_LENGTH,
    SOURCE_SUFFIXES_TO_STRIP,
)
from utils import get_webpage_content, parse_date

# --- Health Monitoring ---


@dataclass
class FetchHealthReport:
    """Tracks the health of a single fetch operation for monitoring."""

    source_name: str
    ticker: str
    direct_count: int = 0
    fallback_count: int = 0
    error: str | None = None
    method_used: str = 'none'  # 'direct', 'fallback', 'both', 'none'


# --- Headline Cleaning & Quality ---


def clean_headline(headline: str) -> str:
    """Remove source attribution suffixes and normalize whitespace in headlines."""
    cleaned = headline.strip()
    for suffix in SOURCE_SUFFIXES_TO_STRIP:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


def is_article_fresh(date_posted: str) -> bool:
    """Check if an article's date is within the acceptable age range."""
    if not date_posted:
        return (
            True  # Allow articles with missing dates (better to include than exclude)
        )
    try:
        article_date = datetime.strptime(date_posted, '%Y-%m-%d %H:%M:%S')
        age = (datetime.now() - article_date).days
        if age > MAX_ARTICLE_AGE_DAYS:
            return False
        if age < 0:  # Future date
            logger.warning(f'Article has future date: {date_posted}')
            return False
    except ValueError:
        pass  # If we can't parse, allow the article
    return True


def is_headline_quality(headline: str) -> bool:
    """
    Check if a headline meets quality standards for reliable sentiment analysis.
    Rejects:
    - Headlines shorter than MIN_HEADLINE_LENGTH (too short for FinBERT)
    - Generic market roundup / listicle headlines (not company-specific)
    """
    if not headline or len(headline) < MIN_HEADLINE_LENGTH:
        return False

    headline_lower = headline.lower()
    for pattern in GENERIC_HEADLINE_PATTERNS:
        if pattern in headline_lower:
            return False

    return True


# --- Base Classes ---


class NewsSource(ABC):
    @abstractmethod
    def get_articles(self, ticker: str) -> list[dict[str, str]]:
        """
        make http request to the news source, parse the response, and return a list of articles
        """
        pass


class GoogleNewsSiteSource(NewsSource):
    """Fetches ticker-specific news via Google News RSS with a site: filter."""

    def __init__(self, site_domain: str, source_name: str) -> None:
        self.base_url = 'https://news.google.com/rss/search'
        self.site_domain = site_domain
        self.source_name = source_name
        self.articles: list[dict[str, str]] = []

    @staticmethod
    def _parse_rss_date(value: str) -> str:
        if not value:
            return ''
        try:
            parsed = parsedate_to_datetime(value)
            if parsed is None:
                logger.warning(f'Invalid RSS date: {value}')
                return ''
            age_days = (datetime.now(parsed.tzinfo) - parsed).days
            if age_days > MAX_ARTICLE_AGE_DAYS:
                logger.debug(f'Skipping stale RSS article ({age_days}d old): {value}')
                return ''  # Return empty so the article is skipped
            return parsed.strftime('%Y-%m-%d %H:%M:%S')
        except (TypeError, ValueError) as exc:
            logger.warning(f'Invalid RSS date: {value} ({exc})')
            return ''

    def get_articles(self, ticker: str) -> list[dict[str, str]]:
        query = f'{ticker} site:{self.site_domain}'
        url = f'{self.base_url}?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en'
        response = get_webpage_content(
            url, custom_header=False, source_name='GoogleNewsRSS'
        )
        if not response:
            logger.warning(f'No response from Google News RSS for {ticker}')
            return self.articles

        soup = BeautifulSoup(response, 'xml')
        for item in soup.find_all('item'):
            title_tag = item.find('title')
            link_tag = item.find('link')
            pub_date_tag = item.find('pubDate')

            title = title_tag.text.strip() if title_tag else ''
            link = link_tag.text.strip() if link_tag else ''
            pub_date = pub_date_tag.text.strip() if pub_date_tag else ''

            if not title or not link or not pub_date:
                logger.warning(f'Missing RSS fields for {self.source_name} {ticker}')
                continue

            date_posted = self._parse_rss_date(pub_date)
            if not date_posted:
                logger.warning(
                    f'Empty date_posted from {self.source_name} for {ticker}: {pub_date}'
                )
                continue

            if not is_article_fresh(date_posted):
                continue

            self.articles.append(
                {
                    'ticker': ticker,
                    'headline': clean_headline(title),
                    'date_posted': date_posted,
                    'article_link': link,
                    'source': self.source_name,
                }
            )

        return self.articles


class DualStrategySource(NewsSource):
    """
    Base class for sources that try direct scraping first, then fall back to
    Google News RSS. Subclasses implement _fetch_direct() for site-specific logic.
    """

    def __init__(self, site_domain: str, source_name: str) -> None:
        self.site_domain = site_domain
        self.source_name = source_name
        self.articles: list[dict[str, str]] = []
        self.health_report: FetchHealthReport | None = None

    @abstractmethod
    def _fetch_direct(self, ticker: str) -> list[dict[str, str]]:
        """Fetch articles directly from the source website. Return a list of article dicts."""
        pass

    def _fetch_rss_fallback(self, ticker: str) -> list[dict[str, str]]:
        """Fallback: fetch via Google News RSS site: filter."""
        rss_source = GoogleNewsSiteSource(self.site_domain, self.source_name)
        return rss_source.get_articles(ticker)

    @override
    def get_articles(self, ticker: str) -> list[dict[str, str]]:
        report = FetchHealthReport(source_name=self.source_name, ticker=ticker)

        # Try direct scraping first
        try:
            direct_articles = self._fetch_direct(ticker)
            report.direct_count = len(direct_articles)
            if direct_articles:
                self.articles.extend(direct_articles)
                report.method_used = 'direct'
        except Exception as e:
            report.error = f'Direct fetch error: {e}'
            logger.warning(f'Direct fetch failed for {self.source_name}/{ticker}: {e}')

        # Fall back to RSS if direct returned nothing
        if report.direct_count == 0:
            try:
                rss_articles = self._fetch_rss_fallback(ticker)
                report.fallback_count = len(rss_articles)
                if rss_articles:
                    self.articles.extend(rss_articles)
                    report.method_used = (
                        'both' if report.direct_count > 0 else 'fallback'
                    )
            except Exception as e:
                if report.error:
                    report.error += f'; RSS fallback error: {e}'
                else:
                    report.error = f'RSS fallback error: {e}'
                logger.warning(
                    f'RSS fallback failed for {self.source_name}/{ticker}: {e}'
                )

        if not self.articles:
            report.method_used = 'none'

        self.health_report = report
        total = report.direct_count + report.fallback_count
        if total > 0:
            logger.debug(
                f'{self.source_name}/{ticker}: {total} articles '
                f'(direct={report.direct_count}, fallback={report.fallback_count})'
            )
        return self.articles


# --- Existing Sources (unchanged logic) ---


@final
class GoogleFinanceSource(NewsSource):
    def __init__(self):
        self.base_url: str = 'https://www.google.com/finance/quote'
        self.articles: list[dict[str, str]] = []
        self.article_selector: str = 'div.z4rs2b'
        self.headline_selectors: list[str] = [
            'div.Yfwt5',
            'div.Yfwt5 span',
            'div.tC2Wod',
        ]
        self.date_selectors: list[str] = [
            'div.Adak',
            'div.Adak span',
            'time',
        ]
        self.source_selectors: list[str] = [
            'div.sfyJob',
            'div.AaVjTc',
        ]
        self.link_selectors: list[str] = [
            'a',
        ]

    @staticmethod
    def _select_text(article: Tag, selectors: list[str]) -> str:
        for selector in selectors:
            tag: Tag | None = article.select_one(selector)
            if tag and tag.text:
                text = tag.text.strip().replace('\n', '')
                if text:
                    return text
        return ''

    @staticmethod
    def _select_link(article: Tag, selectors: list[str]) -> str:
        for selector in selectors:
            tag: Tag | None = article.select_one(selector)
            if not tag:
                continue
            href = tag.get('href')
            if href:
                return str(href)
        return ''

    @override
    def get_articles(self, ticker: str) -> list[dict[str, str]]:
        try:
            url = f'{self.base_url}/{ticker}:NSE'
            response = get_webpage_content(url, source_name='GoogleFinance')
            if not response:
                logger.warning(f'No response from Google Finance for {ticker}')
                return self.articles
            soup = BeautifulSoup(response, 'html.parser')
            article_elements = soup.select(self.article_selector)

            for article in article_elements:
                try:
                    headline: str = self._select_text(article, self.headline_selectors)
                    relative_date_str: str = self._select_text(
                        article, self.date_selectors
                    )
                    source: str = self._select_text(article, self.source_selectors)
                    article_link: str = self._select_link(article, self.link_selectors)

                    if not all([headline, relative_date_str, source, article_link]):
                        logger.warning(
                            f'Missing elements in Google Finance article for {ticker}'
                        )
                        continue

                    if not article_link:
                        logger.warning(
                            f'Missing article link in Google Finance article for {ticker}'
                        )
                        continue

                    date_posted: str | None = parse_date(relative_date_str)
                    if not date_posted:
                        logger.warning(
                            f'Empty date_posted from Google Finance for {ticker}: {relative_date_str}'
                        )

                    if date_posted and not is_article_fresh(date_posted):
                        continue

                    self.articles.append(
                        {
                            'ticker': ticker,
                            'headline': clean_headline(headline),
                            'date_posted': date_posted,
                            'article_link': article_link,
                            'source': source,
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f'Error parsing Google Finance article for {ticker}: {str(e)}'
                    )
                    continue

        except Exception as e:
            logger.error(f'Error fetching from Google Finance for {ticker}: {str(e)}')
        return self.articles


@final
class YahooFinanceSource(NewsSource):
    def __init__(self):
        self.base_url = 'https://finance.yahoo.com/quote'
        self.articles: list[dict[str, str]] = []
        self.article_selectors: list[str] = [
            'li.stream-item.story-item',
            'li.js-stream-content',
            'li[data-test-locator="stream-item"]',
        ]
        self.headline_selectors: list[str] = [
            'a h3',
            'h3',
            'a[data-test-locator="stream-item-title"]',
        ]
        self.footer_selectors: list[str] = [
            'div.publishing',
            'div.publishing.yf-1weyqlp',
            'div.caas-attr-time-style',
        ]
        self.link_selectors: list[str] = [
            'a',
        ]

    @staticmethod
    def _select_tag(article: Tag, selectors: list[str]) -> Tag | None:
        for selector in selectors:
            tag: Tag | None = article.select_one(selector)
            if tag:
                return tag
        return None

    @staticmethod
    def _select_text(article: Tag, selectors: list[str]) -> str:
        tag = YahooFinanceSource._select_tag(article, selectors)
        if tag and tag.text:
            return tag.text.strip()
        return ''

    @staticmethod
    def _select_link(article: Tag, selectors: list[str]) -> str:
        tag = YahooFinanceSource._select_tag(article, selectors)
        if not tag:
            return ''
        href = tag.get('href')
        return str(href) if href else ''

    @staticmethod
    def _parse_iso_datetime(value: str) -> str:
        try:
            sanitized = value.replace('Z', '+00:00')
            parsed = datetime.fromisoformat(sanitized)
            return parsed.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            logger.warning(f'Invalid ISO datetime: {value}')
            return ''

    @override
    def get_articles(self, ticker: str) -> list[dict[str, str]]:
        try:
            url = f'{self.base_url}/{ticker}.NS/news/'
            response = get_webpage_content(
                url, impersonate=True, source_name='YahooFinance'
            )
            if not response:
                logger.warning(f'No response from Yahoo Finance for {ticker}')
                return self.articles

            soup = BeautifulSoup(response, 'html.parser')
            article_elements = soup.select(', '.join(self.article_selectors))

            for article in article_elements:
                try:
                    article_link_raw = self._select_link(article, self.link_selectors)
                    headline: str = self._select_text(article, self.headline_selectors)
                    footer_tag: Tag | None = self._select_tag(
                        article, self.footer_selectors
                    )

                    if not article_link_raw or not headline:  # Footer is optional
                        logger.warning(
                            f'Missing link or headline in Yahoo Finance article for {ticker}'
                        )
                        continue

                    article_link: str = (
                        str(article_link_raw) if article_link_raw else ''
                    )
                    if not article_link:
                        logger.warning(
                            f'Missing article link in Yahoo Finance article for {ticker}'
                        )
                        continue

                    # Make sure we have a full URL
                    if not article_link.startswith('http'):
                        article_link = 'https://finance.yahoo.com' + article_link

                    # Get publisher and date from the footer
                    source = 'Yahoo Finance'  # Default source
                    time_str = ''
                    if footer_tag:
                        footer_text = footer_tag.text.strip()
                        parts = footer_text.split('•')
                        source = parts[0].strip() if len(parts) > 0 else 'Yahoo Finance'
                        time_str = parts[1].strip() if len(parts) > 1 else ''
                    iso_time = ''
                    if not time_str:
                        time_tag: Tag | None = None
                        if footer_tag:
                            time_tag = footer_tag.select_one('time')
                        if not time_tag:
                            time_tag = article.select_one('time')
                        if time_tag:
                            iso_time = str(time_tag.get('datetime') or '')

                    date_posted = (
                        parse_date(time_str)
                        if time_str
                        else self._parse_iso_datetime(iso_time)
                    )
                    if not date_posted:
                        logger.warning(
                            f'Empty date_posted from Yahoo Finance for {ticker}: {time_str or iso_time}'
                        )

                    if date_posted and not is_article_fresh(date_posted):
                        continue

                    data_dict = {
                        'ticker': ticker,
                        'headline': clean_headline(headline),
                        'date_posted': date_posted,
                        'article_link': article_link,
                        'source': source,
                    }
                    self.articles.append(data_dict)
                except Exception as e:
                    logger.warning(
                        f'Error parsing Yahoo Finance article for {ticker}: {str(e)}'
                    )
                    continue

        except Exception as e:
            logger.error(f'Error fetching from Yahoo Finance for {ticker}: {str(e)}')
        return self.articles


@final
class FinologySource(NewsSource):
    def __init__(self):
        self.base_url = 'https://ticker.finology.in/company'
        self.articles: list[dict[str, str]] = []
        self.article_selector: str = 'div#newsarticles a#btnDetails.newslink'
        self.headline_selectors: list[str] = [
            'span',
            'div.news-left span',
        ]
        self.date_selectors: list[str] = [
            'small',
            'div.news-left small',
        ]

    @staticmethod
    def _select_text(article: Tag, selectors: list[str]) -> str:
        for selector in selectors:
            tag: Tag | None = article.select_one(selector)
            if tag and tag.text:
                return tag.text.strip()
        return ''

    @override
    def get_articles(self, ticker: str) -> list[dict[str, str]]:
        try:
            url = f'{self.base_url}/{ticker}'
            response = get_webpage_content(
                url, custom_header=True, impersonate=True, source_name='Finology'
            )
            if not response:
                logger.warning(f'No response from Finology for {ticker}')
                return self.articles

            soup = BeautifulSoup(response, 'html.parser')
            article_elements = soup.select(self.article_selector)

            for article in article_elements:
                try:
                    headline = self._select_text(article, self.headline_selectors)
                    date_str = self._select_text(article, self.date_selectors)

                    if not headline or not date_str:
                        logger.warning(
                            f'Missing elements in Finology article for {ticker}'
                        )
                        continue

                    date_posted = parse_date(
                        date_str, relative=False, format='%d %b, %I:%M %p'
                    )
                    if not date_posted:
                        logger.warning(
                            f'Empty date_posted from Finology for {ticker}: {date_str}'
                        )

                    if date_posted and not is_article_fresh(date_posted):
                        continue

                    self.articles.append(
                        {
                            'ticker': ticker,
                            'headline': clean_headline(headline),
                            'date_posted': date_posted,
                            'article_link': url,  # Finology links point back to the main page
                            'source': 'Finology',
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f'Error parsing Finology article for {ticker}: {str(e)}'
                    )
                    continue

        except Exception as e:
            logger.error(f'Error fetching from Finology for {ticker}: {str(e)}')
        return self.articles


# --- Direct Scraper Sources (Dual Strategy: Direct + RSS Fallback) ---


@final
class MoneycontrolSource(DualStrategySource):
    """
    Fetches news from Moneycontrol via direct search page scraping,
    with Google News RSS fallback.
    """

    def __init__(self) -> None:
        super().__init__('moneycontrol.com', 'Moneycontrol')
        self.search_url = (
            'https://www.moneycontrol.com/stocks/cptmarket/compsearchnew/searchBox'
        )

    @override
    def _fetch_direct(self, ticker: str) -> list[dict[str, str]]:
        articles: list[dict[str, str]] = []

        # Step 1: Search for the ticker to find the company page
        search_url = f'{self.search_url}?search_query={quote_plus(ticker)}&search_str={quote_plus(ticker)}'
        response = get_webpage_content(
            search_url, impersonate=True, source_name='Moneycontrol'
        )
        if not response:
            return articles

        soup = BeautifulSoup(response, 'html.parser')

        # Try to find news links on the search results page
        # Moneycontrol search returns a list with company links
        news_links = soup.select('a[href*="/news/"], a[href*="/article/"]')
        if not news_links:
            # Try alternative: look for company page link and navigate to news tab
            company_links = soup.select('a[href*="/stocks/company_info/"]')
            if company_links:
                company_url = company_links[0].get('href', '')
                if company_url and not company_url.startswith('http'):
                    company_url = urljoin('https://www.moneycontrol.com', company_url)
                if company_url:
                    return self._fetch_from_company_page(ticker, company_url)
            return articles

        for link_tag in news_links[:15]:  # Limit to 15 articles
            href = link_tag.get('href', '')
            title = link_tag.get_text(strip=True)
            if not href or not title or len(title) < 10:
                continue
            if not href.startswith('http'):
                href = urljoin('https://www.moneycontrol.com', href)

            articles.append(
                {
                    'ticker': ticker,
                    'headline': clean_headline(title),
                    'date_posted': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'article_link': href,
                    'source': 'Moneycontrol',
                }
            )

        return articles

    def _fetch_from_company_page(
        self, ticker: str, company_url: str
    ) -> list[dict[str, str]]:
        """Fetch news from a Moneycontrol company-specific page."""
        articles: list[dict[str, str]] = []
        response = get_webpage_content(
            company_url, impersonate=True, source_name='Moneycontrol'
        )
        if not response:
            return articles

        soup = BeautifulSoup(response, 'html.parser')
        # Look for news section on company page
        news_items = soup.select(
            'div.news_sec a, div.bsr_wleft a, a.arial11_blue, div.MT15 a, li.clearfix a'
        )

        for item in news_items[:15]:
            title = item.get_text(strip=True)
            href = item.get('href', '')
            if not title or len(title) < 10 or not href:
                continue
            if not href.startswith('http'):
                href = urljoin('https://www.moneycontrol.com', href)

            # Try to find date near the link
            parent = item.parent
            date_posted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if parent:
                date_tag = parent.select_one('span.date, span.gray10, time, small')
                if date_tag:
                    date_text = date_tag.get_text(strip=True)
                    parsed = parse_date(date_text)
                    if parsed:
                        date_posted = parsed

            articles.append(
                {
                    'ticker': ticker,
                    'headline': clean_headline(title),
                    'date_posted': date_posted,
                    'article_link': href,
                    'source': 'Moneycontrol',
                }
            )

        return articles


@final
class EconomicTimesMarketsSource(DualStrategySource):
    """
    Fetches news from Economic Times Markets via direct topic page scraping,
    with Google News RSS fallback.
    """

    def __init__(self) -> None:
        super().__init__('economictimes.indiatimes.com', 'Economic Times Markets')

    @override
    def _fetch_direct(self, ticker: str) -> list[dict[str, str]]:
        articles: list[dict[str, str]] = []
        # Economic Times topic page (server-rendered)
        topic_url = f'https://economictimes.indiatimes.com/topic/{quote_plus(ticker)}'
        response = get_webpage_content(
            topic_url, impersonate=True, source_name='EconomicTimesMarkets'
        )
        if not response:
            return articles

        soup = BeautifulSoup(response, 'html.parser')

        # ET topic pages have articles in various container formats
        article_selectors = [
            'div.clr.flt.topicstry',
            'div.eachStory',
            'div.content_wrapper',
            'li.article-list',
            'div.story-box',
        ]

        article_elements = soup.select(', '.join(article_selectors))

        if not article_elements:
            # Fallback: try general link-based extraction
            links = soup.select(
                'a[href*="/markets/"], a[href*="/stocks/"], a[href*="/industry/"]'
            )
            for link in links[:15]:
                title = link.get_text(strip=True)
                href = link.get('href', '')
                if not title or len(title) < 15 or not href:
                    continue
                if not href.startswith('http'):
                    href = urljoin('https://economictimes.indiatimes.com', href)
                articles.append(
                    {
                        'ticker': ticker,
                        'headline': clean_headline(title),
                        'date_posted': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'article_link': href,
                        'source': 'Economic Times Markets',
                    }
                )
            return articles

        for article in article_elements[:15]:
            try:
                # Extract headline
                headline_tag = article.select_one('a h2, a h3, a.title, a')
                if not headline_tag:
                    continue
                headline = headline_tag.get_text(strip=True)
                if not headline or len(headline) < 10:
                    continue

                # Extract link
                link_tag = article.select_one('a[href]')
                href = link_tag.get('href', '') if link_tag else ''
                if not href:
                    continue
                if not href.startswith('http'):
                    href = urljoin('https://economictimes.indiatimes.com', href)

                # Extract date
                date_posted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                date_tag = article.select_one(
                    'time, span.date-format, span.time_cptn, '
                    'div.date-text, span.artDate'
                )
                if date_tag:
                    date_text = date_tag.get('datetime') or date_tag.get_text(
                        strip=True
                    )
                    if date_text:
                        parsed = parse_date(str(date_text))
                        if parsed:
                            date_posted = parsed

                articles.append(
                    {
                        'ticker': ticker,
                        'headline': clean_headline(headline),
                        'date_posted': date_posted,
                        'article_link': href,
                        'source': 'Economic Times Markets',
                    }
                )
            except Exception as e:
                logger.warning(f'Error parsing ET Markets article for {ticker}: {e}')
                continue

        return articles


@final
class BusinessStandardSource(DualStrategySource):
    """
    Fetches news from Business Standard via direct search page scraping,
    with Google News RSS fallback.
    """

    def __init__(self) -> None:
        super().__init__('business-standard.com', 'Business Standard')

    @override
    def _fetch_direct(self, ticker: str) -> list[dict[str, str]]:
        articles: list[dict[str, str]] = []
        search_url = (
            f'https://www.business-standard.com/search?type=news&q={quote_plus(ticker)}'
        )
        response = get_webpage_content(
            search_url, impersonate=True, source_name='BusinessStandard'
        )
        if not response:
            return articles

        soup = BeautifulSoup(response, 'html.parser')

        # BS search results page selectors
        result_selectors = [
            'div.listing-txt',
            'div.cardHolder',
            'div.search-result',
            'li.article-list',
            'div.main-card',
        ]
        result_elements = soup.select(', '.join(result_selectors))

        if not result_elements:
            # Fallback: extract from general links
            links = soup.select(
                'a[href*="/article/"], a[href*="/companies/"], a[href*="/markets/"]'
            )
            for link in links[:15]:
                title = link.get_text(strip=True)
                href = link.get('href', '')
                if not title or len(title) < 15 or not href:
                    continue
                if not href.startswith('http'):
                    href = urljoin('https://www.business-standard.com', href)
                articles.append(
                    {
                        'ticker': ticker,
                        'headline': clean_headline(title),
                        'date_posted': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'article_link': href,
                        'source': 'Business Standard',
                    }
                )
            return articles

        for result in result_elements[:15]:
            try:
                headline_tag = result.select_one('a h2, a h3, a.hdln, a')
                if not headline_tag:
                    continue
                headline = headline_tag.get_text(strip=True)
                if not headline or len(headline) < 10:
                    continue

                link_tag = result.select_one('a[href]')
                href = link_tag.get('href', '') if link_tag else ''
                if not href:
                    continue
                if not href.startswith('http'):
                    href = urljoin('https://www.business-standard.com', href)

                date_posted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                date_tag = result.select_one(
                    'time, span.date, div.posted-on, span.date-txt'
                )
                if date_tag:
                    date_text = date_tag.get('datetime') or date_tag.get_text(
                        strip=True
                    )
                    if date_text:
                        parsed = parse_date(str(date_text))
                        if parsed:
                            date_posted = parsed

                articles.append(
                    {
                        'ticker': ticker,
                        'headline': clean_headline(headline),
                        'date_posted': date_posted,
                        'article_link': href,
                        'source': 'Business Standard',
                    }
                )
            except Exception as e:
                logger.warning(
                    f'Error parsing Business Standard article for {ticker}: {e}'
                )
                continue

        return articles


@final
class CnbcTv18Source(DualStrategySource):
    """
    Fetches news from CNBC TV18 via direct tag page scraping,
    with Google News RSS fallback.
    """

    def __init__(self) -> None:
        super().__init__('cnbctv18.com', 'CNBC TV18')

    @override
    def _fetch_direct(self, ticker: str) -> list[dict[str, str]]:
        articles: list[dict[str, str]] = []
        # Try CNBC TV18 search page first, then tags page
        urls_to_try = [
            f'https://www.cnbctv18.com/search/?query={quote_plus(ticker)}',
            f'https://www.cnbctv18.com/tags/{quote_plus(ticker.lower())}.htm',
        ]
        response = None
        for url in urls_to_try:
            response = get_webpage_content(
                url, impersonate=True, source_name='CnbcTv18'
            )
            if response:
                break
        if not response:
            return articles

        soup = BeautifulSoup(response, 'html.parser')

        # CNBC TV18 tag page article selectors
        article_selectors = [
            'li.jsx-article',
            'div.listitem',
            'div.story-box',
            'div.card',
            'article',
        ]
        article_elements = soup.select(', '.join(article_selectors))

        if not article_elements:
            # Fallback: general link extraction
            links = soup.select(
                'a[href*="/market/"], a[href*="/stocks/"], '
                'a[href*="/economy/"], a[href*="/business/"]'
            )
            for link in links[:15]:
                title = link.get_text(strip=True)
                href = link.get('href', '')
                if not title or len(title) < 15 or not href:
                    continue
                if not href.startswith('http'):
                    href = urljoin('https://www.cnbctv18.com', href)
                articles.append(
                    {
                        'ticker': ticker,
                        'headline': clean_headline(title),
                        'date_posted': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'article_link': href,
                        'source': 'CNBC TV18',
                    }
                )
            return articles

        for article in article_elements[:15]:
            try:
                headline_tag = article.select_one('a h2, a h3, a.title, a')
                if not headline_tag:
                    continue
                headline = headline_tag.get_text(strip=True)
                if not headline or len(headline) < 10:
                    continue

                link_tag = article.select_one('a[href]')
                href = link_tag.get('href', '') if link_tag else ''
                if not href:
                    continue
                if not href.startswith('http'):
                    href = urljoin('https://www.cnbctv18.com', href)

                date_posted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                date_tag = article.select_one(
                    'time, span.date, span.posted-on, div.date'
                )
                if date_tag:
                    date_text = date_tag.get('datetime') or date_tag.get_text(
                        strip=True
                    )
                    if date_text:
                        parsed = parse_date(str(date_text))
                        if parsed:
                            date_posted = parsed

                articles.append(
                    {
                        'ticker': ticker,
                        'headline': clean_headline(headline),
                        'date_posted': date_posted,
                        'article_link': href,
                        'source': 'CNBC TV18',
                    }
                )
            except Exception as e:
                logger.warning(f'Error parsing CNBC TV18 article for {ticker}: {e}')
                continue

        return articles


@final
class ReutersSource(DualStrategySource):
    """
    Fetches news from Reuters via direct search page scraping,
    with Google News RSS fallback.

    Note: Reuters is heavily JavaScript-rendered, so direct scraping
    may have limited success. The RSS fallback is particularly important here.
    """

    def __init__(self) -> None:
        super().__init__('reuters.com', 'Reuters')

    @override
    def _fetch_direct(self, ticker: str) -> list[dict[str, str]]:
        articles: list[dict[str, str]] = []
        # Reuters wire search (limited success due to JS rendering)
        search_url = (
            f'https://www.reuters.com/site-search/?query='
            f'{quote_plus(ticker + " India")}&section=all'
        )
        response = get_webpage_content(
            search_url, impersonate=True, source_name='Reuters'
        )
        if not response:
            return articles

        soup = BeautifulSoup(response, 'html.parser')

        # Reuters search result selectors
        result_selectors = [
            'li.search-result__list-item',
            'div.search-result-indiv',
            'article.story',
            'div.media-story-card',
        ]
        result_elements = soup.select(', '.join(result_selectors))

        if not result_elements:
            # Fallback: try extracting from any article-like links
            links = soup.select(
                'a[href*="/business/"], a[href*="/markets/"], a[href*="/world/"]'
            )
            for link in links[:10]:
                title = link.get_text(strip=True)
                href = link.get('href', '')
                if not title or len(title) < 15 or not href:
                    continue
                if not href.startswith('http'):
                    href = urljoin('https://www.reuters.com', href)
                articles.append(
                    {
                        'ticker': ticker,
                        'headline': clean_headline(title),
                        'date_posted': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'article_link': href,
                        'source': 'Reuters',
                    }
                )
            return articles

        for result in result_elements[:10]:
            try:
                headline_tag = result.select_one('a h3, a span, a')
                if not headline_tag:
                    continue
                headline = headline_tag.get_text(strip=True)
                if not headline or len(headline) < 10:
                    continue

                link_tag = result.select_one('a[href]')
                href = link_tag.get('href', '') if link_tag else ''
                if not href:
                    continue
                if not href.startswith('http'):
                    href = urljoin('https://www.reuters.com', href)

                date_posted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                date_tag = result.select_one('time, span.date')
                if date_tag:
                    date_text = date_tag.get('datetime') or date_tag.get_text(
                        strip=True
                    )
                    if date_text:
                        parsed = parse_date(str(date_text))
                        if parsed:
                            date_posted = parsed

                articles.append(
                    {
                        'ticker': ticker,
                        'headline': clean_headline(headline),
                        'date_posted': date_posted,
                        'article_link': href,
                        'source': 'Reuters',
                    }
                )
            except Exception as e:
                logger.warning(f'Error parsing Reuters article for {ticker}: {e}')
                continue

        return articles


# --- Aggregator ---


class TickerNewsObject:
    def __init__(self, ticker: str) -> None:
        self.ticker: str = ticker
        self.news_sources: dict[
            str,
            type[NewsSource],
        ] = {
            'GoogleFinance': GoogleFinanceSource,
            'YahooFinance': YahooFinanceSource,
            'Finology': FinologySource,
            'Moneycontrol': MoneycontrolSource,
            'EconomicTimesMarkets': EconomicTimesMarketsSource,
            'BusinessStandard': BusinessStandardSource,
            'CnbcTv18': CnbcTv18Source,
            'Reuters': ReutersSource,
        }
        self.articles: list[dict[str, str]] = []
        self.health_reports: list[FetchHealthReport] = []

    @staticmethod
    def _deduplicate_articles(
        articles: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        Remove duplicate articles based on normalized headline text,
        and filter out low-quality headlines (too short, generic roundups).
        Keeps the first occurrence (which comes from the direct scraper if available).
        """
        seen_headlines: set[str] = set()
        unique_articles: list[dict[str, str]] = []

        for article in articles:
            headline_raw = article.get('headline', '').strip()

            # Quality filter: skip short/generic headlines
            if not is_headline_quality(headline_raw):
                continue

            # Dedup: normalize and check for duplicates
            normalized = re.sub(r'[^\w\s]', '', headline_raw.lower())
            normalized = re.sub(r'\s+', ' ', normalized).strip()

            if normalized and normalized not in seen_headlines:
                seen_headlines.add(normalized)
                unique_articles.append(article)

        return unique_articles

    def collect_news(self) -> list[dict[str, str]]:
        """
        Calls each news source's get_articles method to fetch articles for the ticker.
        Deduplicates results and tracks health reports.
        """
        for source_name, source_cls in self.news_sources.items():
            logger.info(f'Fetching articles from {source_name} for {self.ticker}')
            try:
                source_instance = source_cls()
                fetched_articles: list[dict[str, str]] = source_instance.get_articles(
                    self.ticker
                )
                logger.info(
                    f'Fetched {len(fetched_articles)} articles from {source_name} for {self.ticker}'
                )

                # Collect health report if available (DualStrategySource instances)
                if (
                    hasattr(source_instance, 'health_report')
                    and source_instance.health_report
                ):
                    self.health_reports.append(source_instance.health_report)

                if fetched_articles:
                    self.articles.extend(fetched_articles)
            except Exception as e:
                logger.error(
                    f'Failed to fetch from {source_name} for {self.ticker}: {e}'
                )
                continue

        # Deduplicate articles
        original_count = len(self.articles)
        self.articles = self._deduplicate_articles(self.articles)
        dedup_count = original_count - len(self.articles)

        if dedup_count > 0:
            logger.info(f'Removed {dedup_count} duplicate articles for {self.ticker}')

        logger.success(
            f'Collected {len(self.articles)} unique articles in total for {self.ticker}'
        )
        return self.articles


if __name__ == '__main__':
    ticker = 'SBIN'

    ticker_news = TickerNewsObject(ticker)
    articles = ticker_news.collect_news()
    logger.info(f'Collected {len(articles)} articles for {ticker}')

    # Print health reports
    for report in ticker_news.health_reports:
        logger.info(
            f'  {report.source_name}: direct={report.direct_count}, '
            f'fallback={report.fallback_count}, method={report.method_used}'
            f'{", error=" + report.error if report.error else ""}'
        )

# Nifty-500 Live Sentiment Analysis

Real-time sentiment analysis of Nifty-500 stocks and Indices.

![app-img](./res/app.png)

## Overview

This project analyzes the sentiment of Nifty-500 stocks in real-time, providing insights into market trends and investor sentiment.

Ticker-specific articles are sourced from **8 news sources** every 3 hours via GitHub Actions and stored in a persistent [DuckDB](https://duckdb.org/) database.

### News Sources

| Source | Strategy | Status |
|--------|----------|--------|
| **Google Finance** | Direct scraping | ✅ Active |
| **Yahoo Finance** | Direct scraping | ✅ Active |
| **Moneycontrol** | Direct scraper + Google News RSS fallback | ✅ Active |
| **Economic Times Markets** | Direct scraper + Google News RSS fallback | ✅ Active |
| **Business Standard** | Direct scraper + Google News RSS fallback | ✅ Active |
| **CNBC TV18** | Direct scraper + Google News RSS fallback | ✅ Active |
| **Reuters** | Direct scraper + Google News RSS fallback | ✅ Active |
| **Finology** | Direct scraping | ✅ Active |

Articles are then processed for sentiment analysis using the [yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone) FinBERT model offline using GitHub Actions.

### Reliability Features

- **Dual-Strategy Fetching** — Each source tries direct scraping first, falls back to Google News RSS if direct returns no results
- **Per-Domain Rate Limiting** — Thread-safe rate limiter prevents overwhelming any single server
- **Rotating User-Agents** — Randomized browser fingerprints reduce blocking
- **CAPTCHA/Block Detection** — Automatic detection of block pages and CAPTCHAs
- **Article Deduplication** — In-memory deduplication by normalized headline text
- **Headline Cleaning** — Strips source attribution suffixes for cleaner sentiment analysis
- **Freshness Filter** — Discards articles older than 30 days
- **Health Monitoring** — Per-source success/failure tracking with summary reports after each run

<details>
<summary>Info about FinBERT model used</summary>

**Model**: [yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone)

FinBERT is a pre-trained NLP model to analyze sentiment of financial text. It is built by further training the BERT language model on a large financial corpus. The model classifies text into three sentiment categories:

- **Positive** — Bullish or favorable financial sentiment
- **Negative** — Bearish or unfavorable financial sentiment
- **Neutral** — No strong sentiment signal

A **compound score** is computed as: `Positive` when positive > negative, else `-Negative`, clipped to [-1, 1].

</details>

## Automation

The project runs automatically via **GitHub Actions** on the following schedule:

| Trigger | Frequency | Details |
|---------|-----------|---------|
| ⏰ Scheduled | **Every 3 hours** | `cron: 0 */3 * * *` (00:00, 03:00, 06:00, ... UTC) |
| 🖱️ Manual | On demand | Via `workflow_dispatch` in GitHub Actions tab |

**Pipeline steps per run:**
1. Fetch news articles from all 8 sources for Nifty-50 tickers
2. Compute sentiment scores using FinBERT for new articles
3. Aggregate scores and push to Google Sheets (24H, 3D, 7D, 1M windows)
4. Generate the sentiment dashboard HTML
5. Commit updated database and dashboard to the repository

## Installation

```bash
# Clone the repository
git clone https://github.com/pags666/Nitesh_2471M.git
cd Nitesh_2471M

# Make sure you have UV installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
make install

# For development (includes linters, formatters, pre-commit hooks)
make dev-setup
```

## Usage

```bash
# Run the full pipeline (fetch news + sentiment analysis + push to sheets)
make run

# Generate the dashboard only
make dashboard

# Run linting
make lint

# Run tests
make test
```

## Project Structure

```
├── src/
│   ├── main.py              # Entry point: fetch → sentiment → sheets
│   ├── news_fetcher.py      # 8 news source scrapers with dual-strategy fallback
│   ├── utils.py             # HTTP client, rate limiter, sentiment analysis
│   ├── config.py            # All configuration (sources, DB, rate limits)
│   ├── database.py          # DuckDB database manager
│   ├── export_to_sheets.py  # Google Sheets integration
│   └── dashboard-generation.py  # HTML dashboard generator
├── database/
│   └── ticker_data.db       # DuckDB database (auto-updated)
├── .github/workflows/
│   └── main.yml             # GitHub Actions workflow (every 3 hours)
└── pyproject.toml           # Dependencies and tool config
```

## Project Analytics

![Alt](https://repobeats.axiom.co/api/embed/ff35eee02b7cadaba90d5a6699bcb47aea0040f9.svg "Repobeats analytics image")

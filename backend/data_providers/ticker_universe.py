"""
Ticker universe for training (S&P 500 by default).
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

import httpx
import csv
import io


CACHE_PATH = Path(__file__).resolve().parent / "sp500_tickers.json"
CACHE_TTL_DAYS = 7


def _cache_is_fresh() -> bool:
    if not CACHE_PATH.exists():
        return False
    mtime = datetime.fromtimestamp(CACHE_PATH.stat().st_mtime)
    return datetime.now() - mtime < timedelta(days=CACHE_TTL_DAYS)


def load_sp500_tickers() -> List[str]:
    """
    Load S&P 500 tickers from cache or a remote source.
    """
    if _cache_is_fresh():
        return json.loads(CACHE_PATH.read_text())

    urls = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
    ]

    last_error = None
    for url in urls:
        try:
            with httpx.Client(timeout=20) as client:
                response = client.get(url)
                response.raise_for_status()
                csv_text = response.text
            reader = csv.DictReader(io.StringIO(csv_text))
            tickers = sorted({row["Symbol"] for row in reader if row.get("Symbol")})
            if tickers:
                CACHE_PATH.write_text(json.dumps(tickers))
                return tickers
        except Exception as exc:
            last_error = exc
            continue

    if last_error:
        raise last_error
    raise RuntimeError("Unable to load S&P 500 tickers from remote sources.")

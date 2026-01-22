"""
Twelve Data provider for historical stock prices.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import os

import httpx
import pandas as pd


class TwelveDataError(Exception):
    """Raised when Twelve Data returns an error response."""


@dataclass
class TwelveDataConfig:
    api_key: str
    base_url: str = "https://api.twelvedata.com"
    interval: str = "1day"
    days: int = 365
    timeout_seconds: int = 20


class TwelveDataProvider:
    def __init__(self, config: TwelveDataConfig):
        self.config = config

    @classmethod
    def from_env(cls, days: int = 365, interval: str = "1day") -> "TwelveDataProvider":
        api_key = os.getenv("TWELVE_API_SECRET_KEY", "").strip()
        if not api_key:
            raise TwelveDataError("TWELVE_API_SECRET_KEY is not set")
        return cls(TwelveDataConfig(api_key=api_key, days=days, interval=interval))

    def fetch_time_series(self, ticker: str, days: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch time series data from Twelve Data and return a DataFrame
        with a 'Close' column sorted ascending by date.
        """
        output_days = days if days is not None else self.config.days
        # Twelve Data caps outputsize depending on plan; keep it reasonable.
        outputsize = min(max(int(output_days), 1), 5000)

        params = {
            "symbol": ticker,
            "interval": self.config.interval,
            "outputsize": outputsize,
            "apikey": self.config.api_key,
            "format": "JSON",
        }

        url = f"{self.config.base_url}/time_series"

        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            response = client.get(url, params=params)
        data = response.json()

        if isinstance(data, dict) and data.get("status") == "error":
            raise TwelveDataError(data.get("message", "Unknown Twelve Data error"))

        values = data.get("values") if isinstance(data, dict) else None
        if not values:
            raise TwelveDataError("No data returned from Twelve Data")

        df = self._values_to_dataframe(values)
        if df.empty:
            raise TwelveDataError("No usable data returned from Twelve Data")

        return df

    def _values_to_dataframe(self, values: List[Dict[str, str]]) -> pd.DataFrame:
        rows = []
        for item in values:
            close_value = item.get("close")
            date_value = item.get("datetime")
            if close_value is None or date_value is None:
                continue
            rows.append({"Date": date_value, "Close": float(close_value)})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

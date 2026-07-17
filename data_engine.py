"""
data_engine.py
==============
Handles all data-fetching responsibilities for StockBot AI Vision Pro:

  - Historical OHLCV data via yfinance (with TTL-based in-memory cache)
  - Live quote (price, change, stats) via yfinance fast_info
  - Company news headlines via yfinance .news
  - Live USD/INR exchange rate for real-time currency conversion
"""

import contextlib
import os
import time

import pandas as pd
import yfinance as yf

from config import CACHE_TTL, COMPANY_ALIASES, DEFAULT_PERIOD

# yfinance ticker used for USD→INR conversion
_USDINR_TICKER = "USDINR=X"


class DataEngine:
    """
    Provides OHLCV history, live quotes, news, and exchange-rate data.

    All remote calls are wrapped in broad exception handlers so that a
    transient API failure degrades gracefully rather than crashing the server.
    """

    def __init__(self) -> None:
        # {cache_key: (DataFrame, timestamp_float)}
        self._price_cache:   dict[str, tuple[pd.DataFrame, float]] = {}
        # {ticker: (info_dict, timestamp_float)}
        self._profile_cache: dict[str, tuple[dict, float]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_ticker(raw: str) -> str:
        """Map a common name or alias to its canonical ticker symbol."""
        return COMPANY_ALIASES.get(raw.strip().upper(), raw.strip().upper())

    @staticmethod
    def _suppress() -> contextlib.AbstractContextManager:
        """Return a context manager that swallows yfinance's verbose stdout/stderr."""
        return contextlib.redirect_stdout(open(os.devnull, 'w'))   # noqa: WPS515

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_stock_data(self, ticker_input: str, period: str = DEFAULT_PERIOD) -> tuple[pd.DataFrame | None, str | None]:
        """
        Download OHLCV history for *ticker_input* over *period*.

        Tries the resolved ticker first, then appends ``.NS`` and ``.BO``
        suffixes for Indian exchanges if the bare symbol returns no data.

        Results are cached for ``CACHE_TTL`` seconds to avoid redundant
        network calls on repeated requests.

        Parameters
        ----------
        ticker_input : str
            Raw user input (e.g. ``"RELIANCE"``, ``"AAPL"``).
        period : str
            yfinance period string: ``"1mo"``, ``"3mo"``, ``"6mo"``, ``"1y"``, ``"5y"``.

        Returns
        -------
        tuple[pd.DataFrame | None, str | None]
            (history_df, resolved_ticker)  or  (None, None) on failure.
        """
        ticker = self._resolve_ticker(ticker_input)
        cache_key = f"{ticker}_{period}"

        # Return cached result if still fresh
        if cache_key in self._price_cache:
            cached_df, ts = self._price_cache[cache_key]
            if time.time() - ts < CACHE_TTL:
                return cached_df, ticker

        # Build candidate list: bare ticker + NSE/BSE variants
        candidates = [ticker]
        if '.' not in ticker:
            candidates += [f"{ticker}.NS", f"{ticker}.BO"]

        for sym in candidates:
            try:
                with open(os.devnull, 'w') as devnull, \
                     contextlib.redirect_stdout(devnull), \
                     contextlib.redirect_stderr(devnull):
                    history = yf.download(sym, period=period, progress=False, auto_adjust=True)

                # Flatten MultiIndex columns produced by newer yfinance versions
                if isinstance(history.columns, pd.MultiIndex):
                    history.columns = history.columns.droplevel(1)

                if not history.empty:
                    self._price_cache[cache_key] = (history, time.time())
                    return history, sym
            except Exception:
                continue

        return None, None

    def fetch_live_quote(self, ticker: str) -> dict | None:
        """
        Fetch a real-time market quote for *ticker*.

        Uses ``yfinance.Ticker.fast_info`` for low-latency price data and
        a TTL-cached ``stock.info`` call for slower fundamentals
        (P/E ratio, 52-week range, average volume, company name).

        Parameters
        ----------
        ticker : str
            Canonical ticker symbol (e.g. ``"RELIANCE.NS"``).

        Returns
        -------
        dict | None
            Quote dictionary or ``None`` on error.
        """
        ticker = self._resolve_ticker(ticker)

        try:
            stock = yf.Ticker(ticker)
            fi    = dict(stock.fast_info)

            last_price = fi.get('lastPrice')
            prev_close = fi.get('previousClose')

            # Fallback: pull last close from 1-day history
            if last_price is None or (isinstance(last_price, float) and pd.isna(last_price)):
                hist = stock.history(period="1d")
                if not hist.empty:
                    last_price = float(hist['Close'].iloc[-1])

            if prev_close is None or (isinstance(prev_close, float) and pd.isna(prev_close)):
                prev_close = last_price

            change         = (last_price - prev_close) if (last_price and prev_close) else 0.0
            change_percent = (change / prev_close * 100) if prev_close else 0.0

            # Slow fundamentals (cached)
            profile = self._get_profile(ticker, stock, fi)

            return {
                'ticker':        ticker,
                'name':          profile['longName'],
                'price':         float(last_price) if last_price is not None else None,
                'prev_close':    float(prev_close) if prev_close is not None else None,
                'change':        float(change),
                'change_percent': float(change_percent),
                'open':          _safe_float(fi.get('open')),
                'high':          _safe_float(fi.get('dayHigh')),
                'low':           _safe_float(fi.get('dayLow')),
                'volume':        _safe_int(fi.get('lastVolume')),
                'avg_volume':    _safe_int(profile.get('averageVolume')),
                'market_cap':    _safe_float(fi.get('marketCap')),
                'pe_ratio':      profile.get('forwardPE', 'N/A'),
                'fifty_two_high': _safe_float(profile.get('fiftyTwoWeekHigh')),
                'fifty_two_low':  _safe_float(profile.get('fiftyTwoWeekLow')),
                'currency':      fi.get('currency', 'USD'),
                'exchange':      fi.get('exchange', ''),
                'timezone':      fi.get('timezone', 'UTC'),
            }
        except Exception as exc:
            print(f"[DataEngine] Error fetching live quote for {ticker}: {exc}")
            return None

    def fetch_news(self, ticker: str, max_items: int = 5) -> list[dict]:
        """
        Return recent news headlines for *ticker* from Yahoo Finance.

        Parameters
        ----------
        ticker : str
            Canonical ticker symbol.
        max_items : int
            Maximum number of headlines to return.

        Returns
        -------
        list[dict]
            Each item contains: title, link, publisher, providerPublishTime.
        """
        try:
            news_raw = yf.Ticker(ticker).news or []
            result = []
            for item in news_raw[:max_items]:
                content = item.get('content', {})
                title   = content.get('title') or item.get('title', '')
                link    = (content.get('canonicalUrl', {}).get('url')
                           or content.get('clickThroughUrl', {}).get('url')
                           or item.get('link', '#'))
                publisher = (content.get('provider', {}).get('displayName')
                             or item.get('publisher', 'Yahoo Finance'))
                pub_time  = content.get('pubDate') or item.get('providerPublishTime', '')
                if title:
                    result.append({
                        'title':     title,
                        'link':      link,
                        'publisher': publisher,
                        'pubDate':   pub_time,
                    })
            return result
        except Exception as exc:
            print(f"[DataEngine] Error fetching news for {ticker}: {exc}")
            return []

    def fetch_exchange_rate(self) -> float | None:
        """
        Fetch the live USD/INR exchange rate from Yahoo Finance.

        Returns
        -------
        float | None
            Current rate (e.g. 83.72) or ``None`` on failure.
        """
        try:
            fi = dict(yf.Ticker(_USDINR_TICKER).fast_info)
            rate = fi.get('lastPrice')
            if rate and not pd.isna(rate):
                return round(float(rate), 4)
        except Exception as exc:
            print(f"[DataEngine] Error fetching USD/INR rate: {exc}")
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_profile(self, ticker: str, stock: yf.Ticker, fi: dict) -> dict:
        """
        Return cached slow-info fundamentals, fetching from Yahoo if stale.

        Parameters
        ----------
        ticker : str
            Canonical ticker symbol (cache key).
        stock : yf.Ticker
            Pre-constructed Ticker object (avoids duplicate construction).
        fi : dict
            Already-fetched fast_info dict (fallback values).
        """
        if ticker in self._profile_cache:
            profile, ts = self._profile_cache[ticker]
            if time.time() - ts < CACHE_TTL * 6:   # profile refreshes every 30 min
                return profile

        try:
            info    = stock.info
            profile = {
                'longName':        info.get('longName', ticker),
                'forwardPE':       info.get('forwardPE', info.get('trailingPE', 'N/A')),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', fi.get('yearHigh')),
                'fiftyTwoWeekLow':  info.get('fiftyTwoWeekLow',  fi.get('yearLow')),
                'averageVolume':   info.get('averageVolume', fi.get('threeMonthAverageVolume')),
            }
        except Exception:
            profile = {
                'longName':        ticker,
                'forwardPE':       'N/A',
                'fiftyTwoWeekHigh': fi.get('yearHigh'),
                'fiftyTwoWeekLow':  fi.get('yearLow'),
                'averageVolume':   fi.get('threeMonthAverageVolume'),
            }

        self._profile_cache[ticker] = (profile, time.time())
        return profile


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _safe_float(value) -> float | None:
    """Return float(value) or None if value is None/NaN."""
    try:
        v = float(value)
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    """Return int(value) or None if value is None/NaN."""
    try:
        v = float(value)
        return None if pd.isna(v) else int(v)
    except (TypeError, ValueError):
        return None

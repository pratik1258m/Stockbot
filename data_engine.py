import yfinance as yf
import pandas as pd
import os
import contextlib
from config import COMPANY_ALIASES, DEFAULT_PERIOD
class DataEngine:
    def __init__(self):
        self.cached_data = {}
    def fetch_stock_data(self, ticker_input, period=DEFAULT_PERIOD):
        ticker_input = ticker_input.strip().upper()
        if ticker_input in COMPANY_ALIASES:
            ticker = COMPANY_ALIASES[ticker_input]
        else:
            ticker = ticker_input
        cache_key = f"{ticker}_{period}"
        if cache_key in self.cached_data:
            return self.cached_data[cache_key], ticker
        variations = [ticker]
        if '.' not in ticker:
            variations.append(f"{ticker}.NS")
            variations.append(f"{ticker}.BO")
        for sym in variations:
            try:
                with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    history = yf.download(sym, period=period, progress=False)
                    if isinstance(history.columns, pd.MultiIndex):
                        history.columns = history.columns.droplevel(1)
                if not history.empty:
                    self.cached_data[cache_key] = history
                    return history, sym
            except Exception:
                continue
        return None, None
    def get_company_info(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            return stock.info
        except Exception:
            return {}

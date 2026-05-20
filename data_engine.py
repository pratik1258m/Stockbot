import yfinance as yf
import pandas as pd
import os
import contextlib
from config import COMPANY_ALIASES, DEFAULT_PERIOD

class DataEngine:
    def __init__(self):
        self.cached_data = {}
        self.cached_company_profiles = {}

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

    def fetch_live_quote(self, ticker):
        ticker = ticker.strip().upper()
        if ticker in COMPANY_ALIASES:
            ticker = COMPANY_ALIASES[ticker]

        try:
            stock = yf.Ticker(ticker)
            fi = dict(stock.fast_info)

            # Fetch company name and P/E ratio (which are in stock.info and slow to fetch, so we cache them)
            company_name = ticker
            pe_ratio = "N/A"
            fifty_two_week_high = fi.get('yearHigh', None)
            fifty_two_week_low = fi.get('yearLow', None)
            avg_volume = fi.get('threeMonthAverageVolume', None)

            if ticker not in self.cached_company_profiles:
                try:
                    info = stock.info
                    self.cached_company_profiles[ticker] = {
                        'longName': info.get('longName', ticker),
                        'forwardPE': info.get('forwardPE', info.get('trailingPE', 'N/A')),
                        'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', fifty_two_week_high),
                        'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', fifty_two_week_low),
                        'averageVolume': info.get('averageVolume', avg_volume)
                    }
                except Exception:
                    self.cached_company_profiles[ticker] = {
                        'longName': ticker,
                        'forwardPE': 'N/A',
                        'fiftyTwoWeekHigh': fifty_two_week_high,
                        'fiftyTwoWeekLow': fifty_two_week_low,
                        'averageVolume': avg_volume
                    }

            profile = self.cached_company_profiles[ticker]
            company_name = profile['longName']
            pe_ratio = profile['forwardPE']
            fifty_two_week_high = profile['fiftyTwoWeekHigh']
            fifty_two_week_low = profile['fiftyTwoWeekLow']
            avg_volume = profile['averageVolume']

            last_price = fi.get('lastPrice')
            prev_close = fi.get('previousClose')

            # Fallback if lastPrice is None
            if last_price is None or pd.isna(last_price):
                # Try getting the last close from history
                hist = stock.history(period="1d")
                if not hist.empty:
                    last_price = float(hist['Close'].iloc[-1])

            if prev_close is None or pd.isna(prev_close):
                prev_close = last_price

            change = last_price - prev_close if last_price and prev_close else 0.0
            change_percent = (change / prev_close) * 100 if prev_close else 0.0

            return {
                'ticker': ticker,
                'name': company_name,
                'price': float(last_price) if last_price is not None else None,
                'prev_close': float(prev_close) if prev_close is not None else None,
                'change': float(change),
                'change_percent': float(change_percent),
                'open': float(fi.get('open')) if fi.get('open') is not None else None,
                'high': float(fi.get('dayHigh')) if fi.get('dayHigh') is not None else None,
                'low': float(fi.get('dayLow')) if fi.get('dayLow') is not None else None,
                'volume': int(fi.get('lastVolume')) if fi.get('lastVolume') is not None else None,
                'avg_volume': int(avg_volume) if avg_volume is not None else None,
                'market_cap': float(fi.get('marketCap')) if fi.get('marketCap') is not None else None,
                'pe_ratio': pe_ratio,
                'fifty_two_high': float(fifty_two_week_high) if fifty_two_week_high is not None else None,
                'fifty_two_low': float(fifty_two_week_low) if fifty_two_week_low is not None else None,
                'currency': fi.get('currency', 'USD'),
                'exchange': fi.get('exchange', ''),
                'timezone': fi.get('timezone', 'UTC')
            }
        except Exception as e:
            print(f"Error fetching live quote for {ticker}: {e}")
            return None

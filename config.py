"""
config.py
=========
Central configuration for StockBot AI Vision Pro.
All tunable parameters live here so nothing is hardcoded across modules.
"""

# ---------------------------------------------------------------------------
# Company name → ticker symbol aliases
# Lets users type "RELIANCE" instead of "RELIANCE.NS", etc.
# ---------------------------------------------------------------------------
COMPANY_ALIASES: dict[str, str] = {
    # US Tech
    'GOOGLE': 'GOOGL',
    'MICROSOFT': 'MSFT',
    'APPLE': 'AAPL',
    'TESLA': 'TSLA',
    'AMAZON': 'AMZN',
    'NETFLIX': 'NFLX',
    'FACEBOOK': 'META',
    'META': 'META',
    'NVIDIA': 'NVDA',
    'AMD': 'AMD',
    # Indian Large-Cap
    'INFOSYS': 'INFY.NS',
    'TATA STEEL': 'TATASTEEL.NS',
    'TATA MOTORS': 'TATAMOTORS.NS',
    'RELIANCE': 'RELIANCE.NS',
    'WIPRO': 'WIPRO.NS',
    'HDFC': 'HDFCBANK.NS',
    'HDFC BANK': 'HDFCBANK.NS',
    'ICICI BANK': 'ICICIBANK.NS',
    'SBI': 'SBIN.NS',
    'STATE BANK OF INDIA': 'SBIN.NS',
    'BAJAJ FINANCE': 'BAJFINANCE.NS',
    'ASIAN PAINTS': 'ASIANPAINT.NS',
    'MARUTI': 'MARUTI.NS',
    'TITAN': 'TITAN.NS',
    'SUN PHARMA': 'SUNPHARMA.NS',
    'ULTRATECH': 'ULTRACEMCO.NS',
    'POWER GRID': 'POWERGRID.NS',
    'TCS': 'TCS.NS',
    'ITC': 'ITC.NS',
    'PW': 'PWL.NS',
    'PWL': 'PWL.NS',
    'PHYSICSWALLAH': 'PWL.NS',
    'PHYSICS WALLAH': 'PWL.NS',
}

# ---------------------------------------------------------------------------
# Data fetching defaults
# ---------------------------------------------------------------------------
DEFAULT_PERIOD: str = "1y"

# In-memory cache TTL in seconds (5 minutes).
# After this duration a fresh yfinance download is triggered.
CACHE_TTL: int = 300

# ---------------------------------------------------------------------------
# Technical indicator windows (all in trading days)
# ---------------------------------------------------------------------------
INDICATOR_WINDOWS: dict[str, int] = {
    'SMA_FAST': 20,       # Short-term Simple Moving Average
    'SMA_SLOW': 50,       # Long-term Simple Moving Average
    'RSI': 14,            # Relative Strength Index (Wilder's period)
    'MACD_FAST': 12,      # MACD fast EMA span
    'MACD_SLOW': 26,      # MACD slow EMA span
    'MACD_SIGNAL': 9,     # MACD signal EMA span
    'BB': 20,             # Bollinger Bands rolling window
    'ATR': 14,            # Average True Range period
}

# ---------------------------------------------------------------------------
# Price prediction
# ---------------------------------------------------------------------------
PREDICTION_DAYS: int = 10   # Number of future days to project via linear regression

# ---------------------------------------------------------------------------
# Frontend polling interval (milliseconds) — sent to the template
# ---------------------------------------------------------------------------
POLL_INTERVAL_MS: int = 10_000   # 10 seconds

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
MAX_TICKER_LENGTH: int = 20
ALLOWED_TICKER_CHARS: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789. "

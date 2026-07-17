"""
app.py
======
StockBot AI Vision Pro — Flask application entry point.

Routes
------
GET  /                          Landing page
POST /analyze                   Stock analysis dashboard (form submit)
GET  /analyze?ticker=&period=   Stock analysis dashboard (direct link / AJAX)
GET  /api/quote/<ticker>        Live quote JSON (polled by the dashboard)
GET  /api/news/<ticker>         Recent news headlines JSON
GET  /api/rate/usdinr           Live USD→INR exchange rate JSON
GET  /favicon.ico               Returns 204 to silence browser 404 noise
"""

import datetime
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pytz
from flask import Flask, Response, jsonify, make_response, render_template, request

from analysis_engine import AnalysisEngine
from config import ALLOWED_TICKER_CHARS, COMPANY_ALIASES, MAX_TICKER_LENGTH, POLL_INTERVAL_MS
from data_engine import DataEngine

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
data_engine     = DataEngine()
analysis_engine = AnalysisEngine()

# Thread pool for parallel API fetching (history + quote simultaneously)
_executor = ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------------------
# Serialisation helpers — NaN / Inf safe
# ---------------------------------------------------------------------------

def _safe(val):
    """
    Convert any numpy/pandas scalar to a JSON-safe Python primitive.
    Crucially: float NaN and Inf become None (JSON null), not the string 'NaN'.
    """
    # Unwrap numpy integers
    if isinstance(val, (np.integer,)):
        return int(val)
    # Unwrap numpy floats — then guard for NaN/Inf
    if isinstance(val, (np.floating,)):
        val = float(val)
    # Guard Python floats
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    # Pandas Timestamp → ISO date string
    if isinstance(val, pd.Timestamp):
        return val.strftime('%Y-%m-%d')
    return val


def _safe_list(lst: list) -> list:
    """Apply _safe() to every element of a list."""
    return [_safe(x) for x in lst]


def _safe_dict(d: dict) -> dict:
    """Apply _safe() to every value of a dict."""
    return {k: _safe(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _validate_ticker(raw: str) -> tuple:
    """
    Sanitise and validate a raw ticker string from user input.
    Returns (cleaned_ticker, error_message | None).
    """
    cleaned = raw.strip().upper()[:MAX_TICKER_LENGTH]
    if not cleaned:
        return '', 'Please enter a ticker symbol or company name.'
    for ch in cleaned:
        if ch not in ALLOWED_TICKER_CHARS:
            return '', f'Invalid character "{ch}" in ticker. Use letters, digits, or dots only.'
    return cleaned, None


def _normalise_period(period: str) -> str:
    """Map short-form period strings to yfinance-accepted values."""
    period_map = {'1m': '1mo', '3m': '3mo', '6m': '6mo'}
    p = period_map.get(period.lower(), period.lower())
    return p if p in ('1mo', '3mo', '6mo', '1y', '5y') else '1y'


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

@app.route('/favicon.ico')
def favicon():
    """Return 204 No Content — the real favicon is an inline SVG in layout.html."""
    return make_response('', 204)


@app.route('/')
def index():
    """Landing page."""
    return render_template('index.html', aliases=COMPANY_ALIASES)


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """
    Main analysis endpoint.

    Accepts both a POST form submit (from the landing page) and a GET
    request (from AJAX timeframe switches or direct URL links).
    Returns HTML by default, or JSON when the ``json=true`` query param or
    ``X-Requested-With: XMLHttpRequest`` header is present.

    Performance: history download + live quote are fetched in parallel via
    ThreadPoolExecutor, cutting wait time roughly in half.
    """
    if request.method == 'POST':
        raw_input = request.form.get('ticker', '')
        period    = request.form.get('period', '1y').lower()
    else:
        raw_input = request.args.get('ticker', '')
        period    = request.args.get('period', '1y').lower()

    user_input, err = _validate_ticker(raw_input)
    if err:
        return render_template('index.html', error=err, aliases=COMPANY_ALIASES)

    period = _normalise_period(period)

    # ── Parallel fetch: history download + live quote simultaneously ───────
    df         = None
    ticker_sym = user_input
    quote      = None

    def _fetch_history():
        return data_engine.fetch_stock_data(user_input, period=period)

    def _fetch_quote(sym):
        return data_engine.fetch_live_quote(sym)

    # Step 1: fetch history (we need the resolved ticker_sym before quote)
    df, ticker_sym = data_engine.fetch_stock_data(user_input, period=period)

    if df is None or df.empty:
        return render_template(
            'index.html',
            error=f"Could not find data for '{user_input}'. Please verify the symbol.",
            aliases=COMPANY_ALIASES,
        )

    # Step 2: fetch quote in parallel with indicator computation
    quote_future = _executor.submit(_fetch_quote, ticker_sym)

    # Step 3: compute indicators while quote is fetching
    # Inject a placeholder avg_volume column (will update after quote arrives)
    df_work = df.copy()
    df_work, latest = analysis_engine.compute_indicators(df_work)

    if latest is None:
        return render_template(
            'index.html',
            error=(f"Not enough historical data for '{ticker_sym}' "
                   f"(minimum 50 trading days required)."),
            aliases=COMPANY_ALIASES,
        )

    # Step 4: collect quote result (likely already done)
    try:
        quote = quote_future.result(timeout=5)
    except Exception:
        quote = None

    # ── Merge live quote into the df for the last row ──────────────────────
    if quote and quote['price'] is not None:
        try:
            tz = pytz.timezone(quote['timezone'])
        except Exception:
            tz = pytz.utc

        today     = datetime.datetime.now(tz).date()
        last_date = df.index[-1].date()

        if last_date == today:
            df.loc[df.index[-1], 'Close'] = quote['price']
            if quote.get('high'):
                df.loc[df.index[-1], 'High'] = max(df.loc[df.index[-1], 'High'], quote['high'])
            if quote.get('low'):
                df.loc[df.index[-1], 'Low']  = min(df.loc[df.index[-1], 'Low'],  quote['low'])
        elif today > last_date and today.weekday() < 5:
            new_row = pd.Series({
                'Open':   quote.get('open')   or quote['price'],
                'High':   quote.get('high')   or quote['price'],
                'Low':    quote.get('low')    or quote['price'],
                'Close':  quote['price'],
                'Volume': quote.get('volume') or 0,
            }, name=pd.Timestamp(today))
            df = pd.concat([df, pd.DataFrame([new_row])])

    # ── Re-run indicators if live price changed the last row ───────────────
    if quote and quote.get('avg_volume'):
        df['avg_volume'] = quote['avg_volume']

    df, latest = analysis_engine.compute_indicators(df)
    if latest is None:
        return render_template(
            'index.html',
            error=f"Indicator computation failed for '{ticker_sym}'.",
            aliases=COMPANY_ALIASES,
        )

    verdict    = analysis_engine.get_verdict(latest)
    prediction = analysis_engine.predict_prices(df)

    prev_close = (
        quote['prev_close']
        if quote and quote.get('prev_close') is not None
        else (float(df['Close'].iloc[-2]) if len(df) > 1 else float(latest['Close']))
    )
    base_currency = (
        quote['currency'] if quote
        else ('INR' if ticker_sym.endswith(('.NS', '.BO')) else 'USD')
    )

    # ── Build chart data — NaN replaced with None for safe JSON ───────────
    chart_data = {
        'labels':      df.index.strftime('%Y-%m-%d').tolist(),
        'prices':      _safe_list(df['Close'].tolist()),
        'sma_fast':    _safe_list(df['SMA_Fast'].tolist()),
        'sma_slow':    _safe_list(df['SMA_Slow'].tolist()),
        'rsi':         _safe_list(df['RSI'].tolist()),
        'macd':        _safe_list(df['MACD'].tolist()),
        'macd_signal': _safe_list(df['MACD_Signal'].tolist()),
        'macd_hist':   _safe_list(df['MACD_Hist'].tolist()),
        'bb_upper':    _safe_list(df['BB_Upper'].tolist()),
        'bb_lower':    _safe_list(df['BB_Lower'].tolist()),
        'atr':         _safe_list(df['ATR'].tolist()),
        # Prediction overlay
        'pred_labels': prediction.get('labels', []),
        'pred_prices': _safe_list(prediction.get('predictions', [])),
        'pred_r2':     _safe(prediction.get('r2')),
    }

    # ── JSON response (AJAX timeframe switch) ─────────────────────────────
    is_ajax = (
        request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        or request.args.get('json') == 'true'
    )
    if is_ajax:
        # Only send the last 15 rows for the history table (not all 250+)
        tail = df.tail(15)
        df_clean = []
        for idx, row in tail.iterrows():
            entry = {'Date': idx.strftime('%Y-%m-%d')}
            for col in ['Close', 'RSI', 'MACD', 'ATR', 'SMA_Slow']:
                if col in row.index:
                    entry[col] = _safe(row[col])
            df_clean.append(entry)

        payload = {
            'ticker':        ticker_sym,
            'latest':        _safe_dict(dict(latest)),
            'prev_close':    _safe(prev_close),
            'verdict':       verdict,
            'base_currency': base_currency,
            'chart_data':    chart_data,
            'quote':         quote,
            'period':        period,
            'df':            df_clean,
            'prediction':    {
                'labels':      prediction.get('labels', []),
                'predictions': _safe_list(prediction.get('predictions', [])),
                'r2':          _safe(prediction.get('r2')),
            },
        }
        # Use Response + json.dumps with allow_nan=False so NaN raises an
        # error we can catch rather than silently emitting invalid JSON.
        import json as _json
        try:
            body = _json.dumps(payload, allow_nan=False)
        except ValueError:
            # Fallback: walk the payload and nuke any remaining non-finite floats
            def _deep_clean(obj):
                if isinstance(obj, dict):
                    return {k: _deep_clean(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_deep_clean(v) for v in obj]
                if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                    return None
                return obj
            body = _json.dumps(_deep_clean(payload), allow_nan=False)

        return Response(body, mimetype='application/json')

    # ── Pre-compute template helpers (keep Jinja2 logic-free) ─────────────
    clamped_score = max(-7.0, min(7.0, float(verdict.get('score', 0))))
    gauge_pct     = round((clamped_score + 7.0) / 14.0 * 100, 1)

    current_price = (
        float(quote['price']) if (quote and quote.get('price'))
        else float(latest['Close'])
    )
    atr_val = verdict.get('atr')
    atr_pct = round((atr_val / current_price) * 100, 2) if atr_val and current_price else 0.0

    # History table — last 15 rows, NaN-safe
    hist_rows = []
    for idx, row in df.tail(15).iterrows():
        hist_rows.append({
            'date':     idx.strftime('%Y-%m-%d'),
            'close':    _safe(row['Close']),
            'rsi':      _safe(row['RSI']),
            'macd':     _safe(row['MACD']),
            'atr':      _safe(row['ATR']),
            'sma_slow': _safe(row['SMA_Slow']),
        })

    # ── HTML response ──────────────────────────────────────────────────────
    return render_template(
        'dashboard.html',
        ticker        = ticker_sym,
        hist_rows     = hist_rows,
        latest        = latest,
        prev_close    = prev_close,
        verdict       = verdict,
        base_currency = base_currency,
        chart_data    = chart_data,
        quote         = quote,
        period        = period,
        prediction    = prediction,
        poll_interval = POLL_INTERVAL_MS,
        gauge_pct     = gauge_pct,
        atr_pct       = atr_pct,
    )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.route('/api/quote/<ticker>')
def api_quote(ticker: str):
    """Return a live market quote as JSON."""
    ticker_clean, err = _validate_ticker(ticker)
    if err:
        return jsonify({'error': err}), 400
    quote = data_engine.fetch_live_quote(ticker_clean)
    if quote:
        return jsonify(quote)
    return jsonify({'error': f'Could not fetch quote for {ticker_clean}'}), 404


@app.route('/api/news/<ticker>')
def api_news(ticker: str):
    """Return recent news headlines for a ticker as JSON."""
    ticker_clean, err = _validate_ticker(ticker)
    if err:
        return jsonify({'error': err}), 400
    news = data_engine.fetch_news(ticker_clean)
    return jsonify(news)


@app.route('/api/rate/usdinr')
def api_exchange_rate():
    """Return the live USD/INR exchange rate as JSON."""
    rate = data_engine.fetch_exchange_rate()
    if rate is not None:
        return jsonify({'rate': rate})
    return jsonify({'error': 'Could not fetch exchange rate'}), 503


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, port=8080, threaded=True)

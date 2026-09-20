
import datetime
import json
import math
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, make_response, render_template, request

from analysis_engine import AnalysisEngine
from config import ALLOWED_TICKER_CHARS, COMPANY_ALIASES, MAX_TICKER_LENGTH, POLL_INTERVAL_MS
from data_engine import DataEngine

app = Flask(__name__)
data_engine = DataEngine()
analysis_engine = AnalysisEngine()


def _safe(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        val = float(val)
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(val, pd.Timestamp):
        return val.strftime('%Y-%m-%d')
    return val


def _safe_list(lst: list) -> list:
    return [_safe(x) for x in lst]


def _safe_dict(d: dict) -> dict:
    return {k: _safe(v) for k, v in d.items()}


def _validate_ticker(raw: str) -> tuple:
    cleaned = raw.strip().upper()[:MAX_TICKER_LENGTH]
    if not cleaned:
        return '', 'Please enter a ticker symbol or company name.'
    for ch in cleaned:
        if ch not in ALLOWED_TICKER_CHARS:
            return '', f'Invalid character "{ch}" in ticker. Use letters, digits, or dots only.'
    return cleaned, None


def _normalise_period(period: str) -> str:
    period_map = {'1m': '1mo', '3m': '3mo', '6m': '6mo'}
    p = period_map.get(period.lower(), period.lower())
    return p if p in ('1mo', '3mo', '6mo', '1y', '5y') else '1y'


@app.route('/favicon.ico')
def favicon():
    return make_response('', 204)


@app.route('/')
def index():
    return render_template('index.html', aliases=COMPANY_ALIASES)


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        raw_input = request.form.get('ticker', '')
        period = request.form.get('period', '1y').lower()
    else:
        raw_input = request.args.get('ticker', '')
        period = request.args.get('period', '1y').lower()

    user_input, err = _validate_ticker(raw_input)
    if err:
        return render_template('index.html', error=err, aliases=COMPANY_ALIASES)

    period = _normalise_period(period)

    df, ticker_sym = data_engine.fetch_stock_data(user_input, period=period)
    if df is None or df.empty:
        return render_template(
            'index.html',
            error=f"Could not find data for '{user_input}'. Please verify the symbol.",
            aliases=COMPANY_ALIASES,
        )

    quote = data_engine.fetch_live_quote(ticker_sym)

    if quote and quote.get('price') is not None:
        try:
            tz = ZoneInfo(quote.get('timezone', 'UTC'))
        except Exception:
            tz = datetime.timezone.utc

        today = datetime.datetime.now(tz).date()
        last_date = df.index[-1].date()

        if last_date == today:
            df.loc[df.index[-1], 'Close'] = quote['price']
            if quote.get('high'):
                df.loc[df.index[-1], 'High'] = max(df.loc[df.index[-1], 'High'], quote['high'])
            if quote.get('low'):
                df.loc[df.index[-1], 'Low'] = min(df.loc[df.index[-1], 'Low'], quote['low'])
        elif today > last_date and today.weekday() < 5:
            new_row = pd.Series({
                'Open': quote.get('open') or quote['price'],
                'High': quote.get('high') or quote['price'],
                'Low': quote.get('low') or quote['price'],
                'Close': quote['price'],
                'Volume': quote.get('volume') or 0,
            }, name=pd.Timestamp(today))
            df = pd.concat([df, pd.DataFrame([new_row])])

    if quote and quote.get('avg_volume'):
        df['avg_volume'] = quote['avg_volume']

    df, latest = analysis_engine.compute_indicators(df)
    if latest is None:
        return render_template(
            'index.html',
            error=(f"Not enough historical data for '{ticker_sym}' "
                   f"(minimum 50 trading days required)."),
            aliases=COMPANY_ALIASES,
        )

    verdict = analysis_engine.get_verdict(latest)
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

    chart_data = {
        'labels': df.index.strftime('%Y-%m-%d').tolist(),
        'prices': _safe_list(df['Close'].tolist()),
        'sma_fast': _safe_list(df['SMA_Fast'].tolist()),
        'sma_slow': _safe_list(df['SMA_Slow'].tolist()),
        'rsi': _safe_list(df['RSI'].tolist()),
        'macd': _safe_list(df['MACD'].tolist()),
        'macd_signal': _safe_list(df['MACD_Signal'].tolist()),
        'macd_hist': _safe_list(df['MACD_Hist'].tolist()),
        'bb_upper': _safe_list(df['BB_Upper'].tolist()),
        'bb_lower': _safe_list(df['BB_Lower'].tolist()),
        'atr': _safe_list(df['ATR'].tolist()),
        'pred_labels': prediction.get('labels', []),
        'pred_prices': _safe_list(prediction.get('predictions', [])),
        'pred_upper': _safe_list(prediction.get('upper_bounds', [])),
        'pred_lower': _safe_list(prediction.get('lower_bounds', [])),
        'pred_accuracy': _safe(prediction.get('accuracy_pct')),
        'pred_mape': _safe(prediction.get('mape')),
        'pred_mae': _safe(prediction.get('mae')),
        'pred_rmse': _safe(prediction.get('rmse')),
        'pred_dir_acc': _safe(prediction.get('dir_accuracy')),
    }

    is_ajax = (
        request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        or request.args.get('json') == 'true'
    )
    if is_ajax:
        tail = df.tail(15)
        df_clean = []
        for idx, row in tail.iterrows():
            entry = {'Date': idx.strftime('%Y-%m-%d')}
            for col in ['Close', 'RSI', 'MACD', 'ATR', 'SMA_Slow']:
                if col in row.index:
                    entry[col] = _safe(row[col])
            df_clean.append(entry)

        payload = {
            'ticker': ticker_sym,
            'latest': _safe_dict(dict(latest)),
            'prev_close': _safe(prev_close),
            'verdict': verdict,
            'base_currency': base_currency,
            'chart_data': chart_data,
            'quote': quote,
            'period': period,
            'df': df_clean,
            'prediction': {
                'labels': prediction.get('labels', []),
                'predictions': _safe_list(prediction.get('predictions', [])),
                'upper_bounds': _safe_list(prediction.get('upper_bounds', [])),
                'lower_bounds': _safe_list(prediction.get('lower_bounds', [])),
                'accuracy_pct': _safe(prediction.get('accuracy_pct')),
                'mape': _safe(prediction.get('mape')),
                'mae': _safe(prediction.get('mae')),
                'rmse': _safe(prediction.get('rmse')),
                'max_error': _safe(prediction.get('max_error')),
                'dir_accuracy': _safe(prediction.get('dir_accuracy')),
                'dir_match': prediction.get('dir_match'),
                'backtest': [_safe_dict(r) for r in prediction.get('backtest', [])],
                'forecast': [_safe_dict(r) for r in prediction.get('forecast', [])],
                'model_name': prediction.get('model_name'),
            },
        }
        return Response(json.dumps(payload), mimetype='application/json')

    clamped_score = max(-7.0, min(7.0, float(verdict.get('score', 0))))
    gauge_pct = round((clamped_score + 7.0) / 14.0 * 100, 1)

    current_price = (
        float(quote['price']) if (quote and quote.get('price'))
        else float(latest['Close'])
    )
    atr_val = verdict.get('atr')
    atr_pct = round((atr_val / current_price) * 100, 2) if atr_val and current_price else 0.0

    hist_rows = []
    for idx, row in df.tail(15).iterrows():
        hist_rows.append({
            'date': idx.strftime('%Y-%m-%d'),
            'close': _safe(row['Close']),
            'rsi': _safe(row['RSI']),
            'macd': _safe(row['MACD']),
            'atr': _safe(row['ATR']),
            'sma_slow': _safe(row['SMA_Slow']),
        })

    return render_template(
        'dashboard.html',
        ticker=ticker_sym,
        hist_rows=hist_rows,
        latest=latest,
        prev_close=prev_close,
        verdict=verdict,
        base_currency=base_currency,
        chart_data=chart_data,
        quote=quote,
        period=period,
        prediction=prediction,
        poll_interval=POLL_INTERVAL_MS,
        gauge_pct=gauge_pct,
        atr_pct=atr_pct,
    )


@app.route('/api/quote/<ticker>')
def api_quote(ticker: str):
    ticker_clean, err = _validate_ticker(ticker)
    if err:
        return jsonify({'error': err}), 400
    quote = data_engine.fetch_live_quote(ticker_clean)
    if quote:
        return jsonify(quote)
    return jsonify({'error': f'Could not fetch quote for {ticker_clean}'}), 404


@app.route('/api/news/<ticker>')
def api_news(ticker: str):
    ticker_clean, err = _validate_ticker(ticker)
    if err:
        return jsonify({'error': err}), 400
    news = data_engine.fetch_news(ticker_clean)
    return jsonify(news)


@app.route('/api/rate/usdinr')
def api_exchange_rate():
    rate = data_engine.fetch_exchange_rate()
    if rate is not None:
        return jsonify({'rate': rate})
    return jsonify({'error': 'Could not fetch exchange rate'}), 503


if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, port=8080, threaded=True)

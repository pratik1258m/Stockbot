from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import datetime
import pytz
from data_engine import DataEngine
from analysis_engine import AnalysisEngine
from config import COMPANY_ALIASES

def sanitize_val(val):
    if pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (np.integer, np.floating)):
        return float(val)
    if isinstance(val, pd.Timestamp):
        return val.strftime('%Y-%m-%d')
    return val

def sanitize_list(lst):
    return [sanitize_val(x) for x in lst]

app = Flask(__name__)
data_engine = DataEngine()
analysis_engine = AnalysisEngine()

@app.route('/')
def index():
    return render_template('index.html', aliases=COMPANY_ALIASES)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        user_input = request.form.get('ticker', '').upper()
        period = request.form.get('period', '1y').lower()
    else:
        user_input = request.args.get('ticker', '').upper()
        period = request.args.get('period', '1y').lower()

    if not user_input:
        return render_template('index.html', error="Please enter a ticker symbol or company name.", aliases=COMPANY_ALIASES)

    # Map '1m', '3m', '6m' to '1mo', '3mo', '6mo' to prevent yfinance period invalidation
    period_map = {'1m': '1mo', '3m': '3mo', '6m': '6mo'}
    if period in period_map:
        period = period_map[period]
    elif period not in ['1mo', '3mo', '6mo', '1y', '5y']:
        period = '1y'

    df, ticker_sym = data_engine.fetch_stock_data(user_input, period=period)

    if df is not None and not df.empty:
        # Fetch the live quote
        quote = data_engine.fetch_live_quote(ticker_sym)

        # Merge live price into history if available
        if quote and quote['price'] is not None:
            last_date = df.index[-1].date()
            try:
                tz = pytz.timezone(quote['timezone'])
            except Exception:
                tz = pytz.utc
            today = datetime.datetime.now(tz).date()

            # If today's row is in history, update it with live data
            if last_date == today:
                df.loc[df.index[-1], 'Close'] = quote['price']
                if quote['high'] is not None:
                    df.loc[df.index[-1], 'High'] = max(df.loc[df.index[-1], 'High'], quote['high'])
                if quote['low'] is not None:
                    df.loc[df.index[-1], 'Low'] = min(df.loc[df.index[-1], 'Low'], quote['low'])
            elif today > last_date:
                # Add today's row if today is a weekday
                if today.weekday() < 5:
                    new_row = pd.Series({
                        'Open': quote['open'] if quote['open'] is not None else quote['price'],
                        'High': quote['high'] if quote['high'] is not None else quote['price'],
                        'Low': quote['low'] if quote['low'] is not None else quote['price'],
                        'Close': quote['price'],
                        'Volume': quote['volume'] if quote['volume'] is not None else 0
                    }, name=pd.Timestamp(today))
                    df = pd.concat([df, pd.DataFrame([new_row])])

        # Compute technical indicators
        df, latest = analysis_engine.compute_indicators(df)
        if latest is None:
            return render_template('index.html', error=f"Not enough historical data for '{ticker_sym}' to perform reliable AI analysis (minimum 50 days required).", aliases=COMPANY_ALIASES)

        verdict = analysis_engine.get_verdict(latest)

        chart_data = {
            'labels': df.index.strftime('%Y-%m-%d').tolist(),
            'prices': df['Close'].tolist(),
            'sma_fast': df['SMA_Fast'].tolist(),
            'sma_slow': df['SMA_Slow'].tolist(),
            'rsi': df['RSI'].tolist(),
            'macd': df['MACD'].tolist(),
            'macd_signal': df['MACD_Signal'].tolist(),
            'macd_hist': df['MACD_Hist'].tolist(),
            'bb_upper': df['BB_Upper'].tolist(),
            'bb_lower': df['BB_Lower'].tolist()
        }

        prev_close = quote['prev_close'] if quote and quote['prev_close'] is not None else (df['Close'].iloc[-2] if len(df) > 1 else latest['Close'])
        base_currency = quote['currency'] if quote else ("INR" if ticker_sym.endswith(('.NS', '.BO')) else "USD")

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.args.get('json') == 'true':
            clean_latest = {k: sanitize_val(v) for k, v in latest.items()}
            clean_chart_data = {k: (sanitize_list(v) if isinstance(v, list) else sanitize_val(v)) for k, v in chart_data.items()}

            df_clean = []
            for idx, r in df.iterrows():
                rec = {'Date': idx.strftime('%Y-%m-%d')}
                for col in df.columns:
                    rec[col] = sanitize_val(r[col])
                df_clean.append(rec)

            return jsonify({
                'ticker': ticker_sym,
                'latest': clean_latest,
                'prev_close': sanitize_val(prev_close),
                'verdict': verdict,
                'base_currency': base_currency,
                'chart_data': clean_chart_data,
                'quote': quote,
                'period': period,
                'df': df_clean
            })

        return render_template('dashboard.html', 
                               ticker=ticker_sym, 
                               df=df.to_dict('records'), 
                               latest=latest, 
                               prev_close=prev_close,
                               verdict=verdict,
                               base_currency=base_currency,
                               chart_data=chart_data,
                               quote=quote,
                               period=period)
    else:
        return render_template('index.html', error=f"Could not find data for '{user_input}'. Please verify the symbol.", aliases=COMPANY_ALIASES)

@app.route('/api/quote/<ticker>')
def api_quote(ticker):
    quote = data_engine.fetch_live_quote(ticker)
    if quote:
        return jsonify(quote)
    return jsonify({"error": f"Could not fetch quote for {ticker}"}), 404

if __name__ == '__main__':
    app.run(debug=True, port=8080)

from flask import Flask, render_template, request, jsonify
import pandas as pd
from data_engine import DataEngine
from analysis_engine import AnalysisEngine
from config import COMPANY_ALIASES
app = Flask(__name__)
data_engine = DataEngine()
analysis_engine = AnalysisEngine()
@app.route('/')
def index():
    return render_template('index.html', aliases=COMPANY_ALIASES)
@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.form.get('ticker', '').upper()
    if not user_input:
        return render_template('index.html', error="Please enter a ticker symbol or company name.", aliases=COMPANY_ALIASES)
    df, ticker_sym = data_engine.fetch_stock_data(user_input, period='1y')
    if df is not None and not df.empty:
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
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest['Close']
        base_currency = "INR" if ticker_sym.endswith(('.NS', '.BO')) else "USD"
        return render_template('dashboard.html', 
                               ticker=ticker_sym, 
                               df=df.to_dict('records'), 
                               latest=latest, 
                               prev_close=prev_close,
                               verdict=verdict,
                               base_currency=base_currency,
                               chart_data=chart_data)
    else:
        return render_template('index.html', error=f"Could not find data for '{user_input}'. Please verify the symbol.", aliases=COMPANY_ALIASES)
if __name__ == '__main__':
    app.run(debug=True, port=8080)

import pandas as pd
import numpy as np
from config import INDICATOR_WINDOWS
class AnalysisEngine:
    def __init__(self):
        pass
    def compute_indicators(self, df):
        if len(df) < INDICATOR_WINDOWS['SMA_SLOW']:
            return df, None
        df['SMA_Fast'] = df['Close'].rolling(window=INDICATOR_WINDOWS['SMA_FAST']).mean()
        df['SMA_Slow'] = df['Close'].rolling(window=INDICATOR_WINDOWS['SMA_SLOW']).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=INDICATOR_WINDOWS['RSI']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=INDICATOR_WINDOWS['RSI']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['EMA_Fast'] = df['Close'].ewm(span=INDICATOR_WINDOWS['MACD_FAST'], adjust=False).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=INDICATOR_WINDOWS['MACD_SLOW'], adjust=False).mean()
        df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
        df['MACD_Signal'] = df['MACD'].ewm(span=INDICATOR_WINDOWS['MACD_SIGNAL'], adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        return df, df.iloc[-1]
    def get_verdict(self, latest_data):
        if latest_data is None or pd.isna(latest_data['SMA_Slow']) or pd.isna(latest_data['RSI']):
            return {
                "action": "NEUTRAL",
                "reason": "Insufficient data to form a professional opinion.",
                "risk": "Unknown",
                "score": 0
            }
        price = latest_data['Close']
        sma_slow = latest_data['SMA_Slow']
        rsi = latest_data['RSI']
        macd = latest_data['MACD']
        macd_signal = latest_data['MACD_Signal']
        bb_upper = latest_data['BB_Upper']
        bb_lower = latest_data['BB_Lower']
        score = 0
        reasons = []
        if price > sma_slow:
            score += 1
            reasons.append("Bullish Trend: The current price is trading above its 50-day moving average, a strong signal that institutional buyers are holding the stock and maintaining upward momentum.")
        else:
            score -= 1
            reasons.append("Bearish Trend: The stock is currently trapped below its 50-day moving average, indicating selling pressure and a lack of market confidence.")
        if rsi < 35:
            score += 2
            reasons.append(f"Oversold Bounce Expected: The RSI is extremely low ({rsi:.1f}). This statistically means the stock has been excessively dumped by panic sellers and is highly undervalued right now.")
        elif rsi > 65:
            score -= 2
            reasons.append(f"Overbought Warning: The RSI is dangerously high ({rsi:.1f}). The stock is currently overvalued and due for a sudden correction or pullback. Do not buy at the top.")
        else:
            reasons.append(f"Neutral Momentum: The RSI is at {rsi:.1f}, meaning the stock is currently trading at true market value without extreme panic or greed.")
        if macd > macd_signal:
            score += 1
            reasons.append("Positive Trajectory: The MACD line has crossed above the signal line. This specific pattern often precedes a sharp explosive rally in the short term.")
        else:
            score -= 1
            reasons.append("Negative Trajectory: The MACD line is sinking below the signal line. Momentum is fading rapidly, indicating that sellers are taking control.")
        if price <= bb_lower * 1.02:
            score += 1
            reasons.append("Discount Zone: The price has crashed into the Lower Bollinger Band threshold. Historically, entering here offers a high-probability bounce setup with low risk.")
        elif price >= bb_upper * 0.98:
            score -= 1
            reasons.append("Resistance Zone: The price is slamming into the Upper Bollinger Band ceiling. It is statistically exhausted and likely to face heavy selling rejection immediately.")
        if score >= 3:
            action = "STRONG BUY"
            risk = "Low-Medium"
        elif score >= 1:
            action = "BUY"
            risk = "Medium"
        elif score <= -3:
            action = "STRONG SELL"
            risk = "Medium-High"
        elif score <= -1:
            action = "SELL"
            risk = "High"
        else:
            action = "HOLD"
            risk = "Medium"
        return {
            "action": action,
            "reason": "\n".join([f"- {r}" for r in reasons]),
            "risk": risk,
            "score": score
        }

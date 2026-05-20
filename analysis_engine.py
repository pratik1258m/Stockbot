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
                "score": 0,
                "confidence": 0
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

        # Calculate a highly accurate, dynamic confidence score (1.0 to 5.0) based on mathematical indicator signals
        # 1. Trend strength (price deviation from 50-day moving average, capped at 15%)
        trend_dist = abs(price - sma_slow) / (sma_slow if sma_slow > 0 else 1.0)
        trend_factor = min(trend_dist / 0.15, 1.0)
        
        # 2. RSI extreme strength (deviation from 50 neutral point, capped at 30 points)
        rsi_dist = abs(rsi - 50.0)
        rsi_factor = min(rsi_dist / 30.0, 1.0)
        
        # 3. MACD histogram separation (compared to a baseline of 2% of the price)
        macd_hist = latest_data.get('MACD_Hist', 0.0)
        macd_factor = min(abs(macd_hist) / (price * 0.02 if price > 0 else 1.0), 1.0)
        
        # 4. Bollinger Band pressure (price deviation from mid point relative to band width)
        bb_width = bb_upper - bb_lower if (bb_upper - bb_lower) > 0 else 1.0
        bb_mid = (bb_upper + bb_lower) / 2.0
        bb_factor = min(abs(price - bb_mid) / (bb_width / 2.0 if bb_width > 0 else 1.0), 1.0)
        
        # Average the 4 signal strength factors (0.0 to 1.0)
        avg_factor = (trend_factor + rsi_factor + macd_factor + bb_factor) / 4.0
        
        # Map average factor (0 to 1) to a 1.0-5.0 scale with a baseline of 1.5
        confidence = round(1.5 + (avg_factor * 3.5), 1)
        confidence = max(1.0, min(5.0, confidence))

        return {
            "action": action,
            "reason": "\n".join([f"- {r}" for r in reasons]),
            "risk": risk,
            "score": score,
            "confidence": confidence
        }

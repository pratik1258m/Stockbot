"""
analysis_engine.py
==================
Computes all technical indicators and generates the AI verdict.

Indicators implemented:
  - SMA (20, 50)
  - RSI (14)  using Wilder's EWM smoothing — industry-standard formula
  - MACD (12, 26, 9)
  - Bollinger Bands (20, ±2σ)
  - ATR (14)  — Average True Range for volatility/risk quantification

Verdict engine:
  - Weighted signal scoring across 5 factors (trend, momentum,
    convergence, volatility, volume)
  - Confidence score mapped to 1.0 – 5.0 scale from signal strength
  - Linear regression 10-day price projection
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import INDICATOR_WINDOWS, PREDICTION_DAYS


class AnalysisEngine:
    """Stateless engine: receives a price DataFrame, returns enriched DataFrame + verdict."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_indicators(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        """
        Enrich *df* with technical indicator columns.

        Requires at minimum ``INDICATOR_WINDOWS['SMA_SLOW']`` rows (50).
        Returns the enriched DataFrame and the final row as a Series,
        or (df, None) when there is insufficient history.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with a DatetimeIndex and columns
            Open, High, Low, Close, Volume.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series | None]
            (enriched_df, latest_row)  or  (original_df, None).
        """
        if len(df) < INDICATOR_WINDOWS['SMA_SLOW']:
            return df, None

        df = df.copy()

        # ---- Moving Averages -------------------------------------------------
        df['SMA_Fast'] = df['Close'].rolling(window=INDICATOR_WINDOWS['SMA_FAST']).mean()
        df['SMA_Slow'] = df['Close'].rolling(window=INDICATOR_WINDOWS['SMA_SLOW']).mean()

        # ---- RSI — Wilder's EWM smoothing (industry standard) ----------------
        delta = df['Close'].diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        alpha = 1.0 / INDICATOR_WINDOWS['RSI']
        avg_gain = gain.ewm(alpha=alpha, min_periods=INDICATOR_WINDOWS['RSI'], adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, min_periods=INDICATOR_WINDOWS['RSI'], adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

        # ---- MACD ------------------------------------------------------------
        ema_fast = df['Close'].ewm(span=INDICATOR_WINDOWS['MACD_FAST'], adjust=False).mean()
        ema_slow = df['Close'].ewm(span=INDICATOR_WINDOWS['MACD_SLOW'], adjust=False).mean()
        df['MACD']        = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=INDICATOR_WINDOWS['MACD_SIGNAL'], adjust=False).mean()
        df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

        # ---- Bollinger Bands -------------------------------------------------
        bb_window = INDICATOR_WINDOWS['BB']
        df['BB_Mid']   = df['Close'].rolling(window=bb_window).mean()
        df['BB_Std']   = df['Close'].rolling(window=bb_window).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

        # ---- ATR (Average True Range) ----------------------------------------
        atr_period = INDICATOR_WINDOWS['ATR']
        high_low   = df['High'] - df['Low']
        high_pc    = (df['High'] - df['Close'].shift()).abs()
        low_pc     = (df['Low']  - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
        df['ATR']  = true_range.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()

        return df, df.iloc[-1]

    def get_verdict(self, latest: pd.Series) -> dict:
        """
        Produce a trading verdict from the latest indicator values.

        Scoring matrix (weighted):
          ±1.5  Trend       — price vs SMA-50
          ±2.0  Momentum    — RSI extremes
          ±1.5  Convergence — MACD crossover
          ±1.0  Volatility  — Bollinger Band position
          ±1.0  Volume      — current vs average volume

        Parameters
        ----------
        latest : pd.Series
            The last row of an enriched OHLCV DataFrame.

        Returns
        -------
        dict with keys: action, reason, risk, score, confidence, atr.
        """
        if latest is None or pd.isna(latest.get('SMA_Slow')) or pd.isna(latest.get('RSI')):
            return {
                "action": "NEUTRAL",
                "reason": "Insufficient data to form a professional opinion.",
                "risk": "Unknown",
                "score": 0,
                "confidence": 0,
                "atr": None,
            }

        price      = float(latest['Close'])
        sma_slow   = float(latest['SMA_Slow'])
        rsi        = float(latest['RSI'])
        macd       = float(latest['MACD'])
        macd_sig   = float(latest['MACD_Signal'])
        macd_hist  = float(latest['MACD_Hist'])
        bb_upper   = float(latest['BB_Upper'])
        bb_lower   = float(latest['BB_Lower'])
        bb_mid     = float(latest['BB_Mid'])
        volume     = float(latest.get('Volume', 0) or 0)
        avg_vol    = float(latest.get('avg_volume', 0) or 0)   # injected by app.py when available
        atr        = float(latest['ATR']) if not pd.isna(latest.get('ATR')) else None

        score   = 0.0
        reasons = []

        # ---- 1. Trend factor (±1.5) ------------------------------------------
        if price > sma_slow:
            score += 1.5
            reasons.append(
                "Bullish Trend: Price is trading above the 50-day SMA, signalling sustained "
                "institutional buying pressure and positive market momentum."
            )
        else:
            score -= 1.5
            reasons.append(
                "Bearish Trend: Price has fallen below the 50-day SMA, indicating dominant "
                "selling pressure and weakening market confidence."
            )

        # ---- 2. Momentum factor (±2.0) RSI Wilder's --------------------------
        if rsi < 30:
            score += 2.0
            reasons.append(
                f"Oversold Signal: RSI at {rsi:.1f} is below the 30-threshold. "
                "By Wilder's definition this indicates extreme selling exhaustion — "
                "a high-probability mean-reversion bounce setup."
            )
        elif rsi > 70:
            score -= 2.0
            reasons.append(
                f"Overbought Warning: RSI at {rsi:.1f} exceeds the 70-threshold. "
                "The asset is statistically overextended; profit-taking and short-sellers "
                "are likely to generate a near-term correction."
            )
        else:
            reasons.append(
                f"Neutral Momentum: RSI at {rsi:.1f} sits in the fair-value band (30–70), "
                "indicating balanced buying and selling pressure with no extreme signal."
            )

        # ---- 3. Convergence factor (±1.5) MACD --------------------------------
        if macd > macd_sig:
            score += 1.5
            reasons.append(
                "Bullish Crossover: The MACD line has crossed above its signal line. "
                "This pattern — historically one of the most reliable momentum signals — "
                "often precedes a sustained upward rally."
            )
        else:
            score -= 1.5
            reasons.append(
                "Bearish Crossover: The MACD line has dropped below its signal line, "
                "signalling a loss of upward momentum and a potential trend reversal."
            )

        # ---- 4. Volatility factor (±1.0) Bollinger Bands ----------------------
        bb_width = bb_upper - bb_lower if (bb_upper - bb_lower) > 0 else 1.0
        if price <= bb_lower * 1.01:
            score += 1.0
            reasons.append(
                "Discount Zone: Price has compressed to the Lower Bollinger Band. "
                "Statistically, price tends to revert toward the 20-day mean from here, "
                "offering a favourable risk/reward entry."
            )
        elif price >= bb_upper * 0.99:
            score -= 1.0
            reasons.append(
                "Resistance Zone: Price is pressing against the Upper Bollinger Band. "
                "This level acts as dynamic resistance; exhaustion and rejection are common."
            )
        else:
            reasons.append(
                "Band Mid-Range: Price is oscillating within normal Bollinger Band boundaries. "
                "No extreme volatility signal is present at this time."
            )

        # ---- 5. Volume factor (±1.0) ------------------------------------------
        if avg_vol and avg_vol > 0:
            vol_ratio = volume / avg_vol
            if vol_ratio >= 1.5:
                if price > sma_slow:
                    score += 1.0
                    reasons.append(
                        f"Volume Surge (×{vol_ratio:.1f} avg): Above-average volume on an uptrend "
                        "confirms institutional participation and adds conviction to the bullish signal."
                    )
                else:
                    score -= 1.0
                    reasons.append(
                        f"Volume Surge (×{vol_ratio:.1f} avg): Heavy volume on a downtrend "
                        "signals panic selling or distribution — a bearish confirmation."
                    )

        # ---- Map score to action ----------------------------------------------
        score_rounded = round(score, 1)
        if score >= 3.5:
            action, risk = "STRONG BUY",  "Low-Medium"
        elif score >= 1.5:
            action, risk = "BUY",         "Medium"
        elif score <= -3.5:
            action, risk = "STRONG SELL", "High"
        elif score <= -1.5:
            action, risk = "SELL",        "Medium-High"
        else:
            action, risk = "HOLD",        "Medium"

        # ---- ATR risk annotation ---------------------------------------------
        if atr is not None:
            atr_pct = (atr / price) * 100
            reasons.append(
                f"Volatility (ATR-14): Daily price swing is ≈{atr_pct:.2f}% of price "
                f"(₹{atr:.2f} / ${atr:.2f} per day). "
                + ("High volatility — size positions accordingly." if atr_pct > 3
                   else "Moderate volatility — normal market conditions.")
            )

        # ---- Confidence score (1.0 – 5.0) ------------------------------------
        trend_factor = min(abs(price - sma_slow) / (sma_slow * 0.15), 1.0)
        rsi_factor   = min(abs(rsi - 50.0) / 30.0, 1.0)
        macd_factor  = min(abs(macd_hist) / max(abs(price) * 0.02, 1e-9), 1.0)
        bb_factor    = min(abs(price - bb_mid) / (bb_width / 2.0), 1.0)
        avg_factor   = (trend_factor + rsi_factor + macd_factor + bb_factor) / 4.0
        confidence   = round(max(1.0, min(5.0, 1.5 + avg_factor * 3.5)), 1)

        return {
            "action":     action,
            "reason":     "\n".join(f"- {r}" for r in reasons),
            "risk":       risk,
            "score":      score_rounded,
            "confidence": confidence,
            "atr":        round(atr, 4) if atr is not None else None,
        }

    def predict_prices(self, df: pd.DataFrame) -> dict:
        """
        Project future closing prices using Ordinary Least Squares linear regression.

        Only uses the last 90 trading days as the training window to keep the
        trendline relevant to recent momentum rather than long-term history.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with a DatetimeIndex.

        Returns
        -------
        dict with keys:
            labels      — ISO date strings for future days
            predictions — predicted closing prices
            r2          — R² coefficient of determination (model fit quality)
        """
        series = df['Close'].dropna().tail(90)
        if len(series) < 20:
            return {"labels": [], "predictions": [], "r2": None}

        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values

        model = LinearRegression()
        model.fit(X, y)
        r2 = float(model.score(X, y))

        future_X    = np.arange(len(series), len(series) + PREDICTION_DAYS).reshape(-1, 1)
        predictions = model.predict(future_X).tolist()

        # Generate future business day labels
        last_date    = series.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=PREDICTION_DAYS)
        labels       = [d.strftime('%Y-%m-%d') for d in future_dates]

        return {
            "labels":      labels,
            "predictions": [round(p, 4) for p in predictions],
            "r2":          round(r2, 4),
        }

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import BACKTEST_DAYS, INDICATOR_WINDOWS, PREDICTION_DAYS


class AnalysisEngine:
    def compute_indicators(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        if len(df) < INDICATOR_WINDOWS['SMA_SLOW']:
            return df, None

        df = df.copy()

        df['SMA_Fast'] = df['Close'].rolling(window=INDICATOR_WINDOWS['SMA_FAST']).mean()
        df['SMA_Slow'] = df['Close'].rolling(window=INDICATOR_WINDOWS['SMA_SLOW']).mean()

        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        alpha = 1.0 / INDICATOR_WINDOWS['RSI']
        avg_gain = gain.ewm(alpha=alpha, min_periods=INDICATOR_WINDOWS['RSI'], adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, min_periods=INDICATOR_WINDOWS['RSI'], adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        fallback_rsi = pd.Series(np.where(avg_loss == 0, 100.0, 0.0), index=df.index)
        df['RSI'] = df['RSI'].fillna(fallback_rsi)

        ema_fast = df['Close'].ewm(span=INDICATOR_WINDOWS['MACD_FAST'], adjust=False).mean()
        ema_slow = df['Close'].ewm(span=INDICATOR_WINDOWS['MACD_SLOW'], adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=INDICATOR_WINDOWS['MACD_SIGNAL'], adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        bb_window = INDICATOR_WINDOWS['BB']
        df['BB_Mid'] = df['Close'].rolling(window=bb_window).mean()
        df['BB_Std'] = df['Close'].rolling(window=bb_window).std().replace(0, 1e-6)
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

        atr_period = INDICATOR_WINDOWS['ATR']
        if all(c in df.columns for c in ['High', 'Low', 'Close']):
            high_low = df['High'] - df['Low']
            high_pc = (df['High'] - df['Close'].shift()).abs()
            low_pc = (df['Low'] - df['Close'].shift()).abs()
            true_range = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
            df['ATR'] = true_range.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()
        else:
            df['ATR'] = (df['Close'] * 0.02).ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()

        return df, df.iloc[-1]

    def get_verdict(self, latest: pd.Series) -> dict:
        if latest is None or pd.isna(latest.get('SMA_Slow')) or pd.isna(latest.get('RSI')):
            return {
                "action": "NEUTRAL",
                "reason": "Insufficient data to form a professional opinion.",
                "risk": "Unknown",
                "score": 0,
                "confidence": 0,
                "atr": None,
            }

        price = float(latest['Close'])
        sma_slow = float(latest['SMA_Slow'])
        rsi = float(latest['RSI'])
        macd = float(latest['MACD'])
        macd_sig = float(latest['MACD_Signal'])
        macd_hist = float(latest['MACD_Hist'])
        bb_upper = float(latest['BB_Upper'])
        bb_lower = float(latest['BB_Lower'])
        bb_mid = float(latest['BB_Mid'])
        volume = float(latest.get('Volume', 0) or 0)
        avg_vol = float(latest.get('avg_volume', 0) or 0)
        atr = float(latest['ATR']) if not pd.isna(latest.get('ATR')) else None

        score = 0.0
        reasons = []

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

        score_rounded = round(score, 1)
        if score >= 3.5:
            action, risk = "STRONG BUY", "Low-Medium"
        elif score >= 1.5:
            action, risk = "BUY", "Medium"
        elif score <= -3.5:
            action, risk = "STRONG SELL", "High"
        elif score <= -1.5:
            action, risk = "SELL", "Medium-High"
        else:
            action, risk = "HOLD", "Medium"

        if atr is not None:
            atr_pct = (atr / price) * 100
            reasons.append(
                f"Volatility (ATR-14): Daily price swing is ≈{atr_pct:.2f}% of price "
                f"(₹{atr:.2f} / ${atr:.2f} per day). "
                + ("High volatility — size positions accordingly." if atr_pct > 3
                   else "Moderate volatility — normal market conditions.")
            )

        trend_factor = min(abs(price - sma_slow) / (sma_slow * 0.15), 1.0)
        rsi_factor = min(abs(rsi - 50.0) / 30.0, 1.0)
        macd_factor = min(abs(macd_hist) / max(abs(price) * 0.02, 1e-9), 1.0)
        bb_factor = min(abs(price - bb_mid) / (bb_width / 2.0), 1.0)
        avg_factor = (trend_factor + rsi_factor + macd_factor + bb_factor) / 4.0
        confidence = round(max(1.0, min(5.0, 1.5 + avg_factor * 3.5)), 1)

        return {
            "action": action,
            "reason": "\n".join(f"- {r}" for r in reasons),
            "risk": risk,
            "score": score_rounded,
            "confidence": confidence,
            "atr": round(atr, 4) if atr is not None else None,
        }

    def predict_prices(self, df: pd.DataFrame) -> dict:
        series = df['Close'].dropna()
        if len(series) < 25:
            return {
                "labels": [],
                "predictions": [],
                "upper_bounds": [],
                "lower_bounds": [],
                "accuracy_pct": None,
                "mape": None,
                "mae": None,
                "rmse": None,
                "max_error": None,
                "dir_accuracy": None,
                "dir_match": None,
                "backtest": [],
                "forecast": [],
                "model_name": "Adaptive Multi-Horizon Recency-Weighted Trend Model",
            }

        last_close = float(series.iloc[-1])

        def _fit_and_project(hist: pd.Series, steps: int) -> tuple[np.ndarray, float]:
            n_samples = len(hist)
            lookback = min(90, n_samples)
            sub = hist.iloc[-lookback:]
            X_long = np.arange(len(sub)).reshape(-1, 1)
            y_long = sub.values

            decay_long = 0.018
            weights_long = np.exp(-decay_long * np.arange(len(sub))[::-1])
            model_long = LinearRegression().fit(X_long, y_long, sample_weight=weights_long)
            slope_long = float(model_long.coef_[0])

            w_short = min(20, len(sub))
            sub_short = sub.iloc[-w_short:]
            X_short = np.arange(len(sub_short)).reshape(-1, 1)
            weights_short = np.exp(-0.03 * np.arange(len(sub_short))[::-1])
            model_short = LinearRegression().fit(X_short, sub_short.values, sample_weight=weights_short)
            slope_short = float(model_short.coef_[0])

            blended_slope = 0.60 * slope_short + 0.40 * slope_long

            if len(hist) >= 15:
                delta = hist.diff()
                gain = delta.clip(lower=0).ewm(alpha=1.0 / 14, adjust=False).mean()
                loss = (-delta).clip(lower=0).ewm(alpha=1.0 / 14, adjust=False).mean()
                rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-9)
                rsi_val = 100.0 - (100.0 / (1.0 + rs))
                current_p = float(hist.iloc[-1])
                if rsi_val > 70:
                    blended_slope -= 0.0015 * ((rsi_val - 70.0) / 10.0) * current_p
                elif rsi_val < 30:
                    blended_slope += 0.0015 * ((30.0 - rsi_val) / 10.0) * current_p

            damp = 0.97
            damped_steps = np.cumsum([damp ** i for i in range(steps)])
            base_p = float(hist.iloc[-1])
            projected = base_p + blended_slope * damped_steps
            return projected, blended_slope

        bt_window = min(BACKTEST_DAYS, max(5, len(series) // 5))
        train_series = series.iloc[:-bt_window]
        test_actual = series.iloc[-bt_window:]
        test_dates = [d.strftime('%Y-%m-%d') for d in test_actual.index]

        pred_test, _ = _fit_and_project(train_series, bt_window)
        actual_vals = test_actual.values

        diffs = actual_vals - pred_test
        denom = np.where(actual_vals == 0, 1e-6, actual_vals)
        pct_errors = np.abs(diffs / denom) * 100.0
        mape = float(np.mean(pct_errors))
        mae = float(np.mean(np.abs(diffs)))
        rmse = float(np.sqrt(np.mean(diffs ** 2)))
        max_err = float(np.max(np.abs(diffs)))
        accuracy_pct = round(max(0.0, min(100.0, 100.0 - mape)), 1)

        base_tr = float(train_series.iloc[-1])
        act_daily = np.diff(np.insert(actual_vals, 0, base_tr))
        pred_daily = np.diff(np.insert(pred_test, 0, base_tr))
        dir_matches = (act_daily * pred_daily) >= 0
        dir_acc = round(float(np.mean(dir_matches) * 100.0), 1)
        cum_dir_match = bool(np.sign(actual_vals[-1] - base_tr) == np.sign(pred_test[-1] - base_tr))

        backtest_rows = []
        for d_str, act, exp, df_val, pe, dm in zip(test_dates, actual_vals, pred_test, diffs, pct_errors, dir_matches):
            backtest_rows.append({
                'date': d_str,
                'actual': round(float(act), 2),
                'expected': round(float(exp), 2),
                'diff': round(float(df_val), 2),
                'pct_error': round(float(pe), 2),
                'dir_match': bool(dm),
            })

        future_preds, _ = _fit_and_project(series, PREDICTION_DAYS)

        res_std = float(np.std(diffs)) if len(diffs) > 1 else last_close * 0.02
        upper_bounds = [round(float(p + 1.645 * res_std * np.sqrt(i + 1)), 2) for i, p in enumerate(future_preds)]
        lower_bounds = [round(float(p - 1.645 * res_std * np.sqrt(i + 1)), 2) for i, p in enumerate(future_preds)]
        predictions = [round(float(p), 2) for p in future_preds]

        last_date = series.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=PREDICTION_DAYS)
        labels = [d.strftime('%Y-%m-%d') for d in future_dates]

        forecast_rows = []
        for d_str, exp, up, low in zip(labels, predictions, upper_bounds, lower_bounds):
            pct_chg = round(float(((exp - last_close) / last_close) * 100.0), 2)
            forecast_rows.append({
                'date': d_str,
                'expected': exp,
                'upper': up,
                'lower': low,
                'pct_change': pct_chg,
            })

        return {
            "labels": labels,
            "predictions": predictions,
            "upper_bounds": upper_bounds,
            "lower_bounds": lower_bounds,
            "accuracy_pct": accuracy_pct,
            "mape": round(mape, 2),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "max_error": round(max_err, 2),
            "dir_accuracy": dir_acc,
            "dir_match": cum_dir_match,
            "backtest": backtest_rows,
            "forecast": forecast_rows,
            "model_name": "Adaptive Multi-Horizon Recency-Weighted Trend Model",
        }

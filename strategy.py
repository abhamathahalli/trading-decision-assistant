import pandas as pd

def generate_signal(df: pd.DataFrame) -> dict:
    """
    Rule-based, defensible decision support:
    - Trend: fast MA vs slow MA
    - Momentum: RSI
    - Risk note: Volatility (for explanation)
    """
    last = df.dropna().iloc[-1]

    close = float(last["Close"])
    fast = float(last["MA_fast"])
    slow = float(last["MA_slow"])
    rsi = float(last["RSI"])
    vol = float(last["Volatility"]) if "Volatility" in last and pd.notna(last["Volatility"]) else None

    reasons = []
    score = 0

    # Trend
    if fast > slow:
        score += 1
        reasons.append("Trend is UP (fast MA > slow MA).")
    elif fast < slow:
        score -= 1
        reasons.append("Trend is DOWN (fast MA < slow MA).")
    else:
        reasons.append("Trend is NEUTRAL (fast MA ≈ slow MA).")

    # RSI momentum
    if rsi <= 30:
        score += 1
        reasons.append(f"RSI is low ({rsi:.1f}) → oversold pressure (potential rebound).")
    elif rsi >= 70:
        score -= 1
        reasons.append(f"RSI is high ({rsi:.1f}) → overbought pressure (pullback risk).")
    else:
        reasons.append(f"RSI is normal ({rsi:.1f}).")

    # Final decision
    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Risk note using volatility
    if vol is not None:
        if vol >= 0.40:
            reasons.append(f"Volatility is HIGH ({vol:.2f}) → reduce size / widen stops.")
        elif vol <= 0.20:
            reasons.append(f"Volatility is LOW ({vol:.2f}) → cleaner price action.")
        else:
            reasons.append(f"Volatility is MEDIUM ({vol:.2f}).")

    return {
        "signal": signal,
        "score": score,
        "close": close,
        "reasons": reasons
    }

def backtest_lite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick directional evaluation:
    BUY is 'correct' if next day return > 0
    SELL is 'correct' if next day return < 0
    HOLD ignored (None)
    """
    d = df.dropna().copy()
    d["next_ret"] = d["Close"].pct_change().shift(-1)

    def row_signal(row):
        score = 0
        if row["MA_fast"] > row["MA_slow"]:
            score += 1
        elif row["MA_fast"] < row["MA_slow"]:
            score -= 1

        if row["RSI"] <= 30:
            score += 1
        elif row["RSI"] >= 70:
            score -= 1

        if score >= 2:
            return "BUY"
        if score <= -2:
            return "SELL"
        return "HOLD"

    d["signal"] = d.apply(row_signal, axis=1)

    def correct(row):
        if row["signal"] == "BUY":
            return row["next_ret"] > 0
        if row["signal"] == "SELL":
            return row["next_ret"] < 0
        return None

    d["correct"] = d.apply(correct, axis=1)
    return d
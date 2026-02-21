import numpy as np
import pandas as pd

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi_val = 100 - (100 / (1 + rs))

    return rsi_val.clip(0, 100)

def volatility(close: pd.Series, window: int = 20) -> pd.Series:
    returns = close.pct_change()
    return returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)
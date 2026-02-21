import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import clean_ohlcv
from indicators import sma, ema, rsi, volatility
from strategy import generate_signal, backtest_lite

st.set_page_config(page_title="Trading Decision Assistant", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; }
      .badge { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700; }
      .buy { background:#16a34a20; color:#86efac; border:1px solid #16a34a50; }
      .sell { background:#ef444420; color:#fecaca; border:1px solid #ef444450; }
      .hold { background:#f59e0b20; color:#fde68a; border:1px solid #f59e0b50; }
      .card { padding:16px; border-radius:16px; border:1px solid #ffffff1a; background:#0b1220; }
      .small { opacity:0.85; font-size:0.95rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 AI Trading Decision Assistant — Decision Support Prototype")
st.caption("Decision-support prototype (not financial advice). Upload OHLCV data → indicators → signal + reasoning.")
st.markdown("""
This interactive decision-support prototype allows users to upload OHLCV financial data,
compute technical indicators (Moving Averages, RSI, Volatility), and generate interpretable trading signals.

It demonstrates applied data analysis, feature engineering, and decision reasoning in an interactive product.

**Educational purpose only — not financial advice.**
""")
st.markdown("Source code: https://github.com/abhamathahalli/trading-decision-assistant")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
ma_type = st.sidebar.selectbox("Moving Average Type", ["SMA", "EMA"])
fast_window = st.sidebar.slider("Fast MA window", 5, 50, 20, 1)
slow_window = st.sidebar.slider("Slow MA window", 20, 200, 50, 1)
rsi_period = st.sidebar.slider("RSI period", 7, 30, 14, 1)
vol_window = st.sidebar.slider("Volatility window", 10, 60, 20, 1)
show_backtest = st.sidebar.toggle("Show backtest-lite (accuracy thinking)", value=True)

uploaded = st.file_uploader(
    "Upload a CSV with columns: Date, Open, High, Low, Close, Volume",
    type=["csv"]
)

if uploaded is None:
    st.info("Upload a CSV to begin. If you don’t have one, tell me and I’ll give a ready sample CSV template.")
    st.stop()

# Load + clean
try:
    raw = pd.read_csv(uploaded)
    df = clean_ohlcv(raw)
except Exception as e:
    st.error(f"Could not read/clean CSV: {e}")
    st.stop()

# Compute indicators
close = df["Close"]

if ma_type == "SMA":
    df["MA_fast"] = sma(close, fast_window)
    df["MA_slow"] = sma(close, slow_window)
else:
    df["MA_fast"] = ema(close, fast_window)
    df["MA_slow"] = ema(close, slow_window)

df["RSI"] = rsi(close, rsi_period)
df["Volatility"] = volatility(close, vol_window)

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧠 Signal & Reasoning", "🧾 Data Preview"])

with tab1:
    left, right = st.columns([2, 1], gap="large")

    with left:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df["Date"],
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Price"
        ))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA_fast"], name="MA fast", mode="lines"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA_slow"], name="MA slow", mode="lines"))
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        rfig = go.Figure()
        rfig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI", mode="lines"))
        rfig.add_hline(y=70)
        rfig.add_hline(y=30)
        rfig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(rfig, use_container_width=True)

    with right:
        latest = df.dropna().iloc[-1]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📌 Latest snapshot")
        st.write(f"**Date:** {latest['Date'].date()}")
        st.write(f"**Close:** {latest['Close']:.2f}")
        st.write(f"**MA fast ({fast_window}):** {latest['MA_fast']:.2f}")
        st.write(f"**MA slow ({slow_window}):** {latest['MA_slow']:.2f}")
        st.write(f"**RSI ({rsi_period}):** {latest['RSI']:.1f}")
        if pd.notna(latest["Volatility"]):
            st.write(f"**Volatility:** {latest['Volatility']:.2f}")
        st.markdown('<p class="small">Tip: change settings in the sidebar and see signals adapt.</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    try:
        result = generate_signal(df)
    except Exception:
        st.warning("Not enough data yet after indicator windows. Upload more rows (more days).")
        st.stop()

    sig = result["signal"]
    badge_class = "hold"
    if sig == "BUY":
        badge_class = "buy"
    elif sig == "SELL":
        badge_class = "sell"

    st.markdown(f'<span class="badge {badge_class}">Signal: {sig}</span>', unsafe_allow_html=True)
    st.write("")
    st.subheader("Why this signal?")
    for r in result["reasons"]:
        st.write("• " + r)

    st.divider()
    st.subheader("Accuracy thinking (lite)")
    st.caption("Quick evaluation to show prototyping mindset. Not a full trading backtest.")

    if show_backtest:
        bt = backtest_lite(df)
        eval_rows = bt[bt["correct"].notna()].dropna(subset=["next_ret"])
        if len(eval_rows) < 20:
            st.warning("Not enough evaluated rows. Upload more historical rows for better evaluation.")
        else:
            acc = float(eval_rows["correct"].mean())
            st.metric("Directional accuracy (BUY/SELL only)", f"{acc*100:.1f}%")
            st.dataframe(eval_rows[["Date", "Close", "signal", "next_ret", "correct"]].tail(15), use_container_width=True)
    else:
        st.info("Backtest-lite is off in the sidebar.")

with tab3:
    st.subheader("Cleaned data preview")
    st.dataframe(df.tail(200), use_container_width=True)
\# Trading Decision Assistant (Streamlit)



Decision-support prototype (not financial advice).  

Upload OHLCV data → compute indicators (MA, RSI, Volatility) → generate BUY/SELL/HOLD + reasoning.



\## Features

\- Upload CSV (Date, Open, High, Low, Close, Volume)

\- Candlestick chart + moving averages

\- RSI chart with thresholds

\- Signal + explanation

\- Backtest-lite (directional accuracy on next-day returns)



\## Run locally

```bash

pip install -r requirements.txt

streamlit run app.py


from groq import Groq
import json
import numpy as np

def run_backtest(df, fast_ma, slow_ma):
    df = df.copy()
    df['fast'] = df['Close'].rolling(fast_ma).mean()
    df['slow'] = df['Close'].rolling(slow_ma).mean()
    df['signal'] = 0
    df.loc[df['fast'] > df['slow'], 'signal'] = 1
    df.loc[df['fast'] < df['slow'], 'signal'] = -1
    df['returns'] = df['Close'].pct_change()
    df['strat_returns'] = df['signal'].shift(1) * df['returns']

    sharpe = df['strat_returns'].mean() / df['strat_returns'].std() * np.sqrt(252)
    win_rate = (df['strat_returns'] > 0).sum() / (df['strat_returns'] != 0).sum()
    total_return = (1 + df['strat_returns']).prod() - 1

    return {
        "sharpe_ratio": round(float(sharpe), 3),
        "win_rate": round(float(win_rate), 3),
        "total_return_pct": round(float(total_return) * 100, 2),
        "num_trades": int((df['signal'].diff() != 0).sum())
    }

def multi_call_analysis(df, indicators_summary, fast_ma, slow_ma, api_key):
    client = Groq(api_key=api_key)

    # CALL 1: LLM decides whether to run backtest
    call1 = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""You are a quantitative trading analyst.
Given these indicators: {json.dumps(indicators_summary)}
Should we run a backtest with fast_ma={fast_ma}, slow_ma={slow_ma}?
Reply with JSON only, no extra text: {{"run_backtest": true, "reason": "one sentence"}}"""
        }],
        temperature=0.2
    )

    raw = call1.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    decision = json.loads(raw.strip())

    # TOOL EXECUTION: python runs the backtest
    backtest_results = None
    if decision.get("run_backtest"):
        backtest_results = run_backtest(df, fast_ma, slow_ma)

    # CALL 2: LLM interprets backtest results
    call2 = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""You are a senior trading analyst writing a report.
Indicator summary: {json.dumps(indicators_summary)}
Backtest results: {json.dumps(backtest_results)}

Write a structured report with exactly these 3 sections:
1. **Market Conditions** (2 sentences based on indicators)
2. **Strategy Performance** (interpret sharpe ratio, win rate, total return)
3. **Recommendation** (Buy / Hold / Sell with confidence level 1-10)

Be concise and professional."""
        }],
        temperature=0.4
    )

    return {
        "decision": decision,
        "backtest": backtest_results,
        "report": call2.choices[0].message.content
    }
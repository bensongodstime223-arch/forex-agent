"""
Daily multi-pair chart analysis using REAL OHLC candlestick data.

Pairs: EUR/USD, GBP/USD (Alpha Vantage FX_DAILY - real open/high/low/close)
       XAU/USD gold (xaus.com - close/high/low; open approximated as prior close)

For each pair this:
  1. Pulls real daily candles (not just closing price).
  2. Computes trend (20/50-day MA) and momentum (RSI) indicators.
  3. Draws an actual candlestick chart image (saved as PNG).
  4. Prints a plain-English readout.
  5. Optionally sends a phone push notification (text) via ntfy.sh.

IMPORTANT - what this tool is and isn't:
- It is: real chart data and standard technical indicators, informational.
- It is NOT: a proven trading signal. Our earlier backtest of a simple
  news-based rule found no edge across 18 years of real EUR/USD data.
  Chart patterns and indicators shown here are NOT validated to predict
  price moves either - treat this as market awareness, not instruction.
- It does NOT place trades.

Setup:
    pip install requests pandas numpy matplotlib
    Get a free Alpha Vantage key (no card): https://www.alphavantage.co/support/#api-key
    export ALPHAVANTAGE_API_KEY="your_key_here"

Usage:
    python3 daily_chart_analysis.py
"""
import os
import time
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import date, timedelta

ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC")  # optional phone notifications, https://ntfy.sh
CHART_DAYS = 60  # how many recent days to plot/analyze


def send_notification(text: str):
    if not NTFY_TOPIC:
        return
    try:
        requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=text.encode("utf-8"), timeout=10)
    except Exception as e:
        print(f"Notification failed (non-fatal): {e}")


# ---------- Data fetchers ----------

def fetch_forex_ohlc(from_symbol, to_symbol):
    """Real daily OHLC candles via Alpha Vantage FX_DAILY."""
    if not ALPHAVANTAGE_API_KEY:
        raise RuntimeError("Set ALPHAVANTAGE_API_KEY env var first.")
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_DAILY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "outputsize": "compact",  # last ~100 days
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    series = data.get("Time Series FX (Daily)")
    if series is None:
        raise RuntimeError(f"Unexpected API response for {from_symbol}/{to_symbol}: {data}")

    rows = []
    for d, ohlc in series.items():
        rows.append({
            "date": pd.Timestamp(d),
            "open": float(ohlc["1. open"]),
            "high": float(ohlc["2. high"]),
            "low": float(ohlc["3. low"]),
            "close": float(ohlc["4. close"]),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def fetch_gold_ohlc():
    """XAU/USD candles from xaus.com. Open is approximated as prior close
    since this free source provides close/high/low but not open."""
    resp = requests.get("https://xaus.com/api/v1/history", timeout=20)
    resp.raise_for_status()
    data = resp.json()
    rows = [(pd.Timestamp(pt["d"]), pt["c"], pt.get("h", pt["c"]), pt.get("l", pt["c"]))
            for pt in data["points"]]
    df = pd.DataFrame(rows, columns=["date", "close", "high", "low"]).sort_values("date").reset_index(drop=True)
    df["open"] = df["close"].shift(1).fillna(df["close"])
    cutoff = pd.Timestamp(date.today() - timedelta(days=CHART_DAYS + 60))
    return df[df["date"] >= cutoff].reset_index(drop=True)


# ---------- Indicators ----------

def compute_indicators(df):
    df = df.copy()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["recent_low"] = df["close"].rolling(20).min()
    df["recent_high"] = df["close"].rolling(20).max()
    return df


def describe_trend(latest):
    price, ma20, ma50 = latest["close"], latest["ma20"], latest["ma50"]
    if pd.isna(ma50):
        return "Not enough history yet for a 50-day trend read."
    if price > ma20 > ma50:
        return "Above both 20-day and 50-day averages (20 above 50) - short-term uptrend within a longer uptrend."
    elif price < ma20 < ma50:
        return "Below both 20-day and 50-day averages (20 below 50) - short-term downtrend within a longer downtrend."
    elif price > ma20 and ma20 < ma50:
        return "Above the 20-day average, but longer 50-day trend still down - possible bounce or early reversal."
    elif price < ma20 and ma20 > ma50:
        return "Below the 20-day average, but longer 50-day trend still up - possible pullback or early reversal."
    else:
        return "Roughly in line with recent averages - no strong trend either way."


def describe_momentum(latest):
    rsi = latest["rsi"]
    if pd.isna(rsi):
        return "Not enough history yet for an RSI read."
    if rsi >= 70:
        return f"RSI {rsi:.1f} - 'overbought' zone."
    elif rsi <= 30:
        return f"RSI {rsi:.1f} - 'oversold' zone."
    else:
        return f"RSI {rsi:.1f} - neutral zone."


def build_pair_report(name, df):
    latest = df.iloc[-1]
    lines = [
        f"{name}",
        f"  Close: {latest['close']:.5f}   High: {latest['high']:.5f}   Low: {latest['low']:.5f}",
        f"  20-day range: {latest['recent_low']:.5f} - {latest['recent_high']:.5f}",
        f"  Trend: {describe_trend(latest)}",
        f"  Momentum: {describe_momentum(latest)}",
    ]
    return "\n".join(lines)


# ---------- Candlestick chart ----------

def plot_candlestick(name, df, filename):
    plot_df = df.tail(CHART_DAYS).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, row in plot_df.iterrows():
        color = "#26a269" if row["close"] >= row["open"] else "#c01c28"
        ax.plot([i, i], [row["low"], row["high"]], color=color, linewidth=1)
        body_low = min(row["open"], row["close"])
        body_height = abs(row["close"] - row["open"]) or 0.0001
        ax.add_patch(Rectangle((i - 0.3, body_low), 0.6, body_height, color=color))

    if "ma20" in plot_df.columns:
        ax.plot(range(len(plot_df)), plot_df["ma20"], color="#1a5fb4", linewidth=1, label="20-day MA")
    if "ma50" in plot_df.columns:
        ax.plot(range(len(plot_df)), plot_df["ma50"], color="#e5a50a", linewidth=1, label="50-day MA")

    step = max(len(plot_df) // 8, 1)
    ax.set_xticks(range(0, len(plot_df), step))
    ax.set_xticklabels([plot_df["date"].iloc[i].strftime("%m/%d") for i in range(0, len(plot_df), step)], rotation=45)
    ax.set_title(f"{name} - last {CHART_DAYS} days")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)


def append_log(name, df, log_path="daily_analysis_log.csv"):
    row = df.iloc[[-1]].copy()
    row["date"] = row["date"].astype(str)
    row["pair"] = name
    try:
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, row], ignore_index=True).drop_duplicates(subset=["date", "pair"], keep="last")
    except FileNotFoundError:
        combined = row
    combined.to_csv(log_path, index=False)
    return len(combined)


if __name__ == "__main__":
    print("Fetching real OHLC candles for EUR/USD, GBP/USD, XAU/USD...")

    pair_fetchers = {
        "EUR/USD": lambda: fetch_forex_ohlc("EUR", "USD"),
        "GBP/USD": lambda: fetch_forex_ohlc("GBP", "USD"),
    }

    report_sections = [f"DAILY CANDLESTICK ANALYSIS - {date.today()}", "=" * 45]
    log_count = 0

    for name, fetch_fn in pair_fetchers.items():
        try:
            df = fetch_fn()
            df = compute_indicators(df)
            report_sections.append(build_pair_report(name, df))
            report_sections.append("")
            filename = f"chart_{name.replace('/', '')}.png"
            plot_candlestick(name, df, filename)
            print(f"Saved chart: {filename}")
            log_count = append_log(name, df)
            time.sleep(15)  # be polite to the free API rate limit (5 req/min)
        except Exception as e:
            print(f"{name} failed: {e}")

    try:
        gold_df = fetch_gold_ohlc()
        gold_df = compute_indicators(gold_df)
        report_sections.append(build_pair_report("XAU/USD (gold)", gold_df))
        report_sections.append("")
        plot_candlestick("XAU/USD", gold_df, "chart_XAUUSD.png")
        print("Saved chart: chart_XAUUSD.png")
        log_count = append_log("XAU/USD", gold_df)
    except Exception as e:
        print(f"Gold failed: {e}")

    report_sections.append("Reminder: real chart data, but not a proven trading signal.")
    report_text = "\n".join(report_sections)
    print(report_text)
    print(f"\nLog updated - {log_count} total rows across all pairs.")

    send_notification(report_text)
    if NTFY_TOPIC:
        print(f"Notification sent to ntfy.sh/{NTFY_TOPIC}")

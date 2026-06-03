# PROGRAM 3 — REAL-TIME EXTENDED HOURS GAPUP SCANNER (POLYGON VERSION)
# ====================================================================

import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import datetime
import pytz

# -----------------------------
# CONFIG
# -----------------------------
ET = pytz.timezone("US/Eastern")
POLYGON_API_KEY = "pCiLssGqZHjsetUq29dUwcroJJKZriW0"   # <-- PUT YOUR NEW KEY HERE

# -----------------------------
# LOAD WATCHLIST
# -----------------------------
def load_watchlist():
    df = pd.read_csv("watchlist.csv")
    return df[["Yahoo Ticker", "Company Name"]]

watchlist = load_watchlist()

# -----------------------------
# LAST SESSION CLOSE (REGULAR HOURS)
# -----------------------------
def get_last_session_close(ticker):
    stock = yf.Ticker(ticker)
    daily = stock.history(period="10d")
    if daily.empty:
        return None, None

    last_close = daily["Close"].iloc[-1]
    last_date = daily.index[-1]

    last_close_dt = datetime.datetime(
        year=last_date.year,
        month=last_date.month,
        day=last_date.day,
        hour=16,
        minute=0,
        second=0,
        tzinfo=ET
    )
    return last_close, last_close_dt

# -----------------------------
# POLYGON EXTENDED-HOURS DATA
# -----------------------------
def get_polygon_extended_hours(ticker):
    try:
        now = datetime.datetime.now(ET)

        # Determine last session close (4 PM ET)
        last_close_dt = now.replace(hour=16, minute=0, second=0, microsecond=0)
        if now.hour < 16:
            last_close_dt = last_close_dt - datetime.timedelta(days=1)

        start = last_close_dt.isoformat()
        end = now.isoformat()

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/"
            f"{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
        )

        data = requests.get(url).json()

        if "results" not in data:
            return None, None

        candles = data["results"]

        # REAL extended-hours price
        last_price = candles[-1]["c"]

        # REAL extended-hours volume
        ext_vol = sum(c["v"] for c in candles)

        return last_price, ext_vol

    except Exception:
        return None, None

def debug_polygon(ticker):
    now = datetime.datetime.now(ET)

    last_close_dt = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now.hour < 16:
        last_close_dt = last_close_dt - datetime.timedelta(days=1)

    start = last_close_dt.isoformat()
    end = now.isoformat()

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/"
        f"{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )

    st.write("URL:", url)
    st.write("Start:", start)
    st.write("End:", end)

    data = requests.get(url).json()
    st.write("Raw Polygon Response:", data)


# -----------------------------
# GAP% + VOLUME RATIO (POLYGON)
# -----------------------------
def get_extended_gap_and_volume_polygon(ticker):
    last_close, last_close_dt = get_last_session_close(ticker)
    if last_close is None:
        return None, None

    last_price, ext_vol = get_polygon_extended_hours(ticker)
    if last_price is None:
        return None, None

    # GAP %
    gap_pct = ((last_price - last_close) / last_close) * 100

    # Average daily volume
    stock = yf.Ticker(ticker)
    hist_vol = stock.history(period="6mo")
    avg_vol = hist_vol["Volume"].mean()

    vol_ratio = ext_vol / avg_vol if avg_vol > 0 else None

    return gap_pct, vol_ratio

# -----------------------------
# NEWS CATALYST
# -----------------------------
def get_news_catalyst(ticker):
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return None

        for item in news[:5]:
            title = item.get("title", "").lower()

            if any(k in title for k in ["earnings", "beat", "guidance"]):
                return "Earnings / Guidance"
            if any(k in title for k in ["fda", "approval"]):
                return "FDA / Approval"
            if any(k in title for k in ["acquire", "acquisition", "merger"]):
                return "Acquisition / Merger"
            if "upgrade" in title:
                return "Analyst Upgrade"
            if "contract" in title:
                return "New Contract"
            if any(k in title for k in ["launch", "product"]):
                return "Product Launch"

        return None
    except Exception:
        return None

# -----------------------------
# SCORING ENGINE
# -----------------------------
def compute_score(gap_pct, vol_ratio, news):
    score = 0

    # Gap % (25%)
    if gap_pct is not None:
        if gap_pct >= 8:
            score += 25
        elif gap_pct >= 5:
            score += 18
        elif gap_pct >= 3:
            score += 12
        elif gap_pct >= 1:
            score += 6

    # Volume ratio (15%)
    if vol_ratio is not None:
        if vol_ratio >= 10:
            score += 15
        elif vol_ratio >= 5:
            score += 11
        elif vol_ratio >= 2:
            score += 7
        elif vol_ratio >= 1:
            score += 4

    # News (60%)
    if news:
        n = news.lower()
        if "earnings" in n or "guidance" in n or "beat" in n:
            score += 60
        elif "fda" in n or "approval" in n:
            score += 60
        elif "acquisition" in n or "merger" in n:
            score += 55
        elif "upgrade" in n:
            score += 45
        elif "contract" in n:
            score += 40
        elif "launch" in n or "product" in n:
            score += 35

    return min(score, 100)

# -----------------------------
# SIGNAL LABEL
# -----------------------------
def signal_label(score, news):
    if score >= 75 and news:
        return "BUY"
    elif score >= 50:
        return "WATCH"
    else:
        return "IGNORE"

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Program 3 — Real-Time Extended-Hours GAPUP Scanner (Polygon Version)")

rows = []

for _, row in watchlist.iterrows():
    ticker = row["Yahoo Ticker"]
    name = row["Company Name"]

    gap_pct, vol_ratio = get_extended_gap_and_volume_polygon(ticker)
    news = get_news_catalyst(ticker)

    if gap_pct is None:
        continue

    score = compute_score(gap_pct, vol_ratio, news)
    signal = signal_label(score, news)

    rows.append({
        "Ticker": ticker,
        "Name": name,
        "Gap %": round(gap_pct, 2),
        "Vol Ratio": round(vol_ratio, 3) if vol_ratio else None,
        "News": news if news else "None",
        "Score": score,
        "Signal": signal
    })

if rows:
    df = pd.DataFrame(rows)
    df = df.sort_values(by="Score", ascending=False)
    st.dataframe(df)
else:
    st.write("No extended-hours data available right now.")

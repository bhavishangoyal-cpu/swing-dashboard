import streamlit as st
import pandas as pd
import yfinance as yf

# -----------------------------
# LOAD WATCHLIST (WITH COMPANY)
# -----------------------------
def load_watchlist():
    df = pd.read_csv("watchlist.csv")
    return df[["Yahoo Ticker", "Company Name"]]

watchlist = load_watchlist()

# -----------------------------
# FETCH YAHOO PREMARKET DATA
# -----------------------------
def get_premarket_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2d", interval="1m", prepost=True)
        if hist.empty:
            return None

        pre = hist.between_time("04:00", "09:29")
        if pre.empty:
            return None

        last_pre = pre["Close"].iloc[-1]
        yesterday_close = hist["Close"].iloc[0]

        gap_pct = ((last_pre - yesterday_close) / yesterday_close) * 100

        pre_vol = pre["Volume"].sum()
        avg_vol = stock.info.get("averageVolume", 1)
        vol_ratio = pre_vol / avg_vol if avg_vol and avg_vol > 0 else 0

        return gap_pct, vol_ratio
    except:
        return None

# -----------------------------
# NEWS CATALYST DETECTION
# -----------------------------
def get_news_catalyst(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return None

        headline = news[0]["title"].lower()

        if "beat" in headline or "earnings" in headline:
            return "Earnings beat"
        if "fda" in headline or "approval" in headline:
            return "FDA approval"
        if "acquire" in headline or "acquisition" in headline:
            return "Acquisition"
        if "upgrade" in headline:
            return "Analyst upgrade"
        if "contract" in headline:
            return "New contract"
        if "launch" in headline:
            return "Product launch"

        return None
    except:
        return None

# -----------------------------
# SECTOR ETF MAP
# -----------------------------
sector_map = {
    "NVDA": "SMH", "AMD": "SMH", "AVGO": "SMH", "ASML": "SMH", "LRCX": "SMH", "KLAC": "SMH",
    "MSFT": "XLK", "AAPL": "XLK", "META": "XLK", "GOOGL": "XLK", "CRM": "XLK", "NOW": "XLK",
    "PANW": "XLK", "CRWD": "XLK",
    "JPM": "XLF", "GS": "XLF",
    "XOM": "XLE", "CVX": "XLE",
    "CAT": "XLI", "DE": "XLI",
    "LLY": "XLV", "UNH": "XLV", "ABBV": "XLV",
    "SPY": "SPY", "QQQ": "QQQ", "VTI": "VTI", "XLK": "XLK", "SMH": "SMH", "XLF": "XLF", "XLE": "XLE"
}

def get_sector_change(ticker):
    etf = sector_map.get(ticker, "SPY")
    try:
        data = yf.Ticker(etf).history(period="2d", interval="1m", prepost=True)
        if data.empty:
            return 0
        pre = data.between_time("04:00", "09:29")
        if pre.empty:
            return 0
        last_pre = pre["Close"].iloc[-1]
        prev_close = data["Close"].iloc[0]
        return ((last_pre - prev_close) / prev_close) * 100
    except:
        return 0

# -----------------------------
# FUTURES (SPY + NASDAQ VIA QQQ)
# -----------------------------
def get_futures():
    try:
        spy = yf.Ticker("SPY").history(period="2d", interval="1m", prepost=True)
        qqq = yf.Ticker("QQQ").history(period="2d", interval="1m", prepost=True)

        if spy.empty or qqq.empty:
            return 0, 0

        spy_pre = spy.between_time("04:00", "09:29")
        qqq_pre = qqq.between_time("04:00", "09:29")

        if spy_pre.empty or qqq_pre.empty:
            return 0, 0

        spy_pct = ((spy_pre["Close"].iloc[-1] - spy["Close"].iloc[0]) / spy["Close"].iloc[0]) * 100
        qqq_pct = ((qqq_pre["Close"].iloc[-1] - qqq["Close"].iloc[0]) / qqq["Close"].iloc[0]) * 100

        return spy_pct, qqq_pct
    except:
        return 0, 0

spy_fut_pct, qqq_fut_pct = get_futures()

# -----------------------------
# SCORING ENGINE
# -----------------------------
def compute_score(gap_pct, vol_ratio, news, sector_pct, futures_pct):
    score = 0

    # Gap (25%)
    if gap_pct >= 8:
        score += 25
    elif gap_pct >= 5:
        score += 20
    elif gap_pct >= 3:
        score += 15
    elif gap_pct >= 2:
        score += 10
    elif gap_pct >= 1:
        score += 5

    # Volume (20%)
    if vol_ratio >= 10:
        score += 20
    elif vol_ratio >= 5:
        score += 15
    elif vol_ratio >= 2:
        score += 10
    elif vol_ratio >= 1:
        score += 5

    # News (25%)
    if news:
        n = news.lower()
        if "earnings" in n or "beat" in n:
            score += 25
        elif "fda" in n:
            score += 25
        elif "acquisition" in n:
            score += 25
        elif "guidance" in n:
            score += 20
        elif "upgrade" in n:
            score += 15
        elif "contract" in n:
            score += 15
        elif "launch" in n:
            score += 10

    # Sector (10%)
    if sector_pct >= 2:
        score += 10
    elif sector_pct >= 1:
        score += 5

    # Futures (10%)
    if futures_pct >= 1:
        score += 10
    elif futures_pct >= 0.5:
        score += 5

    return score

# -----------------------------
# LABEL HELPERS
# -----------------------------
def buy_signal(score, news, vol_ratio):
    if score >= 70 and news != "None" and vol_ratio >= 2:
        return "BUY"
    elif score >= 50:
        return "WATCH"
    else:
        return "IGNORE"

def sector_strength_label(sector_pct):
    if sector_pct >= 2:
        return "Strong"
    elif sector_pct >= 1:
        return "Moderate"
    else:
        return "Weak"

# -----------------------------
# BUILD DATAFRAME
# -----------------------------
rows = []

for _, row in watchlist.iterrows():
    ticker = row["Yahoo Ticker"]
    company = row["Company Name"]

    pm = get_premarket_data(ticker)
    if pm is None:
        continue

    gap_pct, vol_ratio = pm
    news = get_news_catalyst(ticker)
    sector_pct = get_sector_change(ticker)

    # Use SPY futures for all; you could switch to qqq_fut_pct for tech if you want
    score = compute_score(gap_pct, vol_ratio, news, sector_pct, spy_fut_pct)

    rows.append({
        "Ticker": ticker,
        "Company": company,
        "Gap %": gap_pct,
        "Volume Ratio": vol_ratio,
        "News": news or "None",
        "Sector %": sector_pct,
        "SPY Futures %": spy_fut_pct,
        "Score": score
    })

df = pd.DataFrame(rows)

if not df.empty:
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)

    df["Buy Signal"] = df.apply(
        lambda r: buy_signal(r["Score"], r["News"], r["Volume Ratio"]),
        axis=1
    )
    df["Sector Strength"] = df["Sector %"].apply(sector_strength_label)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Pre‑Market Probability Scanner", layout="wide")
st.title("📈 Pre‑Market Probability Scanner (0–100 Score)")

if df.empty:
    st.warning("No valid pre‑market data found for your watchlist.")
else:
    def color_score(val):
        if val >= 75:
            return "background-color:#145A32;color:white;"
        elif val >= 50:
            return "background-color:#F1C40F;color:black;"
        else:
            return "background-color:#922B21;color:white;"

    styled = (
        df.style
        .applymap(color_score, subset=["Score"])
        .format({
            "Gap %": "{:.2f}%",
            "Volume Ratio": "{:.1f}x",
            "Sector %": "{:.2f}%",
            "SPY Futures %": "{:.2f}%",
            "Score": "{:.0f}"
        })
    )

    st.dataframe(styled, use_container_width=True)

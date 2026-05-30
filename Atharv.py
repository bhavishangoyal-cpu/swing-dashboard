# ============================
# 📘 DEFINITIONS DROPDOWN
# ============================

st.markdown("---")
st.header("📘 Indicator & Strategy Definitions")

definitions = {
    # ---------- STRATEGY 1 ----------
    "EMA20": "Short‑term trend line. Price above EMA20 = bullish momentum.",
    "EMA50": "Medium‑term trend line. EMA20 > EMA50 = strong uptrend.",
    "MACD": "Momentum indicator. MACD > Signal = bullish momentum.",
    "MACD Histogram": "Shows momentum strength. Rising histogram = increasing momentum.",
    "RSI": "Measures overbought/oversold. Below 65 = safe for pullback buys.",
    "ATR%": "Volatility percentage. Higher ATR% = bigger expected moves.",
    "Volume Ratio": "Current volume ÷ average volume. >1.2 = strong interest.",
    "Support Level": "Recent swing low where buyers stepped in.",
    "Distance to Support": "How far price is from support (in %). Closer = safer.",
    "Pullback to EMA20": "Price dips to EMA20 inside an uptrend. A high‑probability swing entry.",
    "Breakout Above 5‑Day High": "Price breaks above recent resistance. Momentum entry.",
    "Bullish Divergence": "Price makes lower low but RSI makes higher low. Signals reversal.",
    "ADX": "Trend strength indicator. >25 = strong trend.",
    "SPY Market Context": "Checks if overall market is bullish.",
    "VIX Condition": "Measures market fear. Low VIX = safe. High VIX = avoid trading.",

    # ---------- STRATEGY 2 ----------
    "52‑Week High": "Highest price in the last 252 trading days.",
    "Drop %": "How far the stock is below its 52‑week high.",
    "Drop Buckets": "Groups stocks by drop %: <10%, 10–20%, 20–30%, 30–40%, 40–50%.",

    # ---------- STRATEGY 3 ----------
    "Yesterday Close": "Last price at 4:00 PM previous day.",
    "Premarket Close": "Last traded price before today's market opens (9:29 AM).",
    "Gap %": "(Premarket Close − Yesterday Close) ÷ Yesterday Close × 100.",
    "Premarket Volume": "Total volume traded between 4:00 AM and 9:29 AM.",
    "News Catalyst": "Type of news causing the gap: earnings, FDA, acquisition, upgrade, contract, launch.",
    "Sector Strength": "Checks if the stock’s sector ETF is green.",
    "SPY Futures %": "Shows overall market direction before open.",
    "BUY / WATCH / IGNORE": "Signal based on score and indicators.",
    "Score (0–100)": "Probability score based on gap%, volume ratio, news, sector, futures, trend, volatility."
}

selected_term = st.selectbox("Select a term to view its definition:", list(definitions.keys()))

st.info(definitions[selected_term])

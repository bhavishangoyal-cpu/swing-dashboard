import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import time
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# ==============================================================================
# GLOBAL CONFIG & WATCHLIST HELPERS (Shared across all strategies)
# ==============================================================================
WATCHLIST_PATH = "watchlist.csv"
REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}


def load_watchlist(path: str = WATCHLIST_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        df = pd.DataFrame(columns=["Yahoo Ticker", "Company Name"])
        df.to_csv(path, index=False)
        return df
    df = pd.read_csv(path)
    df["Yahoo Ticker"] = df["Yahoo Ticker"].astype(str).str.strip().str.upper()
    if "Company Name" not in df.columns:
        df["Company Name"] = df["Yahoo Ticker"]
    df["Company Name"] = df["Company Name"].astype(str).str.strip()
    return df[["Yahoo Ticker", "Company Name"]]


def save_watchlist(watchlist_tickers, path: str = WATCHLIST_PATH):
    existing_df = load_watchlist(path)
    rows = []
    for ticker in watchlist_tickers:
        ticker = ticker.strip().upper()
        match = existing_df[existing_df["Yahoo Ticker"] == ticker]
        name = match["Company Name"].values[0] if not match.empty else ticker
        rows.append({"Yahoo Ticker": ticker, "Company Name": name})
    pd.DataFrame(rows).to_csv(path, index=False)


def load_ticker_to_name(path: str = WATCHLIST_PATH) -> dict:
    df = load_watchlist(path)
    return dict(zip(df["Yahoo Ticker"], df["Company Name"]))


def safe_history(ticker: str, interval: str, period: str = "7d") -> pd.DataFrame | None:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval, prepost=False)
    except:
        return None
    if df is None or df.empty or not REQUIRED_COLS.issubset(df.columns):
        return None
    df = df.dropna(subset=list(REQUIRED_COLS))
    return df if not df.empty else None


# ==============================================================================
# STRATEGY 1: ATHARV SWING SCANNER METRICS & LOGIC
# ==============================================================================
def rsi_s1(series: pd.Series, period: int = 14) -> pd.Series:
    series = series.astype(float)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_s1_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI14"] = rsi_s1(df["Close"], 14)
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    return df


def get_s1_last_values(df: pd.DataFrame):
    return {
        "o_last": float(df["Open"].iloc[-1]),
        "h_last": float(df["High"].iloc[-1]),
        "l_last": float(df["Low"].iloc[-1]),
        "c_last": float(df["Close"].iloc[-1]),
        "v_last": float(df["Volume"].iloc[-1]),
        "o_prev": float(df["Open"].iloc[-2]),
        "h_prev": float(df["High"].iloc[-2]),
        "l_prev": float(df["Low"].iloc[-2]),
        "c_prev": float(df["Close"].iloc[-2]),
        "ema20_last": float(df["EMA20"].iloc[-1]),
        "ema50_last": float(df["EMA50"].iloc[-1]),
        "rsi_last": float(df["RSI14"].iloc[-1]),
        "rsi_prev": float(df["RSI14"].iloc[-2]),
        "vol_avg20": float(df["VolAvg20"].iloc[-1]),
    }


def detect_support_resistance(df: pd.DataFrame, lookback: int = 40):
    recent = df.tail(lookback)
    support = recent['Low'].rolling(5).min().iloc[-1]
    resistance = recent['High'].rolling(5).max().iloc[-1]
    return support, resistance


def is_near_level(price: float, level: float, tolerance: float = 0.02) -> bool:
    if level <= 0: return False
    return abs(price - level) / level <= tolerance


def bullish_engulfing(o_prev, c_prev, o_last, c_last) -> bool:
    return (c_prev < o_prev) and (c_last > o_last) and (c_last >= o_prev) and (o_last <= c_prev)


def bearish_engulfing(o_prev, c_prev, o_last, c_last) -> bool:
    return (c_prev > o_prev) and (c_last < o_last) and (c_last <= o_prev) and (o_last >= c_prev)


def hammer(o_last, h_last, l_last, c_last) -> bool:
    body = abs(c_last - o_last)
    rng = h_last - l_last
    if rng == 0: return False
    lower_shadow = min(o_last, c_last) - l_last
    return (lower_shadow > 2 * body) and (body / rng < 0.4)


def volume_strong(v_last: float, vol_avg20: float, factor: float = 1.2) -> bool:
    if np.isnan(vol_avg20) or vol_avg20 == 0: return False
    return v_last > factor * vol_avg20


def analyze_ticker_s1(ticker: str, interval: str) -> dict:
    data = safe_history(ticker, interval=interval, period="7d")
    if data is None or len(data) < 60: return {"ticker": ticker, "status": "NO_DATA", "interval": interval}
    df = compute_s1_indicators(data).dropna()
    if len(df) < 30: return {"ticker": ticker, "status": "NO_DATA", "interval": interval}

    vals = get_s1_last_values(df)
    trend = "UP" if (vals["c_last"] > vals["ema20_last"] > vals["ema50_last"]) else "DOWN"
    support, resistance = detect_support_resistance(df)
    near_support = is_near_level(vals["c_last"], support, tolerance=0.02)
    near_resistance = is_near_level(vals["c_last"], resistance, tolerance=0.02)
    rsi_up, rsi_down = vals["rsi_last"] > vals["rsi_prev"], vals["rsi_last"] < vals["rsi_prev"]
    bull_eng = bullish_engulfing(vals["o_prev"], vals["c_prev"], vals["o_last"], vals["c_last"])
    bear_eng = bearish_engulfing(vals["o_prev"], vals["c_prev"], vals["o_last"], vals["c_last"])
    is_hammer = hammer(vals["o_last"], vals["h_last"], vals["l_last"], vals["c_last"])
    vol_ok = volume_strong(vals["v_last"], vals["vol_avg20"], factor=1.1)

    long_score = sum(
        [trend == "UP", near_support, (28 <= vals["rsi_last"] <= 55) and rsi_up, bull_eng or is_hammer, vol_ok])
    short_score = sum([trend == "DOWN", near_resistance, (55 <= vals["rsi_last"] <= 75) and rsi_down, bear_eng, vol_ok])

    if long_score >= 4:
        decision = "BUY"
    elif short_score >= 4:
        decision = "SHORT"
    elif (trend in ["UP", "DOWN"]) and (near_support or near_resistance):
        decision = "WAIT"
    else:
        decision = "NO ENTER"

    confirmed = ((decision == "BUY") and (trend == "UP") and (long_score >= 3) and (
                28 <= vals["rsi_last"] <= 55) and near_support and (vals["v_last"] > 1.1 * vals["vol_avg20"]))
    confirmed_label = "CONFIRMED" if confirmed else "NOT CONFIRMED"

    strength = 0
    if vals["c_last"] > vals["ema20_last"] > vals["ema50_last"]:
        strength += 2
    elif vals["ema20_last"] > vals["ema50_last"]:
        strength += 1
    if 35 <= vals["rsi_last"] <= 50:
        strength += 2
    elif 28 <= vals["rsi_last"] <= 55:
        strength += 1
    dist = abs(vals["c_last"] - support) / support
    if dist <= 0.01:
        strength += 2
    elif dist <= 0.02:
        strength += 1
    if vals["v_last"] > 1.3 * vals["vol_avg20"]:
        strength += 2
    elif vals["v_last"] > 1.1 * vals["vol_avg20"]:
        strength += 1
    if bull_eng or is_hammer: strength += 1
    if confirmed: strength += 1

    return {
        "ticker": ticker, "status": "OK", "interval": interval, "trend": trend, "close": vals["c_last"],
        "support": support, "resistance": resistance, "rsi": vals["rsi_last"], "long_score": long_score,
        "short_score": short_score, "decision": decision, "confirmed": confirmed_label, "strength": strength,
    }


def decision_color(val: str) -> str:
    if val == "BUY":
        return "background-color:#2ECC71;color:black;"
    elif val == "WAIT":
        return "background-color:#F1C40F;color:black;"
    elif val == "NO ENTER":
        return "background-color:#E74C3C;color:white;"
    return ""


def confirmed_color(val: str) -> str:
    return "background-color:#27AE60;color:white;" if val == "CONFIRMED" else "background-color:#AAB7B8;color:black;"


# ==============================================================================
# STRATEGY 2: GOEL'S STRATEGY METRICS & LOGIC
# ==============================================================================
def add_basic_indicators_s2(df):
    if df.empty or len(df) < 50: return df
    df = df.copy()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    return df


def add_extra_indicators_s2(df):
    if df.empty or len(df) < 50: return df
    df = df.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-Cp'] = (df['High'] - df['Close'].shift()).abs()
    df['L-Cp'] = (df['Low'] - df['Close'].shift()).abs()
    df['TR'] = df[['H-L', 'H-Cp', 'L-Cp']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
    df['Volume_MA20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20'].replace(0, np.nan)
    df['Swing_Low_20'] = df['Low'].rolling(20).min()
    df['Distance_to_EMA20'] = ((df['Close'] - df['EMA20']) / df['EMA20'].replace(0, np.nan)) * 100
    df['MACD_H'] = df['MACD'] - df['MACD_Signal']
    return df.bfill().ffill().fillna(0)


@st.cache_data(ttl=600)
def check_market_context():
    try:
        spy_data = yf.download('SPY', period='3mo', progress=False)
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = [col[0] for col in spy_data.columns]
        spy_data['EMA50'] = spy_data['Close'].ewm(span=50, adjust=False).mean()
        is_bullish = float(spy_data['Close'].iloc[-1]) > float(spy_data['EMA50'].iloc[-1])
        return is_bullish, f"SPY: {'Bullish ✓' if is_bullish else 'Bearish ✗'}"
    except:
        return True, "Market check unavailable"


@st.cache_data(ttl=300)
def check_vix_level():
    try:
        vix_data = yf.download('^VIX', period='1d', progress=False)
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = [col[0] for col in vix_data.columns]
        vix_value = float(vix_data['Close'].iloc[-1])
        if vix_value < 15:
            return vix_value, "Very Calm", "Too slow, hard to make 2-3%", False, "blue"
        elif vix_value < 20:
            return vix_value, "Normal/Healthy", "IDEAL for swing trading ✅", True, "green"
        elif vix_value < 30:
            return vix_value, "Worried", "Getting dangerous - Only take STRONG BUY", "selective", "orange"
        elif vix_value < 40:
            return vix_value, "Scared", "Very risky, avoid trading", False, "red"
        else:
            return vix_value, "Panic", "Market crash, STOP trading", False, "darkred"
    except:
        return None, "Unavailable", "Could not fetch VIX", None, "gray"


def enhanced_signal_s2(df):
    if df.empty or len(df) < 50: return "HOLD", "Insufficient data", "N/A"
    try:
        last = df.iloc[-1]
        vix_value, vix_category, vix_condition, trading_ok, vix_color = check_vix_level()
        trend_up = float(last['EMA20']) > float(last['EMA50'])
        volume_good = float(last.get('Volume_Ratio', 0)) > 1.2
        volatility_good = float(last.get('ATR_Pct', 0)) > 1.5
        market_bullish, market_msg = check_market_context()

        pullback_to_ema = -5.0 <= float(last.get('Distance_to_EMA20', 0)) <= 0.0
        rsi_not_hot = float(last['RSI']) < 65
        pullback_count = sum([trend_up, pullback_to_ema, rsi_not_hot, volume_good, volatility_good])

        breakout = float(last['Close']) > float(df['High'].tail(5).max())
        rsi_not_extreme = float(last['RSI']) < 75
        breakout_count = sum([trend_up, breakout, volume_good, volatility_good, rsi_not_extreme])

        vix_str = f"{vix_value:.1f}" if vix_value else "N/A"

        if vix_value and vix_value > 30:
            if pullback_count >= 4 and market_bullish: return "STRONG BUY (Pullback)", f"Perfect pullback setup | VIX: {vix_str}", vix_str
            if breakout_count >= 4 and market_bullish: return "STRONG BUY (Breakout)", f"Perfect breakout setup | VIX: {vix_str}", vix_str
            return "HOLD", f"⚠️ VIX too high ({vix_str}), SKIP trading", vix_str

        if pullback_count >= 4 and market_bullish:
            return "STRONG BUY (Pullback)", f"Perfect pullback setup | VIX: {vix_str}", vix_str
        elif pullback_count >= 3:
            return "POTENTIAL BUY (Pullback)", f"Good pullback near EMA20 | VIX: {vix_str}", vix_str
        if breakout_count >= 4 and market_bullish:
            return "STRONG BUY (Breakout)", f"Perfect breakout setup | VIX: {vix_str}", vix_str
        elif breakout_count >= 3:
            return "POTENTIAL BUY (Breakout)", f"Breakout above 5-day high | VIX: {vix_str}", vix_str

        return "HOLD", "No setup identified", vix_str
    except Exception as e:
        return "ERROR", str(e), "N/A"


@st.cache_data(ttl=300)
def fetch_safe_s2(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        return df
    except:
        return pd.DataFrame()


def score_signal_s2(row):
    score = 0
    sig = str(row.get('Enhanced Signal', ''))
    if 'STRONG BUY' in sig:
        score += 50
    elif 'POTENTIAL BUY' in sig:
        score += 35
    vr = row.get('Volume Ratio', 0)
    if isinstance(vr, (int, float)) and vr > 1.5: score += 15
    atr_pct = row.get('ATR %', 0)
    if isinstance(atr_pct, (int, float)) and atr_pct > 2: score += 10
    return min(score, 100)


def rating_from_score_s2(score):
    return "A+ (High Prob)" if score >= 85 else "A (Strong Setup)" if score >= 70 else "B (Decent)" if score >= 55 else "C (Weak)"


# ==============================================================================
# STRATEGY 3: 52-WEEK HIGH/LOW BREAKOUT & PROXIMITY LOGIC
# ==============================================================================
@st.cache_data(ttl=600)
def fetch_safe_s3(ticker):
    try:
        df = yf.download(ticker, period="14mo", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        return df
    except:
        return pd.DataFrame()


def analyze_ticker_s3(ticker, df, proximity_pct):
    if df.empty or len(df) < 252: return None
    df_52w = df.tail(252)
    current_price = float(df_52w['Close'].iloc[-1])
    high_52w = float(df_52w['High'].max())
    low_52w = float(df_52w['Low'].min())

    dist_from_high = ((high_52w - current_price) / high_52w) * 100
    dist_from_low = ((current_price - low_52w) / low_52w) * 100
    range_position = ((current_price - low_52w) / (high_52w - low_52w + 1e-9)) * 100

    avg_volume = df_52w['Volume'].mean()
    current_volume = df_52w['Volume'].iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

    if current_price >= high_52w or dist_from_high <= 0:
        condition = "🚀 52W HIGH BREAKOUT"
    elif dist_from_high <= proximity_pct:
        condition = "🔥 NEAR 52W HIGH"
    elif current_price <= low_52w or dist_from_low <= 0:
        condition = "💥 52W LOW BREAKDOWN"
    elif dist_from_low <= proximity_pct:
        condition = "⚠️ NEAR 52W LOW"
    else:
        condition = "🌐 MIDDLE RANGE"

    return {
        "Ticker": ticker, "Condition": condition, "Current Price": round(current_price, 2),
        "52W High": round(high_52w, 2), "52W Low": round(low_52w, 2), "Dist to High (%)": round(dist_from_high, 2),
        "Dist to Low (%)": round(dist_from_low, 2), "Range Position (%)": round(range_position, 1),
        "Volume Ratio": round(volume_ratio, 2)
    }


def s3_condition_color(val):
    if "HIGH BREAKOUT" in val:
        return "background-color:#2ECC71;color:black;font-weight:bold;"
    elif "NEAR 52W HIGH" in val:
        return "background-color:#D4EFDF;color:black;"
    elif "LOW BREAKDOWN" in val:
        return "background-color:#E74C3C;color:white;font-weight:bold;"
    elif "NEAR 52W LOW" in val:
        return "background-color:#FADBD8;color:black;"
    return ""


# ==============================================================================
# MASTER LAYOUT & APP ROUTING
# ==============================================================================
st.set_page_config(page_title="Master Trading Suite", layout="wide")
st.title("🎛️ Master Strategy & Scanning Interface")

tab1, tab2, tab3 = st.tabs(
    ["🎯 Atharv Swing Scanner (5m/15m)", "📈 Goel's Swing Strategy", "📊 52-Week High/Low Strategy"])

# ------------------------------------------------------------------------------
# TAB 1: ATHARV SWING SCANNER
# ------------------------------------------------------------------------------
with tab1:
    st.header("Atharv Swing Trading Scanner (5m + 15m)")
    watchlist_df = load_watchlist()
    tickers = watchlist_df["Yahoo Ticker"].tolist()
    st.write(f"Active Watchlist: **{len(tickers)}** tokens loaded from `watchlist.csv`")

    intervals = ["5m", "15m"]

    for interval in intervals:
        st.subheader(f"⏱️ Interval Timeframe: {interval}")
        rows = []

        with st.spinner(f"Scanning market tickers for {interval}..."):
            for _, row in watchlist_df.iterrows():
                t = row["Yahoo Ticker"]
                company_name = row["Company Name"]
                res = analyze_ticker_s1(t, interval)

                if res["status"] != "OK":
                    rows.append(
                        {"Ticker": t, "Company (Ticker)": f"{company_name} ({t})", "Decision": "NO ENTER", "Trend": "",
                         "Close": "", "Support": "", "Resistance": "", "RSI": "", "LongScore": "", "ShortScore": "",
                         "CONFIRMED": "", "Strength": 0})
                else:
                    rows.append({
                        "Ticker": res["ticker"], "Company (Ticker)": f"{company_name} ({res['ticker']})",
                        "Decision": res["decision"], "Trend": res["trend"], "Close": round(res["close"], 2),
                        "Support": round(res["support"], 2),
                        "Resistance": round(res["resistance"], 2), "RSI": round(res["rsi"], 1),
                        "LongScore": res["long_score"], "ShortScore": res["short_score"], "CONFIRMED": res["confirmed"],
                        "Strength": res["strength"]
                    })

        if rows:
            df_res = pd.DataFrame(rows)
            order = {"BUY": 0, "WAIT": 1, "NO ENTER": 2}
            df_res["Rank"] = df_res["Decision"].map(order).fillna(3)
            df_res = df_res.sort_values(["Rank", "Strength", "LongScore"], ascending=[True, False, False]).drop(
                columns=["Rank"])

            styled = df_res.style.apply(lambda col: [decision_color(v) for v in col], subset=["Decision"]).apply(
                lambda col: [confirmed_color(v) for v in col], subset=["CONFIRMED"])
            st.dataframe(styled, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: GOEL'S STRATEGY
# ------------------------------------------------------------------------------
with tab2:
    st.header("📈 Goel's Market Context & Pullback Strategy")
    st_autorefresh(interval=180000, key="goel_auto_refresh")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        spy_bullish, spy_msg = check_market_context()
        st.success(f"📈 {spy_msg}") if spy_bullish else st.error(f"📉 {spy_msg}")
    with m_col2:
        vix_value, vix_category, vix_condition, trading_ok, vix_color = check_vix_level()
        if vix_value:
            st.info(f"📊 **VIX Matrix:** {vix_value:.2f} | Category: {vix_category} ({vix_condition})")

    st.markdown("### 📋 Watchlist Controls")
    w_df = load_watchlist()
    current_tickers = w_df["Yahoo Ticker"].tolist()

    col_add, col_btn = st.columns([4, 1])
    with col_add:
        new_ticker = st.text_input("Add New Symbol to Watchlist:", "", key="input_add_s2").upper().strip()
    with col_btn:
        st.write(" ")
        if st.button("➕ Add Ticker", key="btn_add_s2") and new_ticker:
            if new_ticker not in current_tickers:
                current_tickers.append(new_ticker)
                save_watchlist(current_tickers)
                st.success(f"Symbol {new_ticker} saved!")
                st.rerun()
            else:
                st.warning("Ticker already in watchlist.")

    if current_tickers:
        if st.button("🔍 Execute Goel Strategy Scanning Engine", key="btn_run_s2"):
            results = []
            progress_bar = st.progress(0)

            for idx, ticker in enumerate(current_tickers):
                df = fetch_safe_s2(ticker)
                if df.empty: continue
                df = add_basic_indicators_s2(df)
                df = add_extra_indicators_s2(df)
                signal, reason, vix_level = enhanced_signal_s2(df)
                last = df.iloc[-1]

                results.append({
                    'Ticker': ticker, 'Company Name': load_ticker_to_name().get(ticker, "-"), 'Enhanced Signal': signal,
                    'Current Price': round(last['Close'], 2), 'Volume Ratio': round(last['Volume_Ratio'], 2),
                    'ATR %': round(last['ATR_Pct'], 2), 'Reason': reason, 'VIX Level': vix_level
                })
                progress_bar.progress((idx + 1) / len(current_tickers))

            progress_bar.empty()

            if results:
                df_results = pd.DataFrame(results)
                df_results['Score'] = df_results.apply(score_signal_s2, axis=1)
                df_results['Rating'] = df_results['Score'].apply(rating_from_score_s2)
                df_results = df_results.sort_values('Score', ascending=False)

                strong = df_results[df_results['Enhanced Signal'].str.contains('STRONG BUY', na=False)]
                potential = df_results[df_results['Enhanced Signal'].str.contains('POTENTIAL BUY', na=False)]
                holds = df_results[df_results['Enhanced Signal'] == 'HOLD']

                st.subheader(f"🚀 Strong Buy Triggers ({len(strong)})")
                st.dataframe(strong, use_container_width=True)

                st.subheader(f"💡 Potential Watch Triggers ({len(potential)})")
                st.dataframe(potential, use_container_width=True)

                with st.expander("Show Inactive / Hold Assets"):
                    st.dataframe(holds, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: 52-WEEK HIGH/LOW STRATEGY
# ------------------------------------------------------------------------------
with tab3:
    st.header("📊 52-Week Range Proximity & Breakout Tracker")
    proximity_slider = st.slider("Select Proximity Threshold Percentage (%)", min_value=1.0, max_value=10.0, value=3.0,
                                 step=0.5, key="s3_slider")

    watchlist_s3 = load_watchlist()
    tickers_s3 = watchlist_s3["Yahoo Ticker"].tolist()

    if tickers_s3:
        if st.button("🔍 Run 52-Week Tracker Engine", key="btn_run_s3"):
            s3_rows = []
            with st.spinner("Analyzing 52-Week positions across watchlist..."):
                for ticker in tickers_s3:
                    df_historical = fetch_safe_s3(ticker)
                    metrics = analyze_ticker_s3(ticker, df_historical, proximity_slider)
                    if metrics:
                        s3_rows.append(metrics)

            if s3_rows:
                df_s3 = pd.DataFrame(s3_rows)

                s3_order = {"🚀 52W HIGH BREAKOUT": 0, "🔥 NEAR 52W HIGH": 1, "⚠️ NEAR 52W LOW": 2,
                            "💥 52W LOW BREAKDOWN": 3, "🌐 MIDDLE RANGE": 4}
                df_s3["Rank"] = df_s3["Condition"].map(s3_order).fillna(5)
                df_s3 = df_s3.sort_values(["Rank", "Volume Ratio"], ascending=[True, False]).drop(columns=["Rank"])

                styled_s3 = df_s3.style.apply(lambda col: [s3_condition_color(v) for v in col], subset=["Condition"])
                st.dataframe(styled_s3, use_container_width=True)
            else:
                st.warning("No sufficient structural historical data available for selected tickers.")
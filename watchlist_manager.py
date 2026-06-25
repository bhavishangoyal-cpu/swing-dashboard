import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import time
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# ==============================================================================
# 1. GLOBAL APP CONFIGURATION & INITIALIZATION (Must be at the absolute top)
# ==============================================================================
st.set_page_config(page_title="Master Trading Suite", layout="wide")
st.title("🎛️ Master Strategy & Scanning Interface")

# Setup Global Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Atharv Swing Scanner (5m/15m)",
    "📈 Goel's Swing Strategy",
    "📊 52-Week High/Low Strategy",
    "⚡ Enhanced Scanner"
])

WATCHLIST_PATH = "watchlist.csv"
REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}


# ==============================================================================
# 2. SHARED DATA UTILITIES & ENGINE CONFIGURATIONS
# ==============================================================================
def shared_load_watchlist(path: str = WATCHLIST_PATH) -> pd.DataFrame:
    """Unified CSV Watchlist Loader used by all strategies"""
    if not os.path.exists(path):
        # Create an empty sample if it doesn't exist
        df = pd.DataFrame(columns=["Yahoo Ticker", "Company Name"])
        df.to_csv(path, index=False)
        return df
    df = pd.read_csv(path)
    df["Yahoo Ticker"] = df["Yahoo Ticker"].astype(str).str.strip().str.upper()
    if "Company Name" not in df.columns:
        df["Company Name"] = df["Yahoo Ticker"]
    else:
        df["Company Name"] = df["Company Name"].astype(str).str.strip()
    return df[["Yahoo Ticker", "Company Name"]]


def shared_save_watchlist(watchlist_tickers, path: str = WATCHLIST_PATH):
    """Saves updated tickers back to CSV while safeguarding format"""
    ticker_to_name = {}
    if os.path.exists(path):
        try:
            old_df = pd.read_csv(path)
            for _, r in old_df.iterrows():
                t = str(r.get('Yahoo Ticker', '')).strip().upper()
                n = str(r.get('Company Name', r.get('Yahoo Ticker', ''))).strip()
                if t: ticker_to_name[t] = n
        except:
            pass

    updated_rows = []
    for t in watchlist_tickers:
        t_clean = t.strip().upper()
        if t_clean:
            updated_rows.append({
                "Yahoo Ticker": t_clean,
                "Company Name": ticker_to_name.get(t_clean, t_clean)
            })
    pd.DataFrame(updated_rows).to_csv(path, index=False)


# ==============================================================================
# 3. STRATEGY 1: ATHARV SWING SCANNER UTILITIES
# ==============================================================================
def s1_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series = series.astype(float)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def s1_compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI14"] = s1_rsi(df["Close"], 14)
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    return df


def s1_get_last_values(df: pd.DataFrame):
    return {
        "o_last": float(df["Open"].iloc[-1]), "h_last": float(df["High"].iloc[-1]),
        "l_last": float(df["Low"].iloc[-1]), "c_last": float(df["Close"].iloc[-1]),
        "v_last": float(df["Volume"].iloc[-1]), "o_prev": float(df["Open"].iloc[-2]),
        "h_prev": float(df["High"].iloc[-2]), "l_prev": float(df["Low"].iloc[-2]),
        "c_prev": float(df["Close"].iloc[-2]), "ema20_last": float(df["EMA20"].iloc[-1]),
        "ema50_last": float(df["EMA50"].iloc[-1]), "rsi_last": float(df["RSI14"].iloc[-1]),
        "rsi_prev": float(df["RSI14"].iloc[-2]), "vol_avg20": float(df["VolAvg20"].iloc[-1]),
        # NEW PROPERTY ADDED HERE:
        "day_range_pos": (float(df["Close"].iloc[-1]) - float(df["Low"].iloc[-1])) / (float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1]) + 1e-9),
    }


def s1_safe_history(ticker: str, interval: str, period: str = "7d") -> pd.DataFrame | None:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval, prepost=False)
        if df is None or df.empty or not REQUIRED_COLS.issubset(df.columns):
            return None
        return df.dropna(subset=list(REQUIRED_COLS))
    except:
        return None


def s1_analyze_ticker(ticker: str, interval: str) -> dict:
    data = s1_safe_history(ticker, interval=interval, period="7d")
    if data is None or len(data) < 60:
        return {"ticker": ticker, "status": "NO_DATA", "interval": interval}

    df = s1_compute_indicators(data).dropna()
    if len(df) < 30:
        return {"ticker": ticker, "status": "NO_DATA", "interval": interval}

    vals = s1_get_last_values(df)
    trend = "UP" if (vals["c_last"] > vals["ema20_last"] > vals["ema50_last"]) else "DOWN"

    recent = df.tail(40)
    support = recent['Low'].rolling(5).min().iloc[-1]
    resistance = recent['High'].rolling(5).max().iloc[-1]

    near_support = (support > 0) and (abs(vals["c_last"] - support) / support <= 0.02)
    near_resistance = (resistance > 0) and (abs(vals["c_last"] - resistance) / resistance <= 0.02)

    rsi_up = vals["rsi_last"] > vals["rsi_prev"]
    rsi_down = vals["rsi_last"] < vals["rsi_prev"]

    bull_eng = (vals["c_prev"] < vals["o_prev"]) and (vals["c_last"] > vals["o_last"]) and (
                vals["c_last"] >= vals["o_prev"]) and (vals["o_last"] <= vals["c_prev"])
    bear_eng = (vals["c_prev"] > vals["o_prev"]) and (vals["c_last"] < vals["o_last"]) and (
                vals["c_last"] <= vals["o_prev"]) and (vals["o_last"] >= vals["c_prev"])

    body = abs(vals["c_last"] - vals["o_last"])
    rng = vals["h_last"] - vals["l_last"]
    is_hammer = (rng > 0) and ((min(vals["o_last"], vals["c_last"]) - vals["l_last"]) > 2 * body) and (body / rng < 0.4)

    # Stricter volume criteria updated to 1.5
    vol_ok = (not np.isnan(vals["vol_avg20"])) and (vals["vol_avg20"] > 0) and (
                vals["v_last"] > 1.5 * vals["vol_avg20"])

    # Long Score with new Top 25% Day Range Position feature
    long_score = sum([
        trend == "UP",
        near_support,
        (28 <= vals["rsi_last"] <= 70) and rsi_up,
        bull_eng or is_hammer,
        vol_ok,
        vals["day_range_pos"] >= 0.75
    ])

    short_score = sum([trend == "DOWN", near_resistance, (55 <= vals["rsi_last"] <= 75) and rsi_down, bear_eng, vol_ok])

    if long_score >= 4:
        decision = "BUY"
    elif short_score >= 4:
        decision = "SHORT"
    elif (trend in ["UP", "DOWN"]) and (near_support or near_resistance):
        decision = "WAIT"
    else:
        decision = "NO ENTER"

    confirmed = (decision == "BUY") and (trend == "UP") and (long_score >= 3) and (
                28 <= vals["rsi_last"] <= 70) and near_support and vol_ok
    confirmed_label = "CONFIRMED" if confirmed else "NOT CONFIRMED"

    strength = 0
    if vals["c_last"] > vals["ema20_last"] > vals["ema50_last"]:
        strength += 2
    elif vals["ema20_last"] > vals["ema50_last"]:
        strength += 1
    if 35 <= vals["rsi_last"] <= 50:
        strength += 2
    elif 28 <= vals["rsi_last"] <= 70:
        strength += 1

    if support > 0:
        dist = abs(vals["c_last"] - support) / support
        if dist <= 0.01:
            strength += 2
        elif dist <= 0.02:
            strength += 1

    if (not np.isnan(vals["vol_avg20"])) and vals["vol_avg20"] > 0:
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
        "range_pos": round(vals["day_range_pos"] * 100, 1),  # Added parameter track back
    }

def s1_decision_color(val: str) -> str:
    if val == "BUY":
        return "background-color:#2ECC71;color:black;"
    elif val == "WAIT":
        return "background-color:#F1C40F;color:black;"
    elif val == "NO ENTER":
        return "background-color:#E74C3C;color:white;"
    return ""


def s1_confirmed_color(val: str) -> str:
    return "background-color:#27AE60;color:white;" if val == "CONFIRMED" else "background-color:#AAB7B8;color:black;"


# ==============================================================================
# 4. STRATEGY 2: GOEL'S SWING STRATEGY UTILITIES
# ==============================================================================
def s2_add_basic_indicators(df):
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


def s2_add_extra_indicators(df):
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
def s2_check_market_context():
    try:
        spy_data = yf.download('SPY', period='3mo', progress=False)
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = [col[0] for col in spy_data.columns]
        spy_data['EMA50'] = spy_data['Close'].ewm(span=50, adjust=False).mean()
        is_bullish = spy_data['Close'].iloc[-1] > spy_data['EMA50'].iloc[-1]
        return is_bullish, f"SPY: {'Bullish ✓' if is_bullish else 'Bearish ✗'}"
    except:
        return True, "Market check unavailable"


@st.cache_data(ttl=300)
def s2_check_vix_level():
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


def s2_enhanced_signal(df):
    if df.empty or len(df) < 50: return "HOLD", "Insufficient data", "N/A"
    try:
        last = df.iloc[-1]
        vix_value, vix_category, _, _, _ = s2_check_vix_level()
        trend_up = float(last['EMA20']) > float(last['EMA50'])
        volume_good = float(last.get('Volume_Ratio', 0)) > 1.2
        volatility_good = float(last.get('ATR_Pct', 0)) > 1.5
        market_bullish, _ = s2_check_market_context()

        pullback_to_ema = -5.0 <= float(last.get('Distance_to_EMA20', 0)) <= 0.0
        rsi_not_hot = float(last['RSI']) < 65
        pullback_count = sum([trend_up, pullback_to_ema, rsi_not_hot, volume_good, volatility_good])

        breakout = float(last['Close']) > float(df['High'].tail(5).max())
        rsi_not_extreme = float(last['RSI']) < 75
        breakout_count = sum([trend_up, breakout, volume_good, volatility_good, rsi_not_extreme])

        vix_str = f"{vix_value:.1f}" if vix_value else "N/A"

        if vix_value and vix_value > 30:
            if pullback_count >= 4 and market_bullish:
                return "STRONG BUY (Pullback)", f"Perfect pullback | VIX: {vix_str}", vix_str
            elif breakout_count >= 4 and market_bullish:
                return "STRONG BUY (Breakout)", f"Perfect breakout | VIX: {vix_str}", vix_str
            return "HOLD", f"⚠️ VIX High ({vix_str}), SKIP | PB:{pullback_count}/5 BC:{breakout_count}/5", vix_str

        if pullback_count >= 4 and market_bullish:
            return "STRONG BUY (Pullback)", f"Perfect pullback | VIX: {vix_str}", vix_str
        elif pullback_count >= 3:
            return "POTENTIAL BUY (Pullback)", f"Good pullback near EMA20 | VIX: {vix_str}", vix_str
        if breakout_count >= 4 and market_bullish:
            return "STRONG BUY (Breakout)", f"Perfect breakout | VIX: {vix_str}", vix_str
        elif breakout_count >= 3:
            return "POTENTIAL BUY (Breakout)", f"Breakout above 5-day high | VIX: {vix_str}", vix_str
        if pullback_count >= 2 or breakout_count >= 3: return "MODERATE BUY", f"Weak setup | VIX: {vix_str}", vix_str
        return "HOLD", f"No setup | PB:{pullback_count}/5 BC:{breakout_count}/5", vix_str
    except Exception as e:
        return "ERROR", str(e), "N/A"


def s2_fetch_safe(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
        if df.empty or not {'Open', 'High', 'Low', 'Close', 'Volume'}.issubset(df.columns): return pd.DataFrame()
        df = df.reset_index()
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date']); df = df.set_index('Date')
        return df.dropna()
    except:
        return pd.DataFrame()


def s2_score_signal(row):
    score = 0
    sig = str(row.get('Enhanced Signal', ''))
    if 'STRONG BUY' in sig:
        score += 50
    elif 'POTENTIAL BUY' in sig:
        score += 35
    elif 'MODERATE BUY' in sig:
        score += 20
    vr = row.get('Volume Ratio', 0)
    if isinstance(vr, (int, float)):
        if vr > 2.0:
            score += 20
        elif vr > 1.5:
            score += 15
        elif vr > 1.2:
            score += 10
    atr_pct = row.get('ATR %', 0)
    if isinstance(atr_pct, (int, float)):
        if atr_pct > 3:
            score += 15
        elif atr_pct > 2:
            score += 10
        elif atr_pct > 1.5:
            score += 5
    return min(score, 100)


def s2_rating_from_score(score):
    if score >= 85:
        return "A+ (High Probability)"
    elif score >= 70:
        return "A (Strong Setup)"
    elif score >= 55:
        return "B (Decent)"
    return "C (Weak)"


# ==============================================================================
# 5. STRATEGY 3: 52-WEEK DROP ANALYZER UTILITIES
# ==============================================================================
@st.cache_data(show_spinner=False)
def s3_download_single_ticker(ticker: str):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df is None or df.empty: return None
        df.index = pd.to_datetime(df.index)
        if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
        return df
    except:
        return None


# ==============================================================================
# 6. STRATEGY 4: ATHARV ENHANCED SCANNER UTILITIES
# ==============================================================================
def s4_get_last_values(df: pd.DataFrame):
    return {
        "o_last": float(df["Open"].iloc[-1]), "h_last": float(df["High"].iloc[-1]),
        "l_last": float(df["Low"].iloc[-1]), "c_last": float(df["Close"].iloc[-1]),
        "v_last": float(df["Volume"].iloc[-1]), "o_prev": float(df["Open"].iloc[-2]),
        "h_prev": float(df["High"].iloc[-2]), "l_prev": float(df["Low"].iloc[-2]),
        "c_prev": float(df["Close"].iloc[-2]), "ema20_last": float(df["EMA20"].iloc[-1]),
        "ema50_last": float(df["EMA50"].iloc[-1]), "rsi_last": float(df["RSI14"].iloc[-1]),
        "rsi_prev": float(df["RSI14"].iloc[-2]), "vol_avg20": float(df["VolAvg20"].iloc[-1]),
        "day_range_pos": (float(df["Close"].iloc[-1]) - float(df["Low"].iloc[-1])) / (
                    float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1]) + 1e-9),
    }


def s4_analyze_ticker(ticker: str, interval: str) -> dict:
    data = s1_safe_history(ticker, interval=interval, period="7d")  # Safely reuses shared loader logic
    if data is None or len(data) < 60:
        return {"ticker": ticker, "status": "NO_DATA", "interval": interval}

    df = s1_compute_indicators(data).dropna()
    if len(df) < 30:
        return {"ticker": ticker, "status": "NO_DATA", "interval": interval}

    vals = s4_get_last_values(df)
    trend = "UP" if (vals["c_last"] > vals["ema20_last"] > vals["ema50_last"]) else "DOWN"

    recent = df.tail(40)
    support = recent['Low'].rolling(5).min().iloc[-1]
    resistance = recent['High'].rolling(5).max().iloc[-1]

    near_support = (support > 0) and (abs(vals["c_last"] - support) / support <= 0.02)
    near_resistance = (resistance > 0) and (abs(vals["c_last"] - resistance) / resistance <= 0.02)

    rsi_up = vals["rsi_last"] > vals["rsi_prev"]
    rsi_down = vals["rsi_last"] < vals["rsi_prev"]

    bull_eng = (vals["c_prev"] < vals["o_prev"]) and (vals["c_last"] > vals["o_last"]) and (
                vals["c_last"] >= vals["o_prev"]) and (vals["o_last"] <= vals["c_prev"])
    bear_eng = (vals["c_prev"] > vals["o_prev"]) and (vals["c_last"] < vals["o_last"]) and (
                vals["c_last"] <= vals["o_prev"]) and (vals["o_last"] >= vals["c_prev"])

    body = abs(vals["c_last"] - vals["o_last"])
    rng = vals["h_last"] - vals["l_last"]
    is_hammer = (rng > 0) and ((min(vals["o_last"], vals["c_last"]) - vals["l_last"]) > 2 * body) and (body / rng < 0.4)

    vol_ok = (not np.isnan(vals["vol_avg20"])) and (vals["vol_avg20"] > 0) and (
                vals["v_last"] > 1.5 * vals["vol_avg20"])

    long_score = sum([
        trend == "UP", near_support,
        (28 <= vals["rsi_last"] <= 70) and rsi_up,
        bull_eng or is_hammer, vol_ok,
        vals["day_range_pos"] >= 0.75
    ])
    short_score = sum([trend == "DOWN", near_resistance, (55 <= vals["rsi_last"] <= 75) and rsi_down, bear_eng, vol_ok])

    if long_score >= 4:
        decision = "BUY"
    elif short_score >= 4:
        decision = "SHORT"
    elif (trend in ["UP", "DOWN"]) and (near_support or near_resistance):
        decision = "WAIT"
    else:
        decision = "NO ENTER"

    confirmed = (decision == "BUY") and (trend == "UP") and (long_score >= 3) and (
                28 <= vals["rsi_last"] <= 70) and near_support and vol_ok
    confirmed_label = "CONFIRMED" if confirmed else "NOT CONFIRMED"

    strength = 0
    if vals["c_last"] > vals["ema20_last"] > vals["ema50_last"]:
        strength += 2
    elif vals["ema20_last"] > vals["ema50_last"]:
        strength += 1
    if 35 <= vals["rsi_last"] <= 50:
        strength += 2
    elif 28 <= vals["rsi_last"] <= 70:
        strength += 1

    if support > 0:
        dist = abs(vals["c_last"] - support) / support
        if dist <= 0.01:
            strength += 2
        elif dist <= 0.02:
            strength += 1

    if (not np.isnan(vals["vol_avg20"])) and vals["vol_avg20"] > 0:
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
        "range_pos": round(vals["day_range_pos"] * 100, 1),
    }

# ==============================================================================
# ==============================================================================
# TAB EXECUTION BLOCKS (Saves components inside isolated spaces)
# ==============================================================================
# ==============================================================================

# ==============================================================================
# 🎯 TAB 1: ATHARV SWING SCANNER
# ==============================================================================
with tab1:
    st.header("Atharv Swing Trading Scanner (5m + 15m)")

    watchlist_df = shared_load_watchlist()
    tickers = watchlist_df["Yahoo Ticker"].tolist()
    st.write(f"Loaded **{len(tickers)}** tickers from internal configuration repository.")

    intervals = ["5m", "15m"]

    if st.button("Run Scanner", key="btn_run_atharv_scanner"):
        for interval in intervals:
            st.subheader(f"Interval Target: {interval}")
            rows = []
            for _, row in watchlist_df.iterrows():
                t = row["Yahoo Ticker"]
                company_name = row["Company Name"]
                res = s1_analyze_ticker(t, interval)

                if res["status"] != "OK":
                    rows.append({
                        "Ticker": t, "Company (Ticker)": f"{company_name} ({t})", "Decision": "NO ENTER",
                        "Trend": "", "Close": "", "Support": "", "Resistance": "", "RSI": "",
                        "LongScore": "", "ShortScore": "", "CONFIRMED": "", "Strength": 0
                    })
                else:
                    rows.append({
                        "Ticker": res["ticker"], "Company (Ticker)": f"{company_name} ({res['ticker']})",
                        "Decision": res["decision"], "Trend": res["trend"], "Close": round(res["close"], 2),
                        "Support": round(res["support"], 2), "Resistance": round(res["resistance"], 2),
                        "RSI": round(res["rsi"], 1), "LongScore": res["long_score"],
                        "ShortScore": res["short_score"], "CONFIRMED": res["confirmed"], "Strength": res["strength"],
                    })

            df_res = pd.DataFrame(rows)
            df_res["Rank"] = df_res["Decision"].map({"BUY": 0, "WAIT": 1, "NO ENTER": 2}).fillna(3)
            df_res["ConfRank"] = df_res["CONFIRMED"].map({"CONFIRMED": 0, "NOT CONFIRMED": 1}).fillna(2)

            df_res = df_res.sort_values(
                ["Rank", "ConfRank", "Strength", "LongScore"],
                ascending=[True, True, False, False]
            ).drop(columns=["Rank", "ConfRank"])

            styled = df_res.style.apply(
                lambda col: [s1_decision_color(v) for v in col], subset=["Decision"]
            ).apply(
                lambda col: [s1_confirmed_color(v) for v in col], subset=["CONFIRMED"]
            )
            st.dataframe(styled, use_container_width=True)

# ==============================================================================
# 📈 TAB 2: GOEL'S SWING STRATEGY
# ==============================================================================
with tab2:
    st.header("📈 Goel's Swing Strategy Engine")

    # Auto-refresh context bound exclusively to Tab 2
    st_autorefresh(interval=180000, key="refresh_goel_tab")

    s2_watchlist = shared_load_watchlist()["Yahoo Ticker"].tolist()
    ticker_to_name = dict(zip(shared_load_watchlist()["Yahoo Ticker"], shared_load_watchlist()["Company Name"]))

    # Market Health Header
    st.markdown("### Market Environmental Conditions")
    col1, col2 = st.columns(2)
    with col1:
        spy_bullish, spy_msg = s2_check_market_context()
        if spy_bullish:
            st.success(f"📈 {spy_msg}")
        else:
            st.error(f"📉 {spy_msg}")
    with col2:
        vix_val, vix_cat, vix_cond, trading_ok, vix_color = s2_check_vix_level()
        if vix_val:
            st.info(f"VIX: {vix_val:.2f} ({vix_cat}) — {vix_cond}")
        else:
            st.warning("VIX Matrix configuration data currently unavailable")

    st.markdown("---")

    # Inline Entry System
    col_add_1, col_add_2 = st.columns([4, 1])
    with col_add_1:
        new_ticker = st.text_input("Enter ticker to add to master engine list:", "", key="txt_add_goel").upper().strip()
    with col_add_2:
        if st.button("Add Ticker Asset", key="btn_add_goel"):
            if new_ticker and new_ticker not in s2_watchlist:
                s2_watchlist.append(new_ticker)
                shared_save_watchlist(s2_watchlist)
                st.success(f"Added {new_ticker}!")
                st.rerun()

    # Calculation loop
    if s2_watchlist:
        st.info(f"Analyzing metrics for {len(s2_watchlist)} tracked parameters...")
        results_s2 = []

        for ticker in s2_watchlist:
            df_s2 = s2_fetch_safe(ticker)
            if df_s2.empty:
                results_s2.append({
                    'Ticker': ticker, 'Company Name': ticker_to_name.get(ticker, ticker), 'Enhanced Signal': 'NO DATA',
                    'Current Price': '-', 'Entry Price': '-', 'Stop Loss (ATR)': '-', 'Target 2%': '-',
                    'Target 3%': '-',
                    'Volume Ratio': '-', 'ATR %': '-', 'Score': 0, 'Reason': 'No data', 'VIX Level': '-'
                })
                continue

            df_s2 = s2_add_basic_indicators(df_s2)
            df_s2 = s2_add_extra_indicators(df_s2)
            signal, reason, vix_level = s2_enhanced_signal(df_s2)
            last_row = df_s2.iloc[-1]

            close_p = float(last_row['Close']) if 'Close' in last_row else 0
            atr_v = float(last_row['ATR']) if 'ATR' in last_row else 0
            v_ratio = float(last_row['Volume_Ratio']) if 'Volume_Ratio' in last_row else 0
            a_pct = float(last_row['ATR_Pct']) if 'ATR_Pct' in last_row else 0

            results_s2.append({
                'Ticker': ticker, 'Company Name': ticker_to_name.get(ticker, ticker), 'Enhanced Signal': signal,
                'Current Price': round(close_p, 2) if close_p else "-",
                'Entry Price': round(close_p, 2) if close_p else "-",
                'Stop Loss (ATR)': round(close_p - (atr_v * 1.5), 2) if close_p and atr_v else "-",
                'Target 2%': round(close_p * 1.02, 2) if close_p else "-",
                'Target 3%': round(close_p * 1.03, 2) if close_p else "-",
                'Volume Ratio': round(v_ratio, 2) if v_ratio else "-", 'ATR %': round(a_pct, 2) if a_pct else "-",
                'Score': 0, 'Reason': reason, 'VIX Level': vix_level
            })

        df_results_s2 = pd.DataFrame(results_s2)
        df_results_s2['Score'] = df_results_s2.apply(s2_score_signal, axis=1)
        df_results_s2['Rating'] = df_results_s2['Score'].apply(s2_rating_from_score)

        # Filters
        strong_df = df_results_s2[df_results_s2['Enhanced Signal'].str.contains('STRONG BUY', na=False)].sort_values(
            'Score', ascending=False)
        potential_df = df_results_s2[
            df_results_s2['Enhanced Signal'].str.contains('POTENTIAL BUY', na=False)].sort_values('Score',
                                                                                                  ascending=False)
        moderate_df = df_results_s2[
            df_results_s2['Enhanced Signal'].str.contains('MODERATE BUY', na=False)].sort_values('Score',
                                                                                                 ascending=False)

        st.subheader(f"🚀 STRONG BUY SETUPS ({len(strong_df)})")
        if not strong_df.empty:
            st.dataframe(strong_df[['Ticker', 'Company Name', 'Enhanced Signal', 'Current Price', 'Entry Price',
                                    'Stop Loss (ATR)', 'Target 2%', 'Target 3%', 'VIX Level', 'Score', 'Rating']],
                         use_container_width=True)
        else:
            st.info("No strong conditions detected.")

        with st.expander(f"💡 POTENTIAL SIGNALS ({len(potential_df)})"):
            if not potential_df.empty:
                st.dataframe(potential_df[['Ticker', 'Company Name', 'Enhanced Signal', 'Current Price', 'Entry Price',
                                           'Stop Loss (ATR)', 'Target 2%', 'Target 3%', 'VIX Level', 'Score',
                                           'Rating']], use_container_width=True)
            else:
                st.info("No elements inside bucket.")

        with st.expander(f"⚠️ MODERATE ALPHA ALERTS ({len(moderate_df)})"):
            if not moderate_df.empty:
                st.dataframe(moderate_df[['Ticker', 'Company Name', 'Enhanced Signal', 'Current Price', 'Entry Price',
                                          'Stop Loss (ATR)', 'Target 2%', 'Target 3%', 'VIX Level', 'Score', 'Rating']],
                             use_container_width=True)
            else:
                st.info("No elements inside bucket.")

# ==============================================================================
# 📊 TAB 3: 52-WEEK HIGH DROP ANALYZER
# ==============================================================================
with tab3:
    st.header("📉 52-Week High Drop Analyzer Overview")

    CATEGORIES = {"<10%": (0.0, 10.0), "10-20%": (10.0, 20.0), "20-30%": (10.0, 30.0), "30-40%": (30.0, 40.0),
                  "40-50%": (40.0, 50.0001)}
    CATEGORY_COLORS = {"<10%": "#d9f0ff", "10-20%": "#b3e6ff", "20-30%": "#ffeeb3", "30-40%": "#ffd6cc",
                       "40-50%": "#ffb3b3"}

    watch_data_s3 = shared_load_watchlist()
    tickers_s3 = watch_data_s3['Yahoo Ticker'].tolist()
    companies_s3 = dict(zip(watch_data_s3['Yahoo Ticker'], watch_data_s3['Company Name']))

    if len(tickers_s3) == 0:
        st.warning("Please add data parameters into watchlist.csv to initialize.")
    else:
        st.info("Computing mathematical rolling matrices against 252-day baseline...")
        collected_s3 = []

        for ticker in tickers_s3:
            df_hist = s3_download_single_ticker(ticker)
            if df_hist is not None and not df_hist.empty and "Close" in df_hist.columns:
                high_s = df_hist["High"] if "High" in df_hist.columns else df_hist["Close"]
                h52_series = high_s.rolling(window=252, min_periods=1).max()

                latest_c = float(df_hist["Close"].iloc[-1])
                latest_h = float(h52_series.iloc[-1])

                if latest_h > 0:
                    drop_p = ((latest_h - latest_c) / latest_h) * 100
                    collected_s3.append({
                        "Company Name": companies_s3.get(ticker, ticker), "Ticker": ticker,
                        "Current Price": round(latest_c, 2), "52-Week High": round(latest_h, 2),
                        "Drop %": round(drop_p, 2)
                    })

        if len(collected_s3) == 0:
            st.error("Historical loading engine failed to construct values.")
        else:
            df_all_s3 = pd.DataFrame(collected_s3)

            for label, rng in CATEGORIES.items():
                b_df = df_all_s3[(df_all_s3["Drop %"] >= rng[0]) & (df_all_s3["Drop %"] < rng[1])].copy()
                b_df.sort_values("Drop %", ascending=False, inplace=True)

                c_color = CATEGORY_COLORS[label]
                with st.expander(f"{label} Bracket Pool — {len(b_df)} listings"):
                    st.markdown(
                        f"<div style='background:{c_color};padding:8px;border-radius:6px;color:black;'><b>Bracket Context: {label}</b> Drop Status</div>",
                        unsafe_allow_html=True)
                    if b_df.empty:
                        st.info("No assets within range boundaries.")
                    else:
                        render_df = b_df.copy()
                        render_df["Current Price"] = render_df["Current Price"].map(lambda x: f"{x:.2f}")
                        render_df["52-Week High"] = render_df["52-Week High"].map(lambda x: f"{x:.2f}")
                        render_df["Drop %"] = render_df["Drop %"].map(lambda x: f"{x:.2f}%")
                        st.dataframe(render_df, use_container_width=True)

# ==============================================================================
# ⚡ TAB 4: ATHARV ENHANCED SCANNER
# ==============================================================================
with tab4:
    st.header("⚡ Atharv Enhanced Swing Scanner (Day Range Pos Filters)")

    watchlist_df_s4 = shared_load_watchlist()
    tickers_s4 = watchlist_df_s4["Yahoo Ticker"].tolist()
    st.write(f"Loaded **{len(tickers_s4)}** tickers from internal configuration repository.")

    intervals_s4 = ["5m", "15m"]

    if st.button("Run Enhanced Scanner", key="btn_run_atharv_enhanced_scanner"):
        for interval in intervals_s4:
            st.subheader(f"Interval Target: {interval}")
            rows_s4 = []
            for _, row in watchlist_df_s4.iterrows():
                t = row["Yahoo Ticker"]
                company_name = row["Company Name"]
                res = s4_analyze_ticker(t, interval)

                if res["status"] != "OK":
                    rows_s4.append({
                        "Ticker": t, "Company (Ticker)": f"{company_name} ({t})", "Decision": "NO ENTER",
                        "Trend": "", "Close": "", "Support": "", "Resistance": "", "RSI": "",
                        "LongScore": "", "ShortScore": "", "CONFIRMED": "", "Strength": 0, "RangePos%": ""
                    })
                else:
                    rows_s4.append({
                        "Ticker": res["ticker"], "Company (Ticker)": f"{company_name} ({res['ticker']})",
                        "Decision": res["decision"], "Trend": res["trend"], "Close": round(res["close"], 2),
                        "Support": round(res["support"], 2), "Resistance": round(res["resistance"], 2),
                        "RSI": round(res["rsi"], 1), "LongScore": res["long_score"],
                        "ShortScore": res["short_score"], "CONFIRMED": res["confirmed"], "Strength": res["strength"],
                        "RangePos%": res["range_pos"]
                    })

            df_res_s4 = pd.DataFrame(rows_s4)
            df_res_s4["Rank"] = df_res_s4["Decision"].map({"BUY": 0, "WAIT": 1, "NO ENTER": 2}).fillna(3)
            df_res_s4["ConfRank"] = df_res_s4["CONFIRMED"].map({"CONFIRMED": 0, "NOT CONFIRMED": 1}).fillna(2)

            df_res_s4 = df_res_s4.sort_values(
                ["Rank", "ConfRank", "Strength", "LongScore"],
                ascending=[True, True, False, False]
            ).drop(columns=["Rank", "ConfRank"])

            styled_s4 = df_res_s4.style.apply(
                lambda col: [s1_decision_color(v) for v in col], subset=["Decision"]
            ).apply(
                lambda col: [s1_confirmed_color(v) for v in col], subset=["CONFIRMED"]
            )
            st.dataframe(styled_s4, use_container_width=True)
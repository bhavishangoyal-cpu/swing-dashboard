import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import time
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from google import genai
from google.genai import types
import os
import streamlit as st
from google import genai
import pandas_ta as ta  # <-- Ensure this is imported as 'ta'

@st.cache_resource
@st.cache_resource
def s4_get_ai_client(): # <-- Ensure NO arguments are inside the brackets here
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            # Forcing the key to the OS environment allows the SDK to structure headers correctly
            os.environ["GEMINI_API_KEY"] = api_key
            return genai.Client()
        except Exception:
            return None
    return None
# ==============================================================================
# 1. GLOBAL APP CONFIGURATION & INITIALIZATION (Must be at the absolute top)
# ==============================================================================
st.set_page_config(page_title="Master Trading Suite", layout="wide")
st.title("🎛️ Master Strategy & Scanning Interface")

# Setup Global Navigation Tabs (Now with Tab 4)
# Setup Global Navigation Tabs (Expanded with Tab 5)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Atharv Swing Scanner (5m/15m)",
    "📈 Goel's Swing Strategy",
    "📊 52-Week High/Low Strategy",
    "🚀 Atharv Corporate Guide",
    "🎯 80%+ Intraday Squeeze"
])

WATCHLIST_PATH = "watchlist.csv"
REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}


# ==============================================================================
# 2. SHARED DATA UTILITIES & ENGINE CONFIGURATIONS
# ==============================================================================
def shared_load_watchlist(path: str = WATCHLIST_PATH) -> pd.DataFrame:
    """Unified CSV Watchlist Loader used by all strategies"""
    if not os.path.exists(path):
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

    vol_ok = (not np.isnan(vals["vol_avg20"])) and (vals["vol_avg20"] > 0) and (
                vals["v_last"] > 1.1 * vals["vol_avg20"])

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

    confirmed = (decision == "BUY") and (trend == "UP") and (long_score >= 3) and (
                28 <= vals["rsi_last"] <= 55) and near_support and vol_ok
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
# 6. STRATEGY 4: ATHARV ENHANCED SCANNER UTILITIES (Your exact functions mapped)
# ==============================================================================
def s4_get_last_values(df: pd.DataFrame):
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
        "day_range_pos": (float(df["Close"].iloc[-1]) - float(df["Low"].iloc[-1])) / (
                    float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1]) + 1e-9),
    }


def s4_detect_support_resistance(df: pd.DataFrame, lookback: int = 40):
    recent = df.tail(lookback)
    support = recent['Low'].rolling(5).min().iloc[-1]
    resistance = recent['High'].rolling(5).max().iloc[-1]
    return support, resistance


def s4_analyze_ticker(ticker: str, interval: str) -> dict:
    data = s1_safe_history(ticker, interval=interval, period="7d")
    if data is None or len(data) < 60:
        return {"ticker": ticker, "status": "NO_DATA", "interval": interval}

    df = s1_compute_indicators(data).dropna()
    if len(df) < 30:
        return {"ticker": ticker, "status": "NO_DATA", "interval": interval}

    vals = s4_get_last_values(df)

    trend = "UP" if (vals["c_last"] > vals["ema20_last"] > vals["ema50_last"]) else "DOWN"

    support, resistance = s4_detect_support_resistance(df)

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
        trend == "UP",
        near_support,
        (28 <= vals["rsi_last"] <= 70) and rsi_up,
        bull_eng or is_hammer,
        vol_ok,
        vals["day_range_pos"] >= 0.75,
    ])

    short_score = sum([
        trend == "DOWN",
        near_resistance,
        (55 <= vals["rsi_last"] <= 75) and rsi_down,
        bear_eng,
        vol_ok,
    ])

    if long_score >= 4:
        decision = "BUY"
    elif short_score >= 4:
        decision = "SHORT"
    elif (trend in ["UP", "DOWN"]) and (near_support or near_resistance):
        decision = "WAIT"
    else:
        decision = "NO ENTER"

    confirmed = (
            (decision == "BUY") and
            (trend == "UP") and
            (long_score >= 3) and
            (28 <= vals["rsi_last"] <= 70) and
            near_support and
            (vals["v_last"] > 1.5 * vals["vol_avg20"])
    )

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
        "ticker": ticker,
        "status": "OK",
        "interval": interval,
        "trend": trend,
        "close": vals["c_last"],
        "support": support,
        "resistance": resistance,
        "rsi": vals["rsi_last"],
        "long_score": long_score,
        "short_score": short_score,
        "decision": decision,
        "confirmed": confirmed_label,
        "strength": strength,
        "range_pos": round(vals["day_range_pos"] * 100, 1),
    }


# ==============================================================================
# ==============================================================================
# TAB EXECUTION BLOCKS
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
    st_autorefresh(interval=180000, key="refresh_goel_tab")

    s2_watchlist = shared_load_watchlist()["Yahoo Ticker"].tolist()
    ticker_to_name = dict(zip(shared_load_watchlist()["Yahoo Ticker"], shared_load_watchlist()["Company Name"]))

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
# ⚡ TAB 4: ATHARV ENHANCED SCANNER (Your exact code UI logic mapped cleanly)
# ==============================================================================
# ==============================================================================
# ⚡ TAB 4: ATHARV CORPORATE SWING TRADING CO-PILOT
# ==============================================================================
with tab4:
    st.header("🚀 Atharv.py — Corporate Swing Trading Co-Pilot")
    st.markdown(
        "Designed for family-managed corporate accounts to identify momentum swings and analyze macro risk factors.")
    st.write("---")

    # Securely pull your free Gemini API Key
    # Call the function with empty parentheses. It pulls from Secrets automatically now!
    client_s4 = s4_get_ai_client()

    # Layout: Split into sidebar inputs or direct columns inside the tab to avoid clashing with global sidebars
    t_col1, t_col2 = st.columns([1, 2])

    with t_col1:
        st.subheader("Asset & Position Selection")
        ticker_input_s4 = st.text_input("Enter Ticker Symbol", value="NVDA", key="txt_ticker_s4").upper().strip()
        st.caption("💡 For Canadian assets, use the '.TO' suffix (e.g., XIU.TO or SHOP.TO)")

        my_purchase_price_s4 = st.number_input(
            "Enter Your Purchase Price ($)",
            value=0.0,
            step=0.01,
            key="num_purchase_s4",
            help="Set to 0.0 if you don't own this stock yet."
        )

    if ticker_input_s4:
        ticker_s4 = yf.Ticker(ticker_input_s4)

        with st.spinner(f"Analyzing {ticker_input_s4} historical structures and pulling live headlines..."):
            try:
                info_s4 = ticker_s4.info
                history_s4 = ticker_s4.history(period="2y")

                # Parse share stats & financial health
                shares_outstanding_s4 = info_s4.get('sharesOutstanding', None)
                float_shares_s4 = info_s4.get('floatShares', None)
                insider_pct_s4 = info_s4.get('heldPercentInsiders', 0) * 100
                inst_pct_s4 = info_s4.get('heldPercentInstitutions', 0) * 100

                avg_vol_3m_s4 = info_s4.get('averageVolume', None) or info_s4.get('averageDailyVolume3Month', None)

                if avg_vol_3m_s4 and shares_outstanding_s4:
                    daily_turnover_pct_s4 = (avg_vol_3m_s4 / shares_outstanding_s4) * 100
                else:
                    daily_turnover_pct_s4 = None

                shares_short_s4 = info_s4.get('sharesShort', None)
                shares_short_prior_s4 = info_s4.get('sharesShortPriorMonth', None)
                short_ratio_s4 = info_s4.get('shortRatio', None)
                short_pct_float_s4 = info_s4.get('shortPercentOfFloat', 0) * 100

                if shares_short_s4 and shares_short_prior_s4 and shares_short_prior_s4 > 0:
                    short_change_pct_s4 = ((shares_short_s4 - shares_short_prior_s4) / shares_short_prior_s4) * 100
                else:
                    short_change_pct_s4 = None

                profit_margin_s4 = info_s4.get('profitMargins', 0) * 100
                debt_to_equity_s4 = info_s4.get('debtToEquity', None)

            except Exception as e:
                st.error(f"Could not load data for '{ticker_input_s4}'. Please verify the symbol.")
                st.stop()

        if history_s4.empty:
            st.error(f"No trading background found for symbol: {ticker_input_s4}")
            st.stop()

        # Data Assignments
        name_s4 = info_s4.get('longName', 'N/A')
        sector_s4 = info_s4.get('sector', 'N/A')
        industry_s4 = info_s4.get('industry', 'N/A')
        summary_s4 = info_s4.get('longBusinessSummary', 'No corporate summary available.')

        pe_ratio_s4 = info_s4.get('trailingPE', 'N/A')
        forward_pe_s4 = info_s4.get('forwardPE', 'N/A')
        market_cap_s4 = info_s4.get('marketCap', 'N/A')

        avg_volume_s4 = info_s4.get('averageVolume', 0)
        beta_s4 = info_s4.get('beta', 1.0)
        held_by_institutions_s4 = info_s4.get('heldPercentInstitutions', 0) * 100

        current_price_s4 = info_s4.get('currentPrice', history_s4['Close'].iloc[-1])
        fifty_two_high_s4 = info_s4.get('fiftyTwoWeekHigh', max(history_s4['Close'][-252:]))
        fifty_two_low_s4 = info_s4.get('fiftyTwoWeekLow', min(history_s4['Close'][-252:]))

        # Technical Calculations
        history_s4['MA50'] = history_s4['Close'].rolling(window=50).mean()
        history_s4['MA200'] = history_s4['Close'].rolling(window=200).mean()
        ma50_now_s4 = history_s4['MA50'].iloc[-1]
        ma200_now_s4 = history_s4['MA200'].iloc[-1]
        pct_from_high_s4 = ((fifty_two_high_s4 - current_price_s4) / fifty_two_high_s4) * 100

        history_s4['MA21'] = history_s4['Close'].rolling(window=21).mean()
        ma21_now_s4 = history_s4['MA21'].iloc[-1]

        if ma21_now_s4 and ma21_now_s4 > 0:
            trend_cushion_pct_s4 = ((current_price_s4 - ma21_now_s4) / ma21_now_s4) * 100
        else:
            trend_cushion_pct_s4 = 0.0

        recent_volume_s4 = history_s4['Volume'].iloc[-5:].mean()
        long_avg_volume_s4 = info_s4.get('averageVolume', 1) if info_s4.get('averageVolume', 1) > 0 else 1
        volume_spike_ratio_s4 = recent_volume_s4 / long_avg_volume_s4

        if current_price_s4 >= ma21_now_s4:
            downward_diagnosis_s4 = "RUNNING"
        else:
            if current_price_s4 > ma200_now_s4:
                downward_diagnosis_s4 = "CORRECTION"
            elif current_price_s4 <= ma200_now_s4 and volume_spike_ratio_s4 > 1.5 and short_change_pct_s4 and short_change_pct_s4 > 10.0:
                downward_diagnosis_s4 = "STRUCTURAL_BLEED"
            else:
                downward_diagnosis_s4 = "MARKET_CRASH_OR_MACRO_FLUSH"

        target_low_s4 = info_s4.get('targetLowPrice', 'N/A')
        target_high_s4 = info_s4.get('targetHighPrice', 'N/A')
        target_mean_s4 = info_s4.get('targetMeanPrice', 'N/A')
        recommendation_s4 = info_s4.get('recommendationKey', 'N/A').replace('_', ' ').title()

        # Display Block
        with t_col2:
            st.header(f"🏢 {name_s4}")
            st.subheader(f"Sector: {sector_s4} | Industry: {industry_s4}")
            with st.expander("📄 View Company Profile Summary"):
                st.write(summary_s4)

        # Matrix Row
        st.write("---")
        st.subheader("📊 Live Technical & Fundamental Matrix")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            st.metric("Current Price", f"${current_price_s4:.2f}")
            st.metric("Trailing P/E",
                      f"{pe_ratio_s4:.2f}" if isinstance(pe_ratio_s4, (int, float)) else f"{pe_ratio_s4}")
        with m_col2:
            st.metric("52-Week High", f"${fifty_two_high_s4:.2f}")
            st.metric("Forward P/E",
                      f"{forward_pe_s4:.2f}" if isinstance(forward_pe_s4, (int, float)) else f"{forward_pe_s4}")
        with m_col3:
            st.metric("Distance from High", f"-{pct_from_high_s4:.1f}%")
            st.metric("Volatility (Beta)", f"{beta_s4:.2f}")
        with m_col4:
            st.metric("Market Cap",
                      f"${market_cap_s4 / 1e9:.2f}B" if isinstance(market_cap_s4, (int, float)) else "N/A")
            st.metric("Institutional Owned", f"{held_by_institutions_s4:.1f}%")

        # Liquidity Dynamics Row
        st.write("---")
        st.subheader("💧 Market Structure & Liquidity Dynamics")
        str_col1, str_col2, str_col3, str_col4 = st.columns(4)

        with str_col1:
            st.markdown("**Supply Structure**")
            st.metric("Shares Outstanding",
                      f"{shares_outstanding_s4 / 1e9:.2f}B" if shares_outstanding_s4 and shares_outstanding_s4 >= 1e9 else f"{shares_outstanding_s4 / 1e6:.2f}M" if shares_outstanding_s4 else "N/A")
            st.metric("Free Float",
                      f"{float_shares_s4 / 1e9:.2f}B" if float_shares_s4 and float_shares_s4 >= 1e9 else f"{float_shares_s4 / 1e6:.2f}M" if float_shares_s4 else "N/A")

        with str_col2:
            st.markdown("**Ownership Dynamics**")
            st.metric("Held by Institutions", f"{inst_pct_s4:.2f}%")
            st.metric("Held by Insiders", f"{insider_pct_s4:.2f}%")

        with str_col3:
            st.markdown("**Trading Liquidity**")
            st.metric("Avg Vol (3 Month)", f"{avg_vol_3m_s4 / 1e6:.2f}M" if avg_vol_3m_s4 else "N/A")
            if daily_turnover_pct_s4:
                if 3.0 <= daily_turnover_pct_s4 <= 7.0:
                    st.metric("Daily Turnover %", f"{daily_turnover_pct_s4:.2f}%", delta="🎯 Swing Sweet Spot")
                elif daily_turnover_pct_s4 > 15.0:
                    st.metric("Daily Turnover %", f"{daily_turnover_pct_s4:.2f}%", delta="⚠️ Hyper-Speculative",
                              delta_color="inverse")
                else:
                    st.metric("Daily Turnover %", f"{daily_turnover_pct_s4:.2f}%")
            else:
                st.metric("Daily Turnover %", "N/A")

        with str_col4:
            st.markdown("**Short Seller Pressure**")
            st.metric("Shares Short", f"{shares_short_s4 / 1e6:.2f}M" if shares_short_s4 else "N/A")
            st.metric("Short % of Float", f"{short_pct_float_s4:.2f}%")
            st.metric("Short Ratio (Days to Cover)", f"{short_ratio_s4:.1f}" if short_ratio_s4 else "N/A")

        # Corporate Health Row
        st.write("---")
        st.subheader("🏥 Financial Health & Short Trajectory")
        h_col1, h_col2, h_col3 = st.columns(3)

        with h_col1:
            st.markdown("**Core Profitability**")
            if profit_margin_s4:
                if profit_margin_s4 >= 20.0:
                    st.metric("Net Profit Margin", f"{profit_margin_s4:.2f}%", delta="🟢 Highly Profitable")
                elif profit_margin_s4 < 0.0:
                    st.metric("Net Profit Margin", f"{profit_margin_s4:.2f}%", delta="🔴 Burning Cash",
                              delta_color="inverse")
                else:
                    st.metric("Net Profit Margin", f"{profit_margin_s4:.2f}%")
            else:
                st.metric("Net Profit Margin", "N/A")

        with h_col2:
            st.markdown("**Leverage Risk**")
            if debt_to_equity_s4 is not None:
                if debt_to_equity_s4 <= 100.0:
                    st.metric("Debt-to-Equity Ratio", f"{debt_to_equity_s4:.1f}%", delta="🟢 Safe Leverage")
                elif debt_to_equity_s4 > 200.0:
                    st.metric("Debt-to-Equity Ratio", f"{debt_to_equity_s4:.1f}%", delta="⚠️ Heavy Debt Loading",
                              delta_color="inverse")
                else:
                    st.metric("Debt-to-Equity Ratio", f"{debt_to_equity_s4:.1f}%")
            else:
                st.metric("Debt-to-Equity Ratio", "N/A / Cash Rich")

        with h_col3:
            st.markdown("**Short Interest Trajectory**")
            if short_change_pct_s4 is not None:
                if short_change_pct_s4 > 10.0:
                    st.metric("Shorts MoM Change", f"{short_change_pct_s4:+.1f}%", delta="⚠️ Bears Accumulating",
                              delta_color="inverse")
                elif short_change_pct_s4 < -10.0:
                    st.metric("Shorts MoM Change", f"{short_change_pct_s4:+.1f}%", delta="🟢 Bears Fleeing")
                else:
                    st.metric("Shorts MoM Change", f"{short_change_pct_s4:+.1f}% (Stable)")
            else:
                st.metric("Shorts MoM Change", "N/A")

        # Institutional Consensus Row
        st.write("---")
        st.subheader("🏛️ Wall Street Institutional Consensus")
        w_col1, w_col2, w_col3, w_col4 = st.columns(4)

        with w_col1:
            st.metric("Consensus Rating", f"{recommendation_s4}")
        with w_col2:
            if isinstance(target_mean_s4, (int, float)):
                upside_s4 = ((target_mean_s4 - current_price_s4) / current_price_s4) * 100
                st.metric("Average Target", f"${target_mean_s4:.2f}", f"+{upside_s4:.1f}% Est. Upside")
            else:
                st.metric("Average Target", "N/A")
        with w_col3:
            st.metric("Bank Low Target", f"${target_low_s4:.2f}" if isinstance(target_low_s4, (int, float)) else "N/A")
        with w_col4:
            st.metric("Bank High Target",
                      f"${target_high_s4:.2f}" if isinstance(target_high_s4, (int, float)) else "N/A")

        # Diagnoser Row
        st.write("---")
        st.subheader("🎯 Institutional Trend & Downward Risk Diagnoser")
        st.markdown(
            "Tracks massive 1,000% runs while accurately diagnosing the exact structural nature of price drops.")

        ex_col1, ex_col2, ex_col3 = st.columns(3)

        with ex_col1:
            st.markdown("**Institutional Launchpad Status**")
            if current_price_s4 >= ma21_now_s4:
                st.metric("Launchpad Cushion", f"+{trend_cushion_pct_s4:.1f}%", delta="💎 Strong Institutional Support")
            else:
                st.metric("Launchpad Cushion", f"{trend_cushion_pct_s4:.1f}%", delta="⚠️ Below Launchpad Floor",
                          delta_color="inverse")

        with ex_col2:
            st.markdown("**Core Technical Baselines**")
            st.write(f"🔹 **21-Day Trend Floor:** ${ma21_now_s4:.2f}")
            st.write(f"🏛️ **200-Day Macro Floor:** ${ma200_now_s4:.2f}")

        with ex_col3:
            st.markdown("**Strategic Execution & Trend Diagnosis**")
            if downward_diagnosis_s4 == "RUNNING":
                st.success(
                    "🚀 RIDE THE RUNNER: Trend is perfectly healthy. Let your profits compound into maximum potential.")
            elif downward_diagnosis_s4 == "CORRECTION":
                st.warning(
                    "🟡 HEALTHY CORRECTION: Price is dipping but remains safely above the 200-Day Macro Floor. No structural damage detected.")
            elif downward_diagnosis_s4 == "MARKET_CRASH_OR_MACRO_FLUSH":
                st.info(
                    "🌊 MACRO FLUSH / CRASH SECTOR: Stock is below major floors but lacking heavy volume liquidation. Hold firm through systemic volatility.")
            elif downward_diagnosis_s4 == "STRUCTURAL_BLEED":
                st.error(
                    "🚨 STRUCTURAL DOWNWARD TREND: Asset has completely broken down below the 200-Day Floor on high institutional volume. DO NOT add fresh capital.")

        # Checklist Positioning Block
        if my_purchase_price_s4 > 0.0:
            st.write("---")
            st.subheader(f"📋 Personalized Execution Checklist for {ticker_input_s4}")
            gain_loss_pct_s4 = ((current_price_s4 - my_purchase_price_s4) / my_purchase_price_s4) * 100
            in_the_green_s4 = current_price_s4 >= my_purchase_price_s4

            list_col1, list_col2 = st.columns([1, 2])

            with list_col1:
                st.markdown("**Your Equity Status Metrics**")
                st.metric("Your Cost Basis", f"${my_purchase_price_s4:.2f}")
                if in_the_green_s4:
                    st.metric("Position Return", f"+{gain_loss_pct_s4:.2f}%", delta="🟢 In The Green")
                else:
                    st.metric("Position Return", f"{gain_loss_pct_s4:.2f}%", delta="🔴 Capital In Drawdown",
                              delta_color="inverse")

            with list_col2:
                st.markdown("**What To Do Right Now (Action List):**")
                if downward_diagnosis_s4 == "RUNNING":
                    if in_the_green_s4:
                        st.markdown(
                            f"* **[HOLD]** Your position is safely in the green (`+{gain_loss_pct_s4:.1f}%`) and institutional momentum is roaring.\n* **[TRAILING TRACK]** Your profit floor is protected by the 21-Day Trend Floor at **${ma21_now_s4:.2f}**.\n* **[EXECUTION]** Take no profit reduction until the price closes below the 21-day floor line.")
                    else:
                        st.markdown(
                            f"* **[HOLD / WATCH]** You are down `{gain_loss_pct_s4:.1f}%` from your entry, but the asset has flipped into a fresh **Launchpad Run**.\n* **[BUY ALIGNMENT]** The path to your break-even point is open above **${ma21_now_s4:.2f}**.\n* **[EXECUTION]** Hold firm. No panic-selling allowed while institutions are buying.")
                elif downward_diagnosis_s4 == "CORRECTION":
                    if in_the_green_s4:
                        st.markdown(
                            f"* **[PROTECT / HOLD]** You are up `+{gain_loss_pct_s4:.1f}%`, but the stock is undergoing a short-term pullback.\n* **[SAFETY MATRIX]** The long-term floor at **${ma200_now_s4:.2f}** is still completely intact.\n* **[EXECUTION]** Use **${ma21_now_s4:.2f}** as a tight soft exit line, or hold safely through the temporary dip.")
                    else:
                        st.markdown(
                            f"* **[HOLD & ACCUMULATE]** You are down `{gain_loss_pct_s4:.1f}%`, but it is diagnosed as a **Healthy Technical Correction**.\n* **[SUPPORT CHECK]** Tracking safely above the long-term macro floor (**${ma200_now_s4:.2f}**).\n* **[EXECUTION]** Do not sell at a loss. This is a safe area to average down your entry cost.")
                elif downward_diagnosis_s4 == "MARKET_CRASH_OR_MACRO_FLUSH":
                    st.markdown(
                        f"* **[STRICT FREEZE & HOLD]** Down due to a broad macro sweep or market panic. Paper variance is `{gain_loss_pct_s4:.1f}%`.\n* **[PHILOSOPHY COMPLIANCE]** Remember your corporate rule: **Never sell at a loss.**\n* **[EXECUTION]** Freeze the position entirely. Let the corporate account carry it safely until the panic passes.")
                elif downward_diagnosis_s4 == "STRUCTURAL_BLEED":
                    st.markdown(
                        f"* **[LOCK CAPITAL / DO NOT ADD]** Broken major technical benchmarks (**${ma200_now_s4:.2f}**) on distribution volume. Down `{gain_loss_pct_s4:.1f}%`.\n* **[RISK WARNING]** Entering a multi-month cooling cycle.\n* **[EXECUTION]** **DO NOT throw good money after bad.** Freeze this ticker, let existing shares sit, and reallocate fresh cash to green runners.")

        # Bottom Deep Dives
        st.write("---")
        bot_col1, bot_col2 = st.columns(2)

        with bot_col1:
            st.subheader("⚙️ Automated Algorithmic Logic")
            st.markdown("**Short-Term Swing Direction:**")
            if current_price_s4 > ma50_now_s4 and ma50_now_s4 > ma200_now_s4:
                st.success("🟢 Strong Upward Momentum. Structural trend is healthy; target pullbacks for entry.")
            elif current_price_s4 < ma50_now_s4 and current_price_s4 > ma200_now_s4:
                st.warning(
                    "🟡 Technical Correction. Price retreating toward the 200-day floor. Monitor for reversal support.")
            else:
                st.error("🔴 Bearish Structural Trend. High capital vulnerability for immediate swing trades.")

            st.markdown("**1-2 Year Structural Outlook:**")
            if isinstance(forward_pe_s4, (int, float)) and isinstance(pe_ratio_s4, (int, float)):
                if forward_pe_s4 < pe_ratio_s4:
                    st.info(
                        "🔵 Positive. Earnings projections expand outward, indicating long-term valuation discount room.")
                else:
                    st.markdown(
                        "⚪ *Premium/Flat. Growth trajectories appear valued-in by core institutional analysts.*")
            else:
                st.markdown("⚪ *Data insufficient to safely cross-verify corporate forwarding horizons.*")

            st.markdown("**Capital Exit Liquidity:**")
            if avg_volume_s4 > 1000000:
                st.success(f"✅ Safe ({avg_volume_s4:,} avg shares/day). Swift exit execution available.")
            elif avg_volume_s4 > 200000:
                st.warning(f"⚠️ Moderate ({avg_volume_s4:,} avg shares/day). Handle under controlled size allocation.")
            else:
                st.error(
                    f"🚨 Extreme Liquidity Risk ({avg_volume_s4:,} shares). High probability of slippage parameters.")

        with bot_col2:
            st.subheader("📰 Live Catalyst Feed & AI Deep Dive")
            if not client_s4:
                st.warning(
                    "⚠️ Enter a valid Gemini API Key at the top of the file to populate the AI sentiment breakdown below.")
            else:
                with st.spinner("Activating Google Search Grounding to fetch live market catalysts..."):
                    prompt_s4 = f"""
                    Perform a live regulatory and sentiment risk assessment for the ticker asset: {ticker_input_s4} ({name_s4}).
                    1. Identify the top 3-4 major news headlines, product announcements, or earnings catalysts from the past 72 hours.
                    2. Evaluate if these events represent short-term volatility plays (swings) or changing structural fundamentals for holding 1-2 years. 
                    3. Explicitly state any immediate hazards to corporate capital reserves or cash liquidity parameters.
                    Format your final response with clean, professional bold headers. Provide direct bullet points for the news events.
                    """
                    try:
                        response_s4 = client_s4.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt_s4,
                            config=types.GenerateContentConfig(tools=[{"google_search": {}}])
                        )
                        st.markdown("#### 🤖 Automated AI Intelligence Report")
                        st.info(response_s4.text)
                    except Exception as ai_err:
                        st.error(f"AI Synthesis module failed to execute: {ai_err}")

# ==============================================================================
# 🎯 TAB 5: 80%+ INTRADAY SQUEEZE BREAKOUT STRATEGY (PATH RESOLVED ENGINE)
# ==============================================================================
with tab5:
    st.header("🎯 High-Effectiveness Intraday Squeeze Scanner")
    st.caption("Filters for 1.2% Volatility Squeezes with Multi-Timeframe Institutional Volume Confirmation")

    try:
        # Load directly from your existing path variable defined in your script
        import os

        if os.path.exists(WATCHLIST_PATH):
            tab5_watchlist_df = pd.read_csv(WATCHLIST_PATH)
        else:
            st.error(f"❌ Watchlist file not found at path: {WATCHLIST_PATH}. Please check your file system.")
            tab5_watchlist_df = pd.DataFrame()

        if not tab5_watchlist_df.empty:
            # Detect right columns automatically (handles different capitalization formats)
            ticker_col = None
            for col in ["Yahoo Ticker", "ticker", "Ticker", "Symbol"]:
                if col in tab5_watchlist_df.columns:
                    ticker_col = col
                    break
            if not ticker_col:
                ticker_col = tab5_watchlist_df.columns[0]

            name_col = None
            for col in ["Company Name", "name", "Name", "Company"]:
                if col in tab5_watchlist_df.columns:
                    name_col = col
                    break
            if not name_col:
                name_col = ticker_col

            # Clean and extract ticker symbols cleanly
            tab5_tickers = []
            for t in tab5_watchlist_df[ticker_col].tolist():
                if pd.notna(t):
                    ticker_str = str(t).strip().upper()
                    if ticker_str and ticker_str not in ["NAN", "NONE", ""]:
                        tab5_tickers.append(ticker_str)

            if not tab5_tickers:
                st.warning("⚠️ No valid tickers found in your watchlist column.")
            else:
                st.write(f"Loaded **{len(tab5_tickers)}** symbols from your live profile.")

                # Manual control button to trigger execution safely
                run_scan = st.button("🔄 Execute Live Intraday Scan", key="final_squeeze_scan_action")

                # Dictionary lookup mapping
                ticker_to_name = {}
                for _, row in tab5_watchlist_df.iterrows():
                    if pd.notna(row[ticker_col]):
                        tk = str(row[ticker_col]).strip().upper()
                        nm = row[name_col] if pd.notna(row[name_col]) else tk
                        ticker_to_name[tk] = nm

                if run_scan:
                    squeeze_rows = []

                    # 🚀 BATCH DOWNLOAD BOTH TIMEFRAMES IN BULK
                    with st.spinner("Streaming live market data matrices from Yahoo Finance..."):
                        master_1h = yf.download(tab5_tickers, period="1mo", interval="1h", progress=False,
                                                group_by="ticker", prepost=True)
                        master_5m = yf.download(tab5_tickers, period="5d", interval="5m", progress=False,
                                                group_by="ticker", prepost=True)

                    if master_5m.empty or master_1h.empty:
                        st.info(
                            "🌙 Market data stream is currently empty or unavailable. Try running during active market hours!")
                    else:
                        is_multi = len(tab5_tickers) > 1

                        # 🧠 IN-MEMORY MATH ENGINE
                        for t in tab5_tickers:
                            try:
                                if is_multi:
                                    if t not in master_1h.columns.get_level_values(
                                            0) or t not in master_5m.columns.get_level_values(0):
                                        continue
                                    df_1h = master_1h[t].copy()
                                    df_5m = master_5m[t].copy()
                                else:
                                    df_1h = master_1h.copy()
                                    df_5m = master_5m.copy()

                                df_1h.columns = [str(c).strip().capitalize() for c in df_1h.columns]
                                df_5m.columns = [str(c).strip().capitalize() for c in df_5m.columns]

                                df_1h = df_1h.dropna(subset=['Close'])
                                df_5m = df_5m.dropna(subset=['Close'])

                                if len(df_1h) < 15 or len(df_5m) < 25:
                                    continue

                                # --- LAYER 1: HOURLY TREND ANCHOR ---
                                df_1h['EMA_50'] = df_1h['Close'].ewm(span=50, adjust=False).mean()
                                macro_uptrend = float(df_1h['Close'].iloc[-1]) > float(df_1h['EMA_50'].iloc[-1])

                                # --- LAYER 2 & 3: INTRADAY SQUEEZE & VOLUME ---
                                df_5m['High_20'] = df_5m['High'].rolling(20).max()
                                df_5m['Low_20'] = df_5m['Low'].rolling(20).min()

                                c_last = float(df_5m['Close'].iloc[-1])
                                high_box_prev = float(df_5m['High_20'].iloc[-2])
                                low_box_prev = float(df_5m['Low_20'].iloc[-2])

                                box_width_pct = (high_box_prev - low_box_prev) / c_last
                                is_squeezed = box_width_pct <= 0.012
                                price_breakout = c_last > high_box_prev

                                # Chaikin Money Flow Calculation
                                df_5m['CMF'] = ta.cmf(df_5m['High'], df_5m['Low'], df_5m['Close'], df_5m['Volume'],
                                                      length=20)
                                cmf_val = float(df_5m['CMF'].iloc[-1]) if not np.isnan(df_5m['CMF'].iloc[-1]) else 0.0
                                volume_confirmed = cmf_val >= 0.10

                                if macro_uptrend and is_squeezed and price_breakout and volume_confirmed:
                                    decision = "🔥 STRONG BUY SETUP"
                                elif macro_uptrend and is_squeezed:
                                    decision = "⏳ Squeezed (Waiting for Breakout)"
                                else:
                                    decision = "❌ No Squeeze Setup"

                                name = ticker_to_name.get(t, t)
                                squeeze_rows.append({
                                    "Company (Ticker)": f"{name} ({t})",
                                    "Decision Signal": decision,
                                    "Live Intraday Price": round(c_last, 2),
                                    "Squeeze Width (%)": f"{round(box_width_pct * 100, 2)}%",
                                    "Institutional Flow (CMF)": round(cmf_val, 2),
                                    "Macro Trend (1H)": "Bullish ✓" if macro_uptrend else "Bearish ✗",
                                    "Risk Stop-Loss Floor": round(low_box_prev, 2)
                                })
                            except:
                                continue

                    # Render Output Table
                    if squeeze_rows:
                        df_results = pd.DataFrame(squeeze_rows)
                        df_results["Sort_Order"] = df_results["Decision Signal"].map({
                            "🔥 STRONG BUY SETUP": 0,
                            "⏳ Squeezed (Waiting for Breakout)": 1,
                            "❌ No Squeeze Setup": 2
                        }).fillna(3)
                        df_results = df_results.sort_values("Sort_Order").drop(columns=["Sort_Order"])


                        def style_squeeze_signals(val):
                            if "STRONG BUY" in val: return "background-color: #2ECC71; color: black; font-weight: bold;"
                            if "Squeezed" in val: return "background-color: #F1C40F; color: black;"
                            return "color: #7F8C8D;"


                        styled_output = df_results.style.applymap(style_squeeze_signals, subset=["Decision Signal"])
                        st.dataframe(styled_output, use_container_width=True, hide_index=True)
                    else:
                        st.info(
                            "ℹ️ Watchlist loaded. Click 'Execute Live Intraday Scan' during market hours to run calculations.")
                else:
                    st.info("ℹ️ Ready to scan. Click 'Execute Live Intraday Scan' to begin processing.")

    except Exception as global_tab_error:
        st.error(f"💥 Tab 5 Functional Error: {str(global_tab_error)}")
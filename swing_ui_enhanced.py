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



# 1. Fetch and Clean Data
spy_data = yf.download("SPY", period="1mo", interval="1d", progress=False)

# Clean multi-index columns if present
if isinstance(spy_data.columns, pd.MultiIndex):
    spy_data.columns = spy_data.columns.get_level_values(0)

# Extract 'Close' series safely
close = spy_data['Close']
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

# 2. Perform Calculations
if len(close) >= 20:
    spy_price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    spy_change = spy_price - prev_close
    spy_percent = (spy_change / prev_close) * 100

    # Moving Average & Z-Score
    mean = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()

    z_score_series = (close - mean) / std
    latest_z_score = float(z_score_series.iloc[-1])

    # Market Health Logic
    if latest_z_score < -1.5:
        market_status = "GOOD TIME (Oversold)"
    elif latest_z_score > 1.5:
        market_status = "BAD TIME (Overbought)"
    else:
        market_status = "STABLE"

    # 3. Display
    st.subheader("Market Context: S&P 500 (SPY)")
    col1, col2, col3 = st.columns(3)
    col1.metric("SPY Price", f"${spy_price:.2f}", f"{spy_change:+.2f} ({spy_percent:+.2f}%)")
    col2.metric("SPY Z-Score", f"{latest_z_score:.2f}")
    col3.info(f"Market Sentiment: {market_status}")

    st.divider()
else:
    st.warning("Gathering market data...")

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
# 1. Update the tabs list to include the 6th title
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 Atharv Swing Scanner (5m/15m)",
    "📈 Goel's Swing Strategy",
    "📊 52-Week High/Low Strategy",
    "🚀 Atharv Corporate Guide",
    "🎯 80%+ Intraday Squeeze",
    "🦅 Institutional Alpha Matrix",
    "🏛️ Fundamental Terminal"
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
# 7. STRATEGY 7 (TAB 7): FUNDAMENTAL & PORTFOLIO TERMINAL UTILITIES
#    4-Pillar scoring, ROCE trend, DCF intrinsic value, DuPont/FCF, analyst
#    consensus, ownership, short interest/options, news, technicals, forensic
#    red-flag checks, peer/portfolio comparison. All function/constant names
#    are prefixed s7_ to avoid any collision with the other strategy tabs.
# ==============================================================================
# 1. PILLAR MAP — raw yfinance line item -> (Pillar, Category)
# ============================================================================
s7_PILLAR_MAP = {
    "Pillar 1: Quality": {
        "Performance": ['Total Revenue', 'Operating Revenue', 'Gross Profit', 'Operating Income',
                         'EBIT', 'EBITDA', 'Normalized EBITDA', 'ROCE %'],
        "Bottom Line": ['Net Income', 'Net Income Common Stockholders', 'Normalized Income'],
        "Efficiency": ['Operating Expense', 'Total Expenses', 'Research And Development',
                        'Selling General And Administration', 'Cost Of Revenue'],
        "Shareholder Value": ['Basic EPS', 'Diluted EPS', 'Basic Average Shares', 'Diluted Average Shares'],
        "Tax & Non-Operating": ['Tax Provision', 'Pretax Income', 'Interest Expense', 'Interest Income'],
    },
    "Pillar 2: Safety": {
        "Debt Load": ['Total Debt', 'Net Debt', 'Long Term Debt', 'Current Debt'],
        "Obligations": ['Current Liabilities', 'Accounts Payable', 'Total Liabilities Net Minority Interest'],
    },
    "Pillar 3: Assets & Liquidity": {
        "Liquidity": ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments',
                       'Accounts Receivable', 'Inventory', 'Current Assets', 'Working Capital'],
        "Hard Assets": ['Total Assets', 'Net PPE', 'Gross PPE', 'Investments And Advances', 'Goodwill'],
        "Cash Flow": ['Operating Cash Flow', 'Capital Expenditure', 'Free Cash Flow'],
    },
    "Pillar 4: Capital Structure": {
        "Equity": ['Stockholders Equity', 'Common Stock Equity', 'Total Capitalization',
                   'Tangible Book Value', 'Retained Earnings'],
        "Shares": ['Share Issued', 'Ordinary Shares Number', 'Treasury Shares Number'],
    }
}

s7_PER_SHARE_ITEMS = {'Basic EPS', 'Diluted EPS'}
s7_SHARE_COUNT_ITEMS = {'Basic Average Shares', 'Diluted Average Shares', 'Share Issued',
                      'Ordinary Shares Number', 'Treasury Shares Number'}
s7_RATIO_ITEMS = {'Tax Rate For Calcs'}
s7_PERCENT_ITEMS = {'ROCE %'}

s7_ALL_ITEMS = [item for pillar in s7_PILLAR_MAP.values() for cat in pillar.values() for item in cat]


# ============================================================================
# 2. DATA FETCH
# ============================================================================
def s7_inject_roce(df):
    df = df.copy()
    if 'EBIT' in df.index and 'Total Assets' in df.index and 'Current Liabilities' in df.index:
        capital_employed = df.loc['Total Assets'] - df.loc['Current Liabilities']
        roce = (df.loc['EBIT'] / capital_employed.replace(0, np.nan)) * 100
        df.loc['ROCE %'] = roce
    return df


@st.cache_data(ttl=3600)
def s7_fetch_statements(ticker):
    stock = yf.Ticker(ticker)
    annual = pd.concat([stock.financials, stock.balance_sheet, stock.cashflow])
    quarterly = pd.concat([stock.quarterly_financials, stock.quarterly_balance_sheet, stock.quarterly_cashflow])
    annual = annual[~annual.index.duplicated(keep='first')]
    quarterly = quarterly[~quarterly.index.duplicated(keep='first')]
    annual = annual.loc[:, ~annual.columns.duplicated()]
    quarterly = quarterly.loc[:, ~quarterly.columns.duplicated()]
    annual = s7_inject_roce(annual)
    quarterly = s7_inject_roce(quarterly)
    return annual, quarterly


@st.cache_data(ttl=3600)
def s7_fetch_price_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2y")
    try:
        info = stock.info
    except Exception:
        info = {}
    return hist, info


@st.cache_data(ttl=3600)
def s7_fetch_extended_data(ticker):
    stock = yf.Ticker(ticker)
    out = {}
    for attr, key in [('insider_transactions', 'insider_tx'),
                       ('institutional_holders', 'inst_holders'),
                       ('earnings_history', 'earnings_hist'),
                       ('earnings_dates', 'earnings_dates')]:
        try:
            out[key] = getattr(stock, attr)
        except Exception:
            out[key] = pd.DataFrame()
    return out


@st.cache_data(ttl=1800)
def s7_fetch_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news if news else []
    except Exception:
        return []


@st.cache_data(ttl=3600)
def s7_fetch_options_snapshot(ticker):
    try:
        stock = yf.Ticker(ticker)
        expiries = stock.options
        if not expiries:
            return None
        nearest = expiries[0]
        chain = stock.option_chain(nearest)
        calls, puts = chain.calls, chain.puts
        call_oi = calls['openInterest'].fillna(0).sum()
        put_oi = puts['openInterest'].fillna(0).sum()
        pc_ratio = round(put_oi / call_oi, 2) if call_oi else np.nan
        avg_iv_call = round(calls['impliedVolatility'].mean() * 100, 1) if 'impliedVolatility' in calls else np.nan
        avg_iv_put = round(puts['impliedVolatility'].mean() * 100, 1) if 'impliedVolatility' in puts else np.nan
        return {'Nearest Expiry': nearest, 'Put/Call OI Ratio': pc_ratio,
                'Avg Call IV %': avg_iv_call, 'Avg Put IV %': avg_iv_put,
                'Call OI': int(call_oi), 'Put OI': int(put_oi)}
    except Exception:
        return None


@st.cache_data(ttl=1800)
def s7_fetch_macro():
    tickers = {'10Y Yield %': '^TNX', 'VIX': '^VIX', 'Dollar Index': 'DX-Y.NYB'}
    out = {}
    for label, tk in tickers.items():
        try:
            h = yf.Ticker(tk).history(period='5d')
            if not h.empty:
                val = h['Close'].iloc[-1]
                if label == '10Y Yield %':
                    val = val / 10  # ^TNX quotes yield x10
                out[label] = round(val, 2)
            else:
                out[label] = np.nan
        except Exception:
            out[label] = np.nan
    return out


# ============================================================================
# 3. RAW STATEMENT VIEW (pillar-grouped, correctly scaled, chronological)
# ============================================================================
def s7_build_display_table(annual, quarterly, n_annual=2):
    annual = annual.sort_index(axis=1, ascending=False)
    quarterly = quarterly.sort_index(axis=1, ascending=False)

    annual_cols = annual.iloc[:, :n_annual].copy()
    latest_annual_ts = annual.columns[0] if len(annual.columns) > 0 else None

    if latest_annual_ts is not None and len(quarterly.columns) > 0:
        valid_q = [c for c in quarterly.columns if c > latest_annual_ts]
        q_cols = quarterly[valid_q].copy()
    else:
        q_cols = pd.DataFrame()

    labeled = {}
    for c in annual_cols.columns:
        labeled[c] = f"FY {c.strftime('%Y-%m-%d')}"
    for c in q_cols.columns:
        labeled[c] = f"Q ({c.strftime('%Y-%m-%d')})"

    ordered_ts = sorted(labeled.keys(), reverse=False)
    combined = pd.concat([annual_cols, q_cols], axis=1)
    combined = combined[ordered_ts]
    combined.columns = [labeled[c] for c in ordered_ts]

    df = combined.reindex(s7_ALL_ITEMS)
    df = df.dropna(how='all')
    return df


def s7_scale_row(row_name, value):
    if pd.isna(value):
        return np.nan
    if row_name in s7_PER_SHARE_ITEMS:
        return round(value, 2)
    if row_name in s7_SHARE_COUNT_ITEMS:
        return round(value / 1e6, 1)
    if row_name in s7_RATIO_ITEMS:
        return round(value * 100, 1)
    if row_name in s7_PERCENT_ITEMS:
        return round(value, 1)
    return round(value / 1e9, 3)


def s7_format_display_table(df):
    out = df.copy()
    for idx in out.index:
        out.loc[idx] = [s7_scale_row(idx, v) for v in out.loc[idx]]
    return out


# ============================================================================
# 4. RATIO ENGINE
# ============================================================================
def s7_safe_get(df, row, col_idx=0):
    try:
        return df.loc[row].iloc[col_idx]
    except (KeyError, IndexError):
        return np.nan


def s7_pct(cur, prior):
    if pd.isna(cur) or pd.isna(prior) or prior == 0:
        return np.nan
    return round((cur - prior) / abs(prior) * 100, 1)


def s7_ratio_pct(num, den):
    if pd.isna(num) or pd.isna(den) or den == 0:
        return np.nan
    return round(num / den * 100, 1)


def s7_safe_div(num, den):
    if pd.isna(num) or pd.isna(den) or den == 0:
        return np.nan
    return round(num / den, 2)


def s7_compute_ratios(annual, quarterly):
    a = annual.sort_index(axis=1, ascending=False)

    rev = s7_safe_get(a, 'Total Revenue', 0)
    rev_prior = s7_safe_get(a, 'Total Revenue', 1)
    gp = s7_safe_get(a, 'Gross Profit', 0)
    ni = s7_safe_get(a, 'Net Income', 0)
    ebit = s7_safe_get(a, 'EBIT', 0)
    ebitda = s7_safe_get(a, 'EBITDA', 0)
    eps = s7_safe_get(a, 'Diluted EPS', 0)
    eps_prior = s7_safe_get(a, 'Diluted EPS', 1)
    interest_exp = s7_safe_get(a, 'Interest Expense', 0)

    total_debt = s7_safe_get(a, 'Total Debt', 0)
    equity = s7_safe_get(a, 'Stockholders Equity', 0)
    current_assets = s7_safe_get(a, 'Current Assets', 0)
    current_liab = s7_safe_get(a, 'Current Liabilities', 0)
    inventory = s7_safe_get(a, 'Inventory', 0)
    cash = s7_safe_get(a, 'Cash And Cash Equivalents', 0)
    total_assets = s7_safe_get(a, 'Total Assets', 0)
    shares_now = s7_safe_get(a, 'Ordinary Shares Number', 0)
    shares_prior = s7_safe_get(a, 'Ordinary Shares Number', 1)

    roce_now = s7_safe_get(a, 'ROCE %', 0)
    roce_prior = s7_safe_get(a, 'ROCE %', 1)

    quick_assets = np.nan
    if not pd.isna(current_assets) and not pd.isna(inventory):
        quick_assets = current_assets - inventory

    return {
        'Revenue YoY %': s7_pct(rev, rev_prior),
        'EPS YoY %': s7_pct(eps, eps_prior),
        'Gross Margin %': s7_ratio_pct(gp, rev),
        'Net Margin %': s7_ratio_pct(ni, rev),
        'EBITDA Margin %': s7_ratio_pct(ebitda, rev),
        'ROCE %': round(roce_now, 1) if not pd.isna(roce_now) else np.nan,
        'ROCE YoY Δ (pts)': round(roce_now - roce_prior, 1) if not pd.isna(roce_now) and not pd.isna(roce_prior) else np.nan,
        'Debt/Equity': s7_safe_div(total_debt, equity),
        'Interest Coverage': s7_safe_div(ebit, interest_exp),
        'Current Ratio': s7_safe_div(current_assets, current_liab),
        'Quick Ratio': s7_safe_div(quick_assets, current_liab),
        'Cash/Debt': s7_safe_div(cash, total_debt),
        'Equity Ratio %': s7_ratio_pct(equity, total_assets),
        'Share Count YoY %': s7_pct(shares_now, shares_prior),
    }


def s7_roce_trend_flag(quarterly, lookback=3):
    q = quarterly.sort_index(axis=1, ascending=False)
    if 'ROCE %' not in q.index:
        return False, []
    series = q.loc['ROCE %'].dropna()
    if len(series) < lookback + 1:
        return False, [(c.strftime('%Y-%m-%d'), round(v, 1)) for c, v in series.items()]
    recent = series.iloc[:lookback + 1].iloc[::-1]
    values = recent.tolist()
    declining = all(values[i] > values[i + 1] for i in range(len(values) - 1))
    display = [(c.strftime('%Y-%m-%d'), round(v, 1)) for c, v in recent.items()]
    return declining, display


# ============================================================================
# 5. SCORING
# ============================================================================
s7_PILLAR_SCORING = {
    "Pillar 1: Quality": [
        ('Revenue YoY %', (10, 0), True),
        ('EPS YoY %', (10, 0), True),
        ('Net Margin %', (15, 5), True),
        ('EBITDA Margin %', (20, 10), True),
        ('ROCE %', (20, 10), True),
    ],
    "Pillar 2: Safety": [
        ('Debt/Equity', (0.5, 1.5), False),
        ('Interest Coverage', (8, 3), True),
        ('Current Ratio', (1.5, 1.0), True),
    ],
    "Pillar 3: Assets & Liquidity": [
        ('Quick Ratio', (1.0, 0.5), True),
        ('Cash/Debt', (0.5, 0.2), True),
    ],
    "Pillar 4: Capital Structure": [
        ('Equity Ratio %', (50, 30), True),
        ('Share Count YoY %', (-1, 1), False),
    ],
}


def s7_score_metric(value, thresholds, higher_is_better):
    if pd.isna(value):
        return None
    strong, moderate = thresholds
    if higher_is_better:
        if value >= strong:
            return 2
        if value >= moderate:
            return 1
        return 0
    else:
        if value <= strong:
            return 2
        if value <= moderate:
            return 1
        return 0


def s7_score_pillars(ratios):
    results = {}
    for pillar, metrics in s7_PILLAR_SCORING.items():
        scores, detail = [], []
        for name, thresholds, higher_better in metrics:
            val = ratios.get(name, np.nan)
            s = s7_score_metric(val, thresholds, higher_better)
            detail.append((name, val, s))
            if s is not None:
                scores.append(s)
        if scores:
            avg = sum(scores) / len(scores)
            pct_score = round(avg / 2 * 100)
            verdict = "Strong" if avg >= 1.5 else "Moderate" if avg >= 0.75 else "Weak"
        else:
            pct_score, verdict = None, "No Data"
        results[pillar] = {'score': pct_score, 'verdict': verdict, 'detail': detail}
    return results


def s7_overall_verdict(pillar_results):
    safety = pillar_results.get("Pillar 2: Safety", {})
    if safety.get('verdict') == 'Weak':
        return "AVOID / HIGH RISK", "Safety pillar failed — this gates the verdict regardless of quality upside."
    scores = [v['score'] for v in pillar_results.values() if v['score'] is not None]
    if not scores:
        return "NO DATA", "Insufficient data to score."
    avg = sum(scores) / len(scores)
    if avg >= 70:
        return "BUY", "All pillars supportive."
    elif avg >= 50:
        return "HOLD", "Mixed signals — acceptable but not compelling."
    else:
        return "AVOID", "Weak fundamentals across pillars."


# ============================================================================
# 6. VALUATION / TIMING OVERLAY
# ============================================================================
def s7_valuation_timing(hist, info, fund_verdict):
    if hist is None or hist.empty:
        return None
    price = hist['Close'].iloc[-1]
    high_2y = hist['Close'].max()
    low_2y = hist['Close'].min()
    pos_pct = (price - low_2y) / (high_2y - low_2y) * 100 if high_2y != low_2y else 50

    trailing_pe = info.get('trailingPE', np.nan)
    forward_pe = info.get('forwardPE', np.nan)

    if pos_pct <= 33:
        zone = "Value Zone (lower third of 2Y range)"
    elif pos_pct <= 66:
        zone = "Fair Zone (mid-range)"
    else:
        zone = "Extended Zone (upper third of 2Y range)"

    if fund_verdict == "BUY" and pos_pct <= 40:
        action = "ENTRY — fundamentals strong, price attractive"
    elif fund_verdict == "BUY" and pos_pct > 75:
        action = "WAIT — fundamentals strong, price extended; wait for pullback"
    elif fund_verdict in ("AVOID", "AVOID / HIGH RISK") and pos_pct > 60:
        action = "EXIT / TRIM — weak fundamentals, price still elevated"
    elif fund_verdict == "HOLD":
        action = "HOLD — monitor, no urgency either way"
    else:
        action = "MONITOR"

    return {'price': price, 'high_2y': high_2y, 'low_2y': low_2y, 'position_pct': round(pos_pct, 1),
            'zone': zone, 'trailing_pe': trailing_pe, 'forward_pe': forward_pe, 'action': action}


# ============================================================================
# 7. VALUATION MULTIPLES + ACCURATE HISTORICAL P/E BAND
# ============================================================================
def s7_valuation_multiples(info):
    return {
        'P/E (Trailing)': info.get('trailingPE', np.nan),
        'P/E (Forward)': info.get('forwardPE', np.nan),
        'P/S (TTM)': info.get('priceToSalesTrailing12Months', np.nan),
        'P/B': info.get('priceToBook', np.nan),
        'EV/EBITDA': info.get('enterpriseToEbitda', np.nan),
        'PEG Ratio': info.get('pegRatio', np.nan),
    }


def s7_pe_band_accurate(hist, quarterly):
    """Real point-in-time trailing P/E: at each quarter-end, sum the trailing
    4 quarters of actual reported Diluted EPS, then divide the actual price
    on that date by that actual TTM EPS. No constant-EPS assumption.
    Limited by how many quarters yfinance returns (often ~4-8), so the band
    may only cover the trailing year or two, not a full multi-year history."""
    if hist is None or hist.empty:
        return None
    q = quarterly.sort_index(axis=1, ascending=True)
    if 'Diluted EPS' not in q.index:
        return None
    eps_row = q.loc['Diluted EPS'].dropna()
    if len(eps_row) < 4:
        return None
    ttm_eps = eps_row.rolling(4).sum().dropna()
    if ttm_eps.empty:
        return None

    points = []
    hist_tz = hist.index.tz
    for dt, eps_val in ttm_eps.items():
        if pd.isna(eps_val) or eps_val == 0:
            continue
        dt_compare = dt.tz_localize(hist_tz) if hist_tz is not None and dt.tzinfo is None else dt
        future_prices = hist.loc[hist.index >= dt_compare]
        if future_prices.empty:
            continue
        price_at = future_prices['Close'].iloc[0]
        points.append((dt.strftime('%Y-%m-%d'), round(price_at / eps_val, 1)))

    if not points:
        return None

    values = [v for _, v in points]
    current_eps = ttm_eps.iloc[-1]
    current_price = hist['Close'].iloc[-1]
    current_pe = round(current_price / current_eps, 1) if current_eps != 0 else np.nan

    return {'min': round(min(values), 1), 'max': round(max(values), 1),
            'median': round(float(np.median(values)), 1), 'current': current_pe,
            'points': points, 'n_quarters': len(points)}


# ============================================================================
# 8. DUPONT (ROE/ROA breakdown) + FREE CASH FLOW QUALITY
# ============================================================================
def s7_dupont_breakdown(annual):
    a = annual.sort_index(axis=1, ascending=False)
    ni = s7_safe_get(a, 'Net Income', 0)
    equity = s7_safe_get(a, 'Stockholders Equity', 0)
    assets = s7_safe_get(a, 'Total Assets', 0)
    rev = s7_safe_get(a, 'Total Revenue', 0)

    roe = s7_safe_div(ni, equity)
    roa = s7_safe_div(ni, assets)
    net_margin = s7_safe_div(ni, rev)
    asset_turnover = s7_safe_div(rev, assets)
    equity_multiplier = s7_safe_div(assets, equity)

    return {
        'ROE %': round(roe * 100, 1) if not pd.isna(roe) else np.nan,
        'ROA %': round(roa * 100, 1) if not pd.isna(roa) else np.nan,
        'Net Margin %': round(net_margin * 100, 1) if not pd.isna(net_margin) else np.nan,
        'Asset Turnover (x)': asset_turnover,
        'Equity Multiplier (x)': equity_multiplier,
    }


def s7_fcf_quality(annual):
    a = annual.sort_index(axis=1, ascending=False)
    fcf = s7_safe_get(a, 'Free Cash Flow', 0)
    if pd.isna(fcf):
        ocf = s7_safe_get(a, 'Operating Cash Flow', 0)
        capex = s7_safe_get(a, 'Capital Expenditure', 0)
        if not pd.isna(ocf) and not pd.isna(capex):
            fcf = ocf + capex
    rev = s7_safe_get(a, 'Total Revenue', 0)
    ni = s7_safe_get(a, 'Net Income', 0)

    fcf_margin = s7_safe_div(fcf, rev)
    fcf_conversion = s7_safe_div(fcf, ni)

    return {
        'FCF ($B)': round(fcf / 1e9, 2) if not pd.isna(fcf) else np.nan,
        'FCF Margin %': round(fcf_margin * 100, 1) if not pd.isna(fcf_margin) else np.nan,
        'FCF / Net Income (x)': fcf_conversion,
    }


# ============================================================================
# 8.5 DCF INTRINSIC VALUE
# ============================================================================
def s7_dcf_intrinsic_value(annual, info, growth_rate, discount_rate, terminal_growth, years=5):
    a = annual.sort_index(axis=1, ascending=False)
    fcf = s7_safe_get(a, 'Free Cash Flow', 0)
    if pd.isna(fcf):
        ocf = s7_safe_get(a, 'Operating Cash Flow', 0)
        capex = s7_safe_get(a, 'Capital Expenditure', 0)
        if not pd.isna(ocf) and not pd.isna(capex):
            fcf = ocf + capex
    if pd.isna(fcf) or fcf <= 0:
        return None

    total_debt = s7_safe_get(a, 'Total Debt', 0)
    cash = s7_safe_get(a, 'Cash And Cash Equivalents', 0)
    shares = info.get('sharesOutstanding', np.nan)
    if pd.isna(shares) or shares == 0:
        shares = s7_safe_get(a, 'Ordinary Shares Number', 0)
    if pd.isna(shares) or shares == 0:
        return None

    if discount_rate <= terminal_growth:
        return None

    pv_sum = 0.0
    projected = []
    for t in range(1, years + 1):
        fcf_t = fcf * (1 + growth_rate) ** t
        pv = fcf_t / (1 + discount_rate) ** t
        pv_sum += pv
        projected.append((t, round(fcf_t / 1e9, 3), round(pv / 1e9, 3)))

    fcf_final = fcf * (1 + growth_rate) ** years
    terminal_value = fcf_final * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** years

    enterprise_value = pv_sum + pv_terminal
    net_debt = (total_debt if not pd.isna(total_debt) else 0) - (cash if not pd.isna(cash) else 0)
    equity_value = enterprise_value - net_debt
    fair_value_per_share = equity_value / shares

    return {
        'fair_value': round(fair_value_per_share, 2),
        'enterprise_value_b': round(enterprise_value / 1e9, 2),
        'equity_value_b': round(equity_value / 1e9, 2),
        'terminal_value_b': round(terminal_value / 1e9, 2),
        'pv_terminal_pct': round(pv_terminal / enterprise_value * 100, 1) if enterprise_value else np.nan,
        'projected': projected,
    }


def s7_dcf_sensitivity(annual, info, growth_rate, base_discount, base_terminal):
    rows = []
    for dr in [base_discount - 0.02, base_discount, base_discount + 0.02]:
        row = {'Discount Rate': f"{dr*100:.1f}%"}
        for tg in [base_terminal - 0.01, base_terminal, base_terminal + 0.01]:
            res = s7_dcf_intrinsic_value(annual, info, growth_rate, dr, tg)
            row[f"Terminal g={tg*100:.1f}%"] = res['fair_value'] if res else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index('Discount Rate')


# ============================================================================
# 9. ANALYST CONSENSUS, OWNERSHIP, EARNINGS QUALITY, DIVIDENDS
# ============================================================================
def s7_analyst_consensus(info):
    target_mean = info.get('targetMeanPrice', np.nan)
    current = info.get('currentPrice', np.nan)
    upside = np.nan
    if not pd.isna(target_mean) and not pd.isna(current) and current != 0:
        upside = round((target_mean - current) / current * 100, 1)
    return {
        'Recommendation': info.get('recommendationKey', 'n/a'),
        'Mean Rating (1=Strong Buy, 5=Sell)': info.get('recommendationMean', np.nan),
        '# Analysts': info.get('numberOfAnalystOpinions', np.nan),
        'Target Mean': target_mean,
        'Target High': info.get('targetHighPrice', np.nan),
        'Target Low': info.get('targetLowPrice', np.nan),
        'Implied Upside %': upside,
    }


def s7_ownership_snapshot(info, insider_tx):
    ins = info.get('heldPercentInsiders', np.nan)
    inst = info.get('heldPercentInstitutions', np.nan)
    result = {
        'Insider Ownership %': round(ins * 100, 1) if ins is not None and not pd.isna(ins) else np.nan,
        'Institutional Ownership %': round(inst * 100, 1) if inst is not None and not pd.isna(inst) else np.nan,
    }
    activity = 'N/A'
    try:
        if insider_tx is not None and not insider_tx.empty:
            recent = insider_tx.head(10)
            text_col = None
            for c in ['Transaction', 'Text', 'transactionText']:
                if c in recent.columns:
                    text_col = c
                    break
            if text_col:
                buys = recent[text_col].astype(str).str.contains('Buy', case=False, na=False).sum()
                sells = recent[text_col].astype(str).str.contains('Sale|Sell', case=False, na=False).sum()
                activity = f"{buys} buys / {sells} sells (last {len(recent)} filings)"
    except Exception:
        activity = 'Could not parse insider filings'
    result['Recent Insider Activity'] = activity
    return result


def s7_earnings_quality(earnings_hist, earnings_dates):
    result = {'beat_rate': 'N/A', 'next_earnings': 'N/A'}
    try:
        if earnings_hist is not None and not earnings_hist.empty:
            recent = earnings_hist.tail(4)
            if 'epsActual' in recent.columns and 'epsEstimate' in recent.columns:
                beats = int((recent['epsActual'] > recent['epsEstimate']).sum())
                result['beat_rate'] = f"{beats}/{len(recent)} beats (last {len(recent)} qtrs)"
    except Exception:
        pass
    try:
        if earnings_dates is not None and not earnings_dates.empty:
            now = pd.Timestamp.now(tz=earnings_dates.index.tz) if earnings_dates.index.tz else pd.Timestamp.now()
            future = earnings_dates[earnings_dates.index > now]
            if len(future) > 0:
                result['next_earnings'] = future.index.min().strftime('%Y-%m-%d')
    except Exception:
        pass
    return result


def s7_dividend_buyback(info, ratios):
    div_yield = info.get('dividendYield', np.nan)
    payout = info.get('payoutRatio', np.nan)
    buyback_yield = -ratios.get('Share Count YoY %', np.nan) if not pd.isna(ratios.get('Share Count YoY %', np.nan)) else np.nan
    dy = div_yield if div_yield else 0.0
    return {
        'Dividend Yield %': round(dy, 2) if not pd.isna(dy) else 0.0,
        'Payout Ratio %': round(payout * 100, 1) if payout and not pd.isna(payout) else np.nan,
        'Buyback Yield %': round(buyback_yield, 1) if not pd.isna(buyback_yield) else np.nan,
    }


def s7_short_interest_snapshot(info):
    spf = info.get('shortPercentOfFloat', np.nan)
    spo = info.get('sharesPercentSharesOut', np.nan)
    return {
        'Short % of Float': round(spf * 100, 2) if spf is not None and not pd.isna(spf) else np.nan,
        'Shares Short': info.get('sharesShort', np.nan),
        'Short Ratio (days to cover)': info.get('shortRatio', np.nan),
        'Short % of Shares Out': round(spo * 100, 2) if spo is not None and not pd.isna(spo) else np.nan,
    }


# ============================================================================
# 10. TECHNICAL CONFIRMATION LAYER
# ============================================================================
def s7_technical_indicators(hist):
    if hist is None or hist.empty or len(hist) < 50:
        return None
    close = hist['Close']
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
    price = close.iloc[-1]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_now = rsi.iloc[-1]

    if not pd.isna(sma200) and price > sma50 > sma200:
        trend = 'Uptrend (price > 50MA > 200MA)'
    elif not pd.isna(sma200) and price < sma50 < sma200:
        trend = 'Downtrend (price < 50MA < 200MA)'
    else:
        trend = 'Mixed / Transitional'

    if pd.isna(rsi_now):
        rsi_note = 'N/A'
    elif rsi_now >= 70:
        rsi_note = 'Overbought'
    elif rsi_now <= 30:
        rsi_note = 'Oversold'
    else:
        rsi_note = 'Neutral'

    return {'Price': round(price, 2), 'SMA50': round(sma50, 2) if not pd.isna(sma50) else np.nan,
            'SMA200': round(sma200, 2) if not pd.isna(sma200) else np.nan,
            'Trend': trend, 'RSI(14)': round(rsi_now, 1) if not pd.isna(rsi_now) else np.nan,
            'RSI Note': rsi_note}


def s7_technical_divergence_note(verdict, technical):
    if not technical:
        return "Not enough price history for a technical read."
    if verdict == "BUY" and 'Downtrend' in technical['Trend']:
        return "Fundamentals say buy, but price is still in a downtrend — no technical confirmation yet."
    if verdict in ("AVOID", "AVOID / HIGH RISK") and 'Uptrend' in technical['Trend']:
        return "Price is rising despite weak fundamentals — that's momentum/euphoria risk, not a quality rally."
    return "Technicals broadly align with the fundamental verdict."


# ============================================================================
# 11. FORENSIC RED FLAGS
# ============================================================================
def s7_forensic_checks(annual):
    a = annual.sort_index(axis=1, ascending=False)
    ni = s7_safe_get(a, 'Net Income', 0)
    ocf = s7_safe_get(a, 'Operating Cash Flow', 0)
    assets = s7_safe_get(a, 'Total Assets', 0)
    goodwill = s7_safe_get(a, 'Goodwill', 0)
    rev = s7_safe_get(a, 'Total Revenue', 0)
    rev_prior = s7_safe_get(a, 'Total Revenue', 1)
    ar = s7_safe_get(a, 'Accounts Receivable', 0)
    ar_prior = s7_safe_get(a, 'Accounts Receivable', 1)

    flags = []
    accrual_ratio = np.nan
    if not pd.isna(ni) and not pd.isna(ocf) and not pd.isna(assets) and assets != 0:
        accrual_ratio = (ni - ocf) / assets * 100
        if accrual_ratio > 5:
            flags.append(f"High accruals ratio ({accrual_ratio:.1f}%) — net income running well ahead of "
                          f"cash flow, an earnings-quality red flag")

    goodwill_pct = np.nan
    if not pd.isna(goodwill) and not pd.isna(assets) and assets != 0:
        goodwill_pct = goodwill / assets * 100
        if goodwill_pct > 30:
            flags.append(f"Goodwill/intangibles are {goodwill_pct:.0f}% of total assets — impairment risk "
                          f"if growth stalls")

    ar_growth = s7_pct(ar, ar_prior)
    rev_growth = s7_pct(rev, rev_prior)
    if not pd.isna(ar_growth) and not pd.isna(rev_growth) and ar_growth > rev_growth + 15:
        flags.append(f"Receivables growing much faster than revenue ({ar_growth}% vs {rev_growth}%) — "
                      f"possible channel stuffing or collection issues")

    return {'Accruals Ratio %': round(accrual_ratio, 1) if not pd.isna(accrual_ratio) else np.nan,
            'Goodwill/Assets %': round(goodwill_pct, 1) if not pd.isna(goodwill_pct) else np.nan,
            'Receivables YoY %': ar_growth, 'Revenue YoY %': rev_growth, 'flags': flags}


# ============================================================================
# 12. PEER / SECTOR / PORTFOLIO SNAPSHOTS
# ============================================================================
def s7_quick_peer_snapshot(ticker):
    try:
        annual, quarterly = s7_fetch_statements(ticker)
        if annual.empty:
            return None
        ratios = s7_compute_ratios(annual, quarterly)
        pillar_results = s7_score_pillars(ratios)
        verdict, _ = s7_overall_verdict(pillar_results)
        scores = [v['score'] for v in pillar_results.values() if v['score'] is not None]
        overall_score = round(np.mean(scores)) if scores else None
        return {'Ticker': ticker, 'Verdict': verdict,
                'Overall Score': overall_score if overall_score is not None else 'N/A',
                'ROCE %': ratios.get('ROCE %', np.nan), 'Net Margin %': ratios.get('Net Margin %', np.nan),
                'Debt/Equity': ratios.get('Debt/Equity', np.nan), 'Revenue YoY %': ratios.get('Revenue YoY %', np.nan)}
    except Exception:
        return None


@st.cache_data(ttl=3600)
def s7_fetch_portfolio_prices(tickers, benchmark='SPY', period='1y'):
    all_tickers = list(dict.fromkeys(tickers + [benchmark]))
    data = {}
    for t in all_tickers:
        try:
            h = yf.Ticker(t).history(period=period)
            if not h.empty:
                data[t] = h['Close']
        except Exception:
            continue
    if not data:
        return None
    return pd.DataFrame(data).dropna(how='all')


def s7_compute_beta(returns, stock_col, bench_col):
    if stock_col not in returns.columns or bench_col not in returns.columns:
        return np.nan
    aligned = returns[[stock_col, bench_col]].dropna()
    if len(aligned) < 20:
        return np.nan
    cov = aligned[stock_col].cov(aligned[bench_col])
    var = aligned[bench_col].var()
    return round(cov / var, 2) if var else np.nan


# ============================================================================
# 13. COMPOSITE GRADE + AUTO-GENERATED THESIS
# ============================================================================
def s7_composite_grade(pillar_results, roce_declining, forensic_flags):
    scores = [v['score'] for v in pillar_results.values() if v['score'] is not None]
    base = np.mean(scores) if scores else 50
    penalty = 0
    if roce_declining:
        penalty += 20
    penalty += min(len(forensic_flags), 3) * 8
    final = max(0, base - penalty)
    if final >= 85:
        grade = 'A'
    elif final >= 70:
        grade = 'B'
    elif final >= 55:
        grade = 'C'
    elif final >= 40:
        grade = 'D'
    else:
        grade = 'F'
    return grade, round(final)


def s7_generate_thesis(ticker, verdict, pillar_results, ratios, roce_declining, val, analyst, technical, forensic):
    lines = [f"**{ticker}: {verdict}.**"]
    q = pillar_results.get("Pillar 1: Quality", {})
    s = pillar_results.get("Pillar 2: Safety", {})
    lines.append(f"Quality is {q.get('verdict', 'N/A')} (ROCE {ratios.get('ROCE %', 'N/A')}%, "
                  f"net margin {ratios.get('Net Margin %', 'N/A')}%); "
                  f"Safety is {s.get('verdict', 'N/A')} (D/E {ratios.get('Debt/Equity', 'N/A')}).")
    if roce_declining:
        lines.append("Capital efficiency is deteriorating over the last 3 quarters — "
                      "treat any bullish price action with caution.")
    if forensic.get('flags'):
        lines.append("Forensic flags: " + "; ".join(forensic['flags']) + ".")
    if val:
        lines.append(f"Price sits in the {val['zone'].split(' (')[0]} of its 2-year range.")
    if analyst.get('Recommendation') not in (None, 'n/a'):
        lines.append(f"Street consensus: {analyst.get('Recommendation')} "
                      f"({analyst.get('# Analysts', '?')} analysts, target ${analyst.get('Target Mean', 'n/a')}, "
                      f"implied upside {analyst.get('Implied Upside %', 'n/a')}%).")
    if technical:
        lines.append(f"Technicals show a {technical['Trend'].lower()}, RSI {technical['RSI Note'].lower()}.")
    return " ".join(lines)


# ============================================================================


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
# 🎯 TAB 5: FINALIZED INTRADAY SQUEEZE BREAKOUT ENGINE
# ==============================================================================
with tab5:
    st.header("🎯 High-Effectiveness Intraday Squeeze Scanner")
    st.caption("Filters for Volatility Squeezes with Multi-Timeframe Institutional Confirmation")

    # 1. SETUP WATCHLIST & SLIDERS
    if 'WATCHLIST_PATH' not in locals(): WATCHLIST_PATH = "watchlist.csv"

    if os.path.exists(WATCHLIST_PATH):
        tab5_watchlist_df = pd.read_csv(WATCHLIST_PATH)
    else:
        st.error(f"❌ Watchlist file not found: {WATCHLIST_PATH}")
        tab5_watchlist_df = pd.DataFrame()

    # Sidebar Sliders (These define your sensitivity)
    st.sidebar.subheader("🎛️ Squeeze Scanner Sensitivity")
    vol_slider = st.sidebar.slider("Squeeze Threshold (%)", 0.5, 5.0, 1.2, 0.1) / 100
    cmf_slider = st.sidebar.slider("CMF Institutional Threshold", -0.5, 0.5, 0.10, 0.05)

    if not tab5_watchlist_df.empty:
        ticker_col = next(
            (col for col in ["Yahoo Ticker", "ticker", "Ticker", "Symbol"] if col in tab5_watchlist_df.columns),
            tab5_watchlist_df.columns[0])
        name_col = next(
            (col for col in ["Company Name", "name", "Name", "Company"] if col in tab5_watchlist_df.columns),
            ticker_col)
        tab5_tickers = [str(t).strip().upper() for t in tab5_watchlist_df[ticker_col] if
                        pd.notna(t) and str(t).strip().upper() not in ["NAN", "NONE"]]

        if st.button("🔄 Execute Live Intraday Scan", key="final_squeeze_scan"):
            squeeze_rows = []

            with st.spinner("Streaming market data..."):
                master_1h = yf.download(tab5_tickers, period="1mo", interval="1h", progress=False, group_by="ticker",
                                        prepost=True)
                master_5m = yf.download(tab5_tickers, period="5d", interval="5m", progress=False, group_by="ticker",
                                        prepost=True)

            if master_5m.empty or master_1h.empty:
                st.info("🌙 Market data stream is empty or unavailable.")
            else:
                is_multi = len(tab5_tickers) > 1

                for t in tab5_tickers:
                    try:
                        df_1h = master_1h[t].copy() if is_multi else master_1h.copy()
                        df_5m = master_5m[t].copy() if is_multi else master_5m.copy()

                        df_1h.columns = [str(c).capitalize() for c in df_1h.columns]
                        df_5m.columns = [str(c).capitalize() for c in df_5m.columns]

                        if len(df_1h) < 50 or len(df_5m) < 25: continue

                        # --- LOGIC ENGINE ---
                        df_1h['EMA_50'] = df_1h['Close'].ewm(span=50, adjust=False).mean()
                        macro_uptrend = float(df_1h['Close'].iloc[-1]) > float(df_1h['EMA_50'].iloc[-1])

                        df_5m['High_20'] = df_5m['High'].rolling(20).max()
                        df_5m['Low_20'] = df_5m['Low'].rolling(20).min()
                        c_last = float(df_5m['Close'].iloc[-1])
                        high_box = float(df_5m['High_20'].iloc[-1])
                        low_box = float(df_5m['Low_20'].iloc[-1])

                        # --- USE SLIDER VARIABLES HERE ---
                        box_width_pct = (high_box - low_box) / c_last
                        is_squeezed = box_width_pct <= vol_slider  # Uses slider
                        price_breakout = c_last > high_box

                        df_5m['CMF'] = ta.cmf(df_5m['High'], df_5m['Low'], df_5m['Close'], df_5m['Volume'], length=20)
                        cmf_val = float(df_5m['CMF'].iloc[-1]) if not np.isnan(df_5m['CMF'].iloc[-1]) else 0.0
                        volume_confirmed = cmf_val >= cmf_slider  # Uses slider

                        # Decision Signal
                        if macro_uptrend and is_squeezed and price_breakout and volume_confirmed:
                            decision = "🔥 STRONG BUY SETUP"
                        elif macro_uptrend and is_squeezed:
                            decision = "⏳ Squeezed (Waiting for Breakout)"
                        else:
                            decision = "❌ No Squeeze Setup"

                        squeeze_rows.append({
                            "Company": tab5_watchlist_df[tab5_watchlist_df[ticker_col] == t][name_col].iloc[0],
                            "Ticker": t,
                            "Decision": decision,
                            "Price": round(c_last, 2),
                            "Squeeze Width": f"{round(box_width_pct * 100, 2)}%",
                            "CMF": round(cmf_val, 2),
                            "Stop Loss": round(low_box, 2)
                        })
                    except Exception:
                        continue

            if squeeze_rows:
                df_res = pd.DataFrame(squeeze_rows)
                df_res["Order"] = df_res["Decision"].map(
                    {"🔥 STRONG BUY SETUP": 0, "⏳ Squeezed (Waiting for Breakout)": 1, "❌ No Squeeze Setup": 2}).fillna(
                    3)
                df_res = df_res.sort_values("Order").drop(columns=["Order"])


                def style_rows(row):
                    color = ""
                    if "STRONG BUY" in row["Decision"]:
                        color = "background-color: #2ECC71; color: black; font-weight: bold;"
                    elif "Squeezed" in row["Decision"]:
                        color = "background-color: #F1C40F; color: black;"
                    return [color] * len(row)


                st.dataframe(df_res.style.apply(style_rows, axis=1), use_container_width=True, hide_index=True)
            else:
                st.info("No squeeze setups detected with current settings. Try widening the sliders!")

# 6. Add your scanner code into tab6

with tab6:
    st.markdown("# 🦅 Institutional Alpha Matrix")
    st.markdown("Real-time confluence tracking of sector rotation.")

    # 1. Configuration: Expanded sector map
    sector_map = {
        "Tech (QQQ)": {"core": "QQQ", "bull": "TQQQ", "bear": "SQQQ"},
        "S&P 500 (SPY)": {"core": "SPY", "bull": "UPRO", "bear": "SPXU"},
        "Semis (SOXX)": {"core": "SOXX", "bull": "SOXL", "bear": "SOXS"},
        "Energy (USO)": {"core": "USO", "bull": "GUSH", "bear": "DRIP"},
        "Biotech (XBI)": {"core": "XBI", "bull": "LABU", "bear": "LABD"},
        "Financial (XLF)": {"core": "XLF", "bull": "FAS", "bear": "FAZ"},
        "Gold (GDX)": {"core": "GDX", "bull": "NUGT", "bear": "DUST"},
        "Real Estate (XLRE)": {"core": "XLRE", "bull": "DRN", "bear": "DRV"}
    }


    # 2. Scanning logic
    @st.fragment(run_every=30)
    def scan():
        tickers = list(set([cfg[k] for cfg in sector_map.values() for k in ["core", "bull", "bear"]]))

        # Pull 5 days of data for the 50-period EMA
        data = yf.download(tickers, period="5d", interval="15m", progress=False)

        rows = []
        for label, cfg in sector_map.items():
            core = cfg["core"]
            if core not in data['Close'].columns: continue

            # Get Price and Volume
            close = data['Close'][core]
            vol = data['Volume'][core]

            # 1. Trend Filter: 50-period EMA (The "Line in the Sand")
            ema50 = ta.ema(close, length=50)

            # 2. Institutional Flow: VWAP
            vwap = (close * vol).cumsum() / vol.cumsum()

            # 3. Volume Spike: Must be 1.5x of the 20-period Average
            vol_sma = ta.sma(vol, length=20)
            is_vol_spike = vol.iloc[-1] > (vol_sma.iloc[-1] * 1.5)

            # Current Status
            curr_price = close.iloc[-1]
            curr_ema = ema50.iloc[-1]
            curr_vwap = vwap.iloc[-1]

            # Strategy:
            # Bullish: Price > EMA50 (Trend) AND Price > VWAP (Strength) AND Vol Spike
            # Bearish: Price < EMA50 (Trend) AND Price < VWAP (Weakness) AND Vol Spike

            signal = "⏳ MONITORING"

            if curr_price > curr_ema and curr_price > curr_vwap and is_vol_spike:
                signal = f"🚀 TREND LONG: {cfg['bull']}"
            elif curr_price < curr_ema and curr_price < curr_vwap and is_vol_spike:
                signal = f"📉 TREND SHORT: {cfg['bear']}"

            rows.append({
                "Sector": label,
                "Price": f"${curr_price:.2f}",
                "Trend": "Bullish" if curr_price > curr_ema else "Bearish",
                "ACTION": signal
            })

        if rows:
            df = pd.DataFrame(rows)

            # Styling for high-visibility
            def highlight_signals(val):
                if "TREND LONG" in str(val): return 'background-color: #004d26; color: #00FFCC; font-weight: bold'
                if "TREND SHORT" in str(val): return 'background-color: #8b0000; color: #FF9999; font-weight: bold'
                return ''

            styled_df = df.style.map(highlight_signals, subset=['ACTION'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("Scanner calibrating...")

# ==============================================================================
# 🏛️ TAB 7: FUNDAMENTAL & PORTFOLIO TERMINAL
# ==============================================================================
with tab7:
    st.header("🏛️ Fundamental & Portfolio Terminal")
    st.caption("4-Pillar fundamental scorecard, ROCE trend, DCF, DuPont/FCF quality, ownership & analyst "
               "consensus, short interest & options, news, technical cross-check, forensic red flags, and "
               "portfolio-level view — separate from the swing/technical tabs above. Use this for the "
               "'should I own this at all' question; use the other tabs for timing.")

    s7_macro = s7_fetch_macro()
    s7_mcols = st.columns(3)
    for i, (k, v) in enumerate(s7_macro.items()):
        s7_mcols[i].metric(k, f"{v}" if not pd.isna(v) else "N/A")

    s7_mode = st.radio("Mode", ["Single Stock Analysis", "My Portfolio"], horizontal=True, key="s7_mode")

    # --------------------------------------------------------------------
    # SINGLE STOCK MODE
    # --------------------------------------------------------------------
    if s7_mode == "Single Stock Analysis":
        s7_ticker = st.text_input("Enter Ticker:", "AAPL", key="s7_ticker_input").upper()

        if st.button("Analyze", key="s7_analyze_btn"):
            with st.spinner(f"Pulling {s7_ticker}..."):
                s7_annual, s7_quarterly = s7_fetch_statements(s7_ticker)
                s7_hist, s7_info = s7_fetch_price_data(s7_ticker)
                s7_ext = s7_fetch_extended_data(s7_ticker)
                s7_news = s7_fetch_news(s7_ticker)
                s7_options_snap = s7_fetch_options_snapshot(s7_ticker)

            if s7_annual.empty:
                st.error("No data returned — check the ticker (note: ETFs won't have financial statements).")
            else:
                s7_ratios = s7_compute_ratios(s7_annual, s7_quarterly)
                s7_pillar_results = s7_score_pillars(s7_ratios)
                s7_verdict, s7_reason = s7_overall_verdict(s7_pillar_results)
                s7_val = s7_valuation_timing(s7_hist, s7_info, s7_verdict)
                s7_roce_declining, s7_roce_series = s7_roce_trend_flag(s7_quarterly)
                s7_forensic = s7_forensic_checks(s7_annual)
                s7_analyst = s7_analyst_consensus(s7_info)
                s7_technical = s7_technical_indicators(s7_hist)
                s7_grade, s7_grade_score = s7_composite_grade(s7_pillar_results, s7_roce_declining, s7_forensic['flags'])
                s7_thesis = s7_generate_thesis(s7_ticker, s7_verdict, s7_pillar_results, s7_ratios, s7_roce_declining,
                                                s7_val, s7_analyst, s7_technical, s7_forensic)

                s7_icon = {"BUY": "🟢", "HOLD": "🟡", "AVOID": "🔴",
                           "AVOID / HIGH RISK": "🔴", "NO DATA": "⚪"}.get(s7_verdict, "⚪")
                s7_c1, s7_c2 = st.columns([1, 4])
                with s7_c1:
                    st.metric("Composite Grade", f"{s7_grade}", f"{s7_grade_score}/100")
                with s7_c2:
                    st.subheader(f"{s7_icon} {s7_verdict}")
                st.write(s7_thesis)

                if s7_val:
                    s7_action = s7_val['action']
                    if s7_roce_declining:
                        s7_action = "EXIT — ROCE deteriorating for 3+ straight quarters"
                    st.write(f"**Price:** ${s7_val['price']:.2f}  |  **2Y Range Position:** {s7_val['position_pct']}% "
                             f"({s7_val['zone']})  |  **Action:** {s7_action}")
                if s7_roce_declining:
                    s7_trend_str = " → ".join(f"{d}: {v}%" for d, v in s7_roce_series)
                    st.warning(f"⚠️ ROCE trend declining: {s7_trend_str}")
                if s7_forensic['flags']:
                    st.warning("🚩 " + " | ".join(s7_forensic['flags']))

                s7_tabs = st.tabs(["📊 Pillar Scorecard", "💰 Valuation", "🎯 Intrinsic Value (DCF)", "🔬 DuPont & FCF",
                                   "👥 Ownership & Analysts", "🩳 Short Interest & Options", "📰 News",
                                   "📉 Technicals", "🚩 Red Flags", "⚖️ Peer Comparison", "📄 Raw Financials"])

                with s7_tabs[0]:
                    s7_cols = st.columns(4)
                    for i, (pillar, res) in enumerate(s7_pillar_results.items()):
                        with s7_cols[i]:
                            st.subheader(pillar.split(": ")[1])
                            s7_score_label = f"{res['score']}/100" if res['score'] is not None else "N/A"
                            st.metric("Score", s7_score_label, res['verdict'])
                            for name, v, s in res['detail']:
                                mark = "✅" if s == 2 else "⚠️" if s == 1 else "❌" if s == 0 else "—"
                                v_str = f"{v}" if not pd.isna(v) else "N/A"
                                st.write(f"{mark} {name}: {v_str}")

                with s7_tabs[1]:
                    st.subheader("Valuation multiples")
                    s7_mult = s7_valuation_multiples(s7_info)
                    s7_vcols = st.columns(3)
                    for i, (k, v) in enumerate(s7_mult.items()):
                        s7_vcols[i % 3].metric(k, f"{v:.2f}" if isinstance(v, (int, float)) and not pd.isna(v) else "N/A")

                    st.subheader("Historical P/E band (actual trailing-4Q EPS at each date)")
                    s7_band = s7_pe_band_accurate(s7_hist, s7_quarterly)
                    if s7_band:
                        b1, b2, b3, b4 = st.columns(4)
                        b1.metric("Low", s7_band['min'])
                        b2.metric("Median", s7_band['median'])
                        b3.metric("High", s7_band['max'])
                        b4.metric("Current", s7_band['current'])
                        st.caption(f"Based on {s7_band['n_quarters']} actual quarterly EPS points.")
                    else:
                        st.write("Not enough quarterly EPS history for a real P/E band.")
                    if s7_val:
                        st.line_chart(s7_hist['Close'])

                with s7_tabs[2]:
                    st.subheader("Discounted Cash Flow — intrinsic value")
                    st.caption("Adjust the assumptions — there's no single 'correct' discount rate or growth rate.")

                    s7_default_growth = s7_ratios.get('Revenue YoY %', np.nan)
                    s7_default_growth = min(max(s7_default_growth / 100, 0.02), 0.25) if not pd.isna(s7_default_growth) else 0.08

                    dc1, dc2, dc3, dc4 = st.columns(4)
                    s7_growth_input = dc1.slider("FCF growth rate (yrs 1-5)", 0.0, 0.30,
                                                  float(round(s7_default_growth, 2)), 0.01, key="s7_dcf_growth")
                    s7_discount_input = dc2.slider("Discount rate (WACC proxy)", 0.05, 0.15, 0.10, 0.005, key="s7_dcf_discount")
                    s7_terminal_input = dc3.slider("Terminal growth rate", 0.0, 0.04, 0.025, 0.0025, key="s7_dcf_terminal")
                    s7_years_input = dc4.selectbox("Projection years", [3, 5, 7, 10], index=1, key="s7_dcf_years")

                    s7_dcf = s7_dcf_intrinsic_value(s7_annual, s7_info, s7_growth_input, s7_discount_input,
                                                     s7_terminal_input, s7_years_input)

                    if s7_dcf is None:
                        st.write("Couldn't compute a DCF — missing FCF/shares data, or discount rate isn't above "
                                 "terminal growth.")
                    else:
                        s7_current_price = s7_val['price'] if s7_val else np.nan
                        s7_margin_of_safety = np.nan
                        if not pd.isna(s7_current_price) and s7_current_price != 0:
                            s7_margin_of_safety = round((s7_dcf['fair_value'] - s7_current_price) / s7_current_price * 100, 1)

                        r1, r2, r3 = st.columns(3)
                        r1.metric("Fair Value / Share", f"${s7_dcf['fair_value']}")
                        r2.metric("Current Price", f"${s7_current_price:.2f}" if not pd.isna(s7_current_price) else "N/A")
                        r3.metric("Margin of Safety", f"{s7_margin_of_safety}%" if not pd.isna(s7_margin_of_safety) else "N/A")

                        if not pd.isna(s7_margin_of_safety):
                            if s7_margin_of_safety >= 20:
                                st.success("Trading well below DCF fair value — margin-of-safety territory.")
                            elif s7_margin_of_safety <= -20:
                                st.warning("Trading well above DCF fair value — priced for a lot of future growth.")
                            else:
                                st.info("Roughly in line with DCF fair value — no strong signal either way.")

                        st.caption(f"Enterprise Value ${s7_dcf['enterprise_value_b']}B · Equity Value ${s7_dcf['equity_value_b']}B "
                                   f"· Terminal Value ${s7_dcf['terminal_value_b']}B ({s7_dcf['pv_terminal_pct']}% of EV).")

                        s7_proj_df = pd.DataFrame(s7_dcf['projected'], columns=['Year', 'Projected FCF ($B)', 'PV of FCF ($B)'])
                        st.dataframe(s7_proj_df.set_index('Year'), use_container_width=True)

                        st.subheader("Sensitivity: fair value across discount rate × terminal growth")
                        s7_sens = s7_dcf_sensitivity(s7_annual, s7_info, s7_growth_input, s7_discount_input, s7_terminal_input)
                        st.dataframe(s7_sens, use_container_width=True)

                with s7_tabs[3]:
                    st.subheader("DuPont breakdown (ROE = Net Margin × Asset Turnover × Equity Multiplier)")
                    s7_dp = s7_dupont_breakdown(s7_annual)
                    s7_dcols = st.columns(5)
                    for i, (k, v) in enumerate(s7_dp.items()):
                        s7_dcols[i].metric(k, f"{v}" if not pd.isna(v) else "N/A")

                    st.subheader("Free cash flow quality")
                    s7_fq = s7_fcf_quality(s7_annual)
                    s7_fcols = st.columns(3)
                    for i, (k, v) in enumerate(s7_fq.items()):
                        s7_fcols[i].metric(k, f"{v}" if not pd.isna(v) else "N/A")

                with s7_tabs[4]:
                    st.subheader("Analyst consensus")
                    s7_acols = st.columns(4)
                    for i, (k, v) in enumerate(s7_analyst.items()):
                        s7_acols[i % 4].metric(k, f"{v}" if not (isinstance(v, float) and pd.isna(v)) else "N/A")

                    st.subheader("Ownership")
                    s7_own = s7_ownership_snapshot(s7_info, s7_ext.get('insider_tx'))
                    s7_ocols = st.columns(3)
                    for i, (k, v) in enumerate(s7_own.items()):
                        s7_ocols[i % 3].metric(k, f"{v}" if not (isinstance(v, float) and pd.isna(v)) else "N/A")

                    st.subheader("Earnings quality")
                    s7_eq = s7_earnings_quality(s7_ext.get('earnings_hist'), s7_ext.get('earnings_dates'))
                    s7_ecols = st.columns(2)
                    s7_ecols[0].metric("Beat/Miss (last 4 qtrs)", s7_eq['beat_rate'])
                    s7_ecols[1].metric("Next Earnings Date", s7_eq['next_earnings'])

                    st.subheader("Dividends & buybacks")
                    s7_db = s7_dividend_buyback(s7_info, s7_ratios)
                    s7_dbcols = st.columns(3)
                    for i, (k, v) in enumerate(s7_db.items()):
                        s7_dbcols[i].metric(k, f"{v}" if not (isinstance(v, float) and pd.isna(v)) else "N/A")

                with s7_tabs[5]:
                    st.subheader("Short interest")
                    s7_si = s7_short_interest_snapshot(s7_info)
                    s7_sicols = st.columns(4)
                    for i, (k, v) in enumerate(s7_si.items()):
                        s7_sicols[i].metric(k, f"{v}" if not (isinstance(v, float) and pd.isna(v)) else "N/A")

                    st.subheader("Options (nearest expiry)")
                    if s7_options_snap:
                        s7_ocols2 = st.columns(3)
                        for i, (k, v) in enumerate(s7_options_snap.items()):
                            s7_ocols2[i % 3].metric(k, f"{v}")
                    else:
                        st.write("No options data available for this ticker.")

                with s7_tabs[6]:
                    st.subheader("Recent news")
                    if s7_news:
                        for item in s7_news[:10]:
                            content = item.get('content', item)
                            title = content.get('title', 'Untitled')
                            link = content.get('canonicalUrl', {}).get('url') if isinstance(content.get('canonicalUrl'), dict) else content.get('link', '')
                            publisher = content.get('provider', {}).get('displayName') if isinstance(content.get('provider'), dict) else content.get('publisher', '')
                            st.write(f"**[{title}]({link})** — {publisher}")
                    else:
                        st.write("No recent news found.")

                with s7_tabs[7]:
                    if s7_technical:
                        s7_tcols = st.columns(5)
                        for i, (k, v) in enumerate(s7_technical.items()):
                            s7_tcols[i % 5].metric(k, f"{v}")
                        st.write(s7_technical_divergence_note(s7_verdict, s7_technical))
                        st.line_chart(s7_hist['Close'])
                    else:
                        st.write("Not enough price history (need 50+ trading days).")

                with s7_tabs[8]:
                    st.subheader("Forensic red flags")
                    s7_frcols = st.columns(4)
                    s7_frcols[0].metric("Accruals Ratio %", s7_forensic['Accruals Ratio %'])
                    s7_frcols[1].metric("Goodwill/Assets %", s7_forensic['Goodwill/Assets %'])
                    s7_frcols[2].metric("Receivables YoY %", s7_forensic['Receivables YoY %'])
                    s7_frcols[3].metric("Revenue YoY %", s7_forensic['Revenue YoY %'])
                    if s7_forensic['flags']:
                        for f in s7_forensic['flags']:
                            st.error(f)
                    else:
                        st.success("No forensic red flags triggered on the current thresholds.")

                with s7_tabs[9]:
                    st.subheader("Peer / sector comparison")
                    s7_peer_input = st.text_input("Peer tickers (comma-separated):", "", key="s7_peer_input")
                    if st.button("Compare Peers", key="s7_compare_peers_btn"):
                        s7_peer_tickers = [t.strip().upper() for t in s7_peer_input.split(",") if t.strip()]
                        if not s7_peer_tickers:
                            st.write("Enter at least one peer ticker.")
                        else:
                            s7_rows = [s7_quick_peer_snapshot(s7_ticker)]
                            for pt in s7_peer_tickers:
                                snap = s7_quick_peer_snapshot(pt)
                                if snap:
                                    s7_rows.append(snap)
                            s7_rows = [r for r in s7_rows if r]
                            if s7_rows:
                                st.dataframe(pd.DataFrame(s7_rows).set_index('Ticker'), use_container_width=True)
                            else:
                                st.write("Couldn't pull data for any of the tickers entered.")

                with s7_tabs[10]:
                    s7_display_df = s7_build_display_table(s7_annual, s7_quarterly)
                    s7_formatted = s7_format_display_table(s7_display_df)
                    st.dataframe(s7_formatted, use_container_width=True)
                    st.caption("Currency items in $B · EPS in $/share · share counts in millions · tax rate and ROCE in %.")

    # --------------------------------------------------------------------
    # PORTFOLIO MODE
    # --------------------------------------------------------------------
    elif s7_mode == "My Portfolio":
        st.subheader("Portfolio holdings")
        st.caption("One holding per line: TICKER, SHARES, COST_BASIS_PER_SHARE  (e.g. META,50,320)")
        s7_holdings_text = st.text_area("Holdings", "META,50,320\nGOOGL,20,140\nPLTR,100,25",
                                         height=120, key="s7_holdings_text")

        if st.button("Analyze Portfolio", key="s7_analyze_portfolio_btn"):
            s7_holdings = []
            for line in s7_holdings_text.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0]:
                    try:
                        tk = parts[0].upper()
                        shares = float(parts[1])
                        cost = float(parts[2]) if len(parts) > 2 and parts[2] else np.nan
                        s7_holdings.append((tk, shares, cost))
                    except ValueError:
                        continue

            if not s7_holdings:
                st.write("Enter at least one valid holding.")
            else:
                s7_tickers_list = [h[0] for h in s7_holdings]
                with st.spinner("Pulling portfolio data..."):
                    s7_price_df = s7_fetch_portfolio_prices(s7_tickers_list)
                    s7_rows = []
                    for tk, shares, cost in s7_holdings:
                        snap = s7_quick_peer_snapshot(tk)
                        if not snap:
                            continue
                        price = np.nan
                        if s7_price_df is not None and tk in s7_price_df.columns:
                            s = s7_price_df[tk].dropna()
                            price = s.iloc[-1] if not s.empty else np.nan
                        value = shares * price if not pd.isna(price) else np.nan
                        pl = (price - cost) * shares if not pd.isna(price) and not pd.isna(cost) else np.nan
                        pl_pct = round((price - cost) / cost * 100, 1) if not pd.isna(price) and not pd.isna(cost) and cost != 0 else np.nan
                        s7_rows.append({**snap, 'Shares': shares,
                                         'Price': round(price, 2) if not pd.isna(price) else np.nan,
                                         'Position Value': round(value, 2) if not pd.isna(value) else np.nan,
                                         'Cost Basis': cost,
                                         'Unrealized P/L': round(pl, 2) if not pd.isna(pl) else np.nan,
                                         'P/L %': pl_pct})

                if s7_rows:
                    s7_pdf = pd.DataFrame(s7_rows).set_index('Ticker')
                    st.dataframe(s7_pdf, use_container_width=True)

                    s7_total_value = s7_pdf['Position Value'].sum(skipna=True)
                    s7_total_pl = s7_pdf['Unrealized P/L'].sum(skipna=True)
                    s7_avg_score = pd.to_numeric(s7_pdf['Overall Score'], errors='coerce').dropna()

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Portfolio Value", f"${s7_total_value:,.0f}")
                    c2.metric("Total Unrealized P/L", f"${s7_total_pl:,.0f}")
                    c3.metric("Weighted Avg Score", f"{s7_avg_score.mean():.0f}/100" if not s7_avg_score.empty else "N/A")

                    if s7_price_df is not None:
                        st.subheader("Correlation matrix (1Y daily returns)")
                        s7_returns = s7_price_df.pct_change().dropna(how='all')
                        s7_valid_tickers = [t for t in s7_tickers_list if t in s7_returns.columns]
                        if len(s7_valid_tickers) >= 2:
                            s7_corr = s7_returns[s7_valid_tickers].corr()
                            st.dataframe(s7_corr, use_container_width=True)

                        st.subheader("Beta vs SPY")
                        s7_beta_rows = [{'Ticker': tk, 'Beta vs SPY': s7_compute_beta(s7_returns, tk, 'SPY')} for tk in s7_valid_tickers]
                        st.dataframe(pd.DataFrame(s7_beta_rows).set_index('Ticker'), use_container_width=True)
                else:
                    st.write("Couldn't pull data for any holdings entered.")
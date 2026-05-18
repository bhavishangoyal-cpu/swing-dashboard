import streamlit as st
import pandas as pd
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
import numpy as np
# ---------------- Indicator functions ----------------
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def analyze_df(df):
    df = df.copy()
    df['20MA'] = df['Close'].rolling(20).mean()
    df['50MA'] = df['Close'].rolling(50).mean()
    df['RSI'] = rsi(df['Close'], period=14)
    df['MACD'], df['MACD_SIG'], df['MACD_H'] = macd(df['Close'])
    return df

# Load company names from CSV
company_df = pd.read_csv("watchlist.csv")  # your CSV file
ticker_to_name = dict(zip(company_df['Yahoo Ticker'], company_df['Company Name']))

st.set_page_config(page_title="Swing Trading Enhanced Dashboard", layout="wide")
st.title("📈 Swing Trading Enhanced Dashboard")
import os

# Load watchlist from CSV if session_state is empty
import os

# Load watchlist from CSV
if 'watchlist' not in st.session_state:
    csv_path = "watchlist.csv"
    if os.path.exists(csv_path):
        df_watchlist = pd.read_csv(csv_path)
        st.session_state.watchlist = df_watchlist['Yahoo Ticker'].tolist()
    else:
        st.session_state.watchlist = []
    if os.path.exists(csv_path):
        df_watchlist = pd.read_csv(csv_path)
        st.session_state.watchlist = df_watchlist['Yahoo Ticker'].tolist()
    else:
        st.session_state.watchlist = []


# Auto-refresh every 3 minute
st_autorefresh(interval=180000)

# --- Watchlist management ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()

# Add new ticker
new_ticker = st.text_input("Enter ticker to add:", "").upper()
if st.button("Add Ticker"):
    if new_ticker:
        if new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker)
            save_watchlist(st.session_state.watchlist)
            st.success(f"{new_ticker} added to watchlist.")
        else:
            st.warning(f"{new_ticker} is already in watchlist.")



# --- Indicator functions ---
def add_basic_indicators(df):
    if df.empty:
        return df

    df = df.copy()  # Work on a copy

    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df


def add_extra_indicators(df):
    """Add ATR, ADX, Stochastic, Volume, Support, Pullback indicators"""
    if df.empty or len(df) < 50:
        return df

    df = df.copy()

    # ===== ATR =====
    df['H-L'] = df['High'] - df['Low']
    df['H-Cp'] = abs(df['High'] - df['Close'].shift())
    df['L-Cp'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['H-L', 'H-Cp', 'L-Cp']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    # ===== ADX =====
    df['UpMove'] = df['High'] - df['High'].shift()
    df['DownMove'] = df['Low'].shift() - df['Low']
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
    df['TR14'] = df['TR'].rolling(14).sum()

    tr14_safe = df['TR14'].replace(0, np.nan)
    df['+DI14'] = 100 * (df['+DM'].rolling(14).sum() / tr14_safe)
    df['-DI14'] = 100 * (df['-DM'].rolling(14).sum() / tr14_safe)

    di_sum = df['+DI14'] + df['-DI14']
    di_sum_safe = di_sum.replace(0, np.nan)
    df['DX'] = 100 * abs(df['+DI14'] - df['-DI14']) / di_sum_safe
    df['ADX'] = df['DX'].rolling(14).mean()

    # ===== STOCHASTIC =====
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    denominator = high14 - low14
    denominator = denominator.replace(0, np.nan)
    stoch_k = 100 * (df['Close'] - low14) / denominator
    stoch_d = stoch_k.rolling(3).mean()
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_d

    # ===== VOLUME =====
    df['Volume_MA20'] = df['Volume'].rolling(20).mean()
    vol_ma_safe = df['Volume_MA20'].replace(0, np.nan)
    df['Volume_Ratio'] = df['Volume'] / vol_ma_safe

    # ===== SUPPORT/SWING LOW =====
    df['Swing_Low_20'] = df['Low'].rolling(20).min()
    swing_safe = df['Swing_Low_20'].replace(0, np.nan)
    df['Distance_to_Support'] = ((df['Close'] - df['Swing_Low_20']) / swing_safe) * 100

    # ===== PULLBACK TO EMA =====
    ema_safe = df['EMA50'].replace(0, np.nan)
    df['Distance_to_EMA50'] = ((df['Close'] - df['EMA50']) / ema_safe) * 100

    # ===== NEW: MACD HISTOGRAM (for momentum strength) =====
    df['MACD_H'] = df['MACD'] - df['MACD_Signal']

    # Fill NaN values
    # New corrected code
    df = df.bfill().ffill().fillna(0)

    return df

def detect_divergence(df):
    """Detect bullish RSI divergence - VERY RELIABLE signal"""
    if len(df) < 30:
        return False, "Insufficient data"

    # Find recent lows in price and RSI
    recent_data = df.tail(20)

    # Find price lows
    price_low_idx = recent_data['Low'].idxmin()
    price_low_val = recent_data.loc[price_low_idx, 'Low']

    # Find RSI lows
    rsi_low_idx = recent_data['RSI'].idxmin()
    rsi_low_val = recent_data.loc[rsi_low_idx, 'RSI']

    # BULLISH DIVERGENCE = Price makes lower low, but RSI makes higher low
    # This means momentum is improving even though price is falling

    if price_low_idx > rsi_low_idx:
        # Price low is more recent than RSI low
        prev_price_low = recent_data['Low'].nsmallest(2).iloc[1]
        prev_rsi_low = recent_data['RSI'].nsmallest(2).iloc[1]

        if price_low_val < prev_price_low and rsi_low_val > prev_rsi_low:
            return True, "Bullish RSI Divergence Detected"

    return False, "No divergence"


def check_support_strength(df):
    """Check how many times support level has been tested"""
    if len(df) < 30:
        return 1

    swing_low = df['Low'].tail(20).min()
    current_price = df['Close'].iloc[-1]

    # Count how many times price touched support (within 1%)
    touches = 0
    for i in range(len(df) - 20, len(df)):
        if abs(df.iloc[i]['Low'] - swing_low) / swing_low < 0.01:
            touches += 1

    # Strength: 1 = weak, 2 = ok, 3+ = strong support
    return min(touches, 3)


def check_macd_strength(df):
    """Check if MACD histogram is growing (momentum accelerating)"""
    if len(df) < 5:
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    # MACD histogram growing = momentum accelerating
    hist_now = last['MACD_H'] if 'MACD_H' in df.columns else (last['MACD'] - last['MACD_Signal'])
    hist_prev = prev['MACD_H'] if 'MACD_H' in df.columns else (prev['MACD'] - prev['MACD_Signal'])
    hist_prev2 = prev2['MACD_H'] if 'MACD_H' in df.columns else (prev2['MACD'] - prev2['MACD_Signal'])

    # Histogram should be positive AND growing
    is_growing = hist_now > hist_prev > hist_prev2

    return is_growing and hist_now > 0


def check_market_context():
    """Check if overall market (SPY) is in uptrend"""
    try:
        spy_data = yf.download('SPY', period='3mo', progress=False)
        spy_data['EMA50'] = spy_data['Close'].ewm(span=50, adjust=False).mean()

        # Market is bullish if SPY close > EMA50
        is_bullish = spy_data['Close'].iloc[-1] > spy_data['EMA50'].iloc[-1]

        return is_bullish, f"SPY: {'Bullish' if is_bullish else 'Bearish'}"
    except:
        return True, "Market check unavailable"  # Default to True if can't fetch


def enhanced_signal(df):
    """Generate ENHANCED trading signals with divergence + support strength"""
    if df.empty or len(df) < 50:
        return "HOLD", "Insufficient data"

    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # ===== CORE CONDITIONS (Original 8) =====
        core_conditions = {
            "Trend Up (EMA50>EMA200)": float(last['EMA50']) > float(last['EMA200']),
            "MACD Cross": (float(last['MACD']) > float(last['MACD_Signal'])) and \
                          (float(prev['MACD']) <= float(prev['MACD_Signal'])),
            "RSI<70": float(last['RSI']) < 70,
            "Stoch<80": float(last['Stoch_K']) < 80,
            "ADX>25": float(last['ADX']) > 25,  # STRICTER: was >20, now >25
            "Volume Up 1.5x": float(last.get('Volume_Ratio', 0)) > 1.5,  # STRICTER: was >1.2, now >1.5
            "Near Support": float(last.get('Distance_to_Support', 100)) < 2.0,  # STRICTER: was <3, now <2
            "Pullback Entry": (float(last.get('Distance_to_EMA50', 0)) >= -2.0) and \
                              (float(last.get('Distance_to_EMA50', 0)) <= 2.0)
        }

        # ===== NEW: BONUS CONDITIONS (Add these) =====
        bonus_conditions = {
            "Bullish Divergence": detect_divergence(df)[0],
            "Strong Support": check_support_strength(df) >= 2,
            "MACD Histogram Growing": check_macd_strength(df),
            "Market Context Bullish": check_market_context()[0]
        }

        core_count = sum(core_conditions.values())
        bonus_count = sum(bonus_conditions.values())

        all_reasons = [k for k, v in core_conditions.items() if v]
        bonus_reasons = [k for k, v in bonus_conditions.items() if v]

        # ===== IMPROVED SCORING =====
        if core_count >= 8 and bonus_count >= 2:
            return "STRONG BUY", "Core: " + ", ".join(all_reasons[:3]) + " | Bonus: " + ", ".join(bonus_reasons)
        elif core_count >= 7 and bonus_count >= 1:
            return "POTENTIAL BUY", "Core: " + ", ".join(all_reasons[:3]) + " | Bonus: " + ", ".join(bonus_reasons)
        elif core_count >= 6:
            return "MODERATE BUY", "Core: " + ", ".join(all_reasons[:3])
        else:
            return "HOLD", f"Core: {core_count}/8, Bonus: {bonus_count}/4"

    except Exception as e:
        return "ERROR", str(e)

def fetch_safe(ticker):
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)

        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        # Ensure we have the columns we need
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in required_cols if col in df.columns]

        if not available_cols:
            return pd.DataFrame()

        df = df[available_cols].copy()

        # Reset index to ensure clean data
        df = df.reset_index()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')

        # Ensure all columns are numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()

        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

# --- Analysis ---
if st.session_state.watchlist:
    results = []
    for ticker in st.session_state.watchlist:
        df = fetch_safe(ticker)
        if df.empty:
            results.append({
                'Ticker': ticker,
                'Price':'-', 'Stop':'-', 'Target':'-', 'RSI':'-',
                'EMA50':'-', 'EMA200':'-', 'MACD':'-', 'MACD_Signal':'-',
                'ADX':'-', 'ATR':'-', 'Stoch_K':'-', 'Stoch_D':'-',
                'Enhanced Signal':'NO DATA', 'Reason':'No historical data'
            })
            continue
        df = add_basic_indicators(df)
        df = add_extra_indicators(df)
        signal, reason = enhanced_signal(df)
        last = df.iloc[-1]


        # Helper to safely get scalar values
        def safe(val):
            if hasattr(val, 'item'):
                return val.item()
            return val


        results.append({
            'Ticker': ticker,
            'Company Name': ticker_to_name.get(ticker, "-"),
            'Enhanced Signal': signal,
            'Price': round(safe(last['Close']), 2) if last['Close'] is not None else "-",
            'Stop': round(safe(last['EMA50']) * 0.97, 2) if last['EMA50'] is not None else "-",
            'Target': round(safe(last['Close']) * 1.04, 2) if last['Close'] is not None else "-",
            'RSI': round(safe(last['RSI']), 1) if last['RSI'] is not None else "-",
            'EMA50': round(safe(last['EMA50']), 2) if last['EMA50'] is not None else "-",
            'EMA200': round(safe(last['EMA200']), 2) if last['EMA200'] is not None else "-",
            'MACD': round(safe(last['MACD']), 2) if last['MACD'] is not None else "-",
            'MACD_Signal': round(safe(last['MACD_Signal']), 2) if last['MACD_Signal'] is not None else "-",
            'ADX': round(safe(last['ADX']), 2) if last['ADX'] is not None else "-",
            'ATR': round(safe(last['ATR']), 2) if last['ATR'] is not None else "-",
            'Stoch_K': round(safe(last['Stoch_K']), 2) if last['Stoch_K'] is not None else "-",
            'Stoch_D': round(safe(last['Stoch_D']), 2) if last['Stoch_D'] is not None else "-",
            'Reason': reason
        })

    df_results = pd.DataFrame(results)


    def highlight_signal(val):
        val = str(val).upper()
        if "STRONG BUY" in val:
            return "background-color: #c6ffbf; font-weight:700; color:green;"
        elif "MODERATE BUY" in val:
            return "background-color: #d4f7c4; color:green;"
        elif "POTENTIAL BUY" in val:
            return "background-color: #e6ffe6; color:green;"
        elif "BUY" in val:
            return "background-color: #e7ffe6;"
        elif "STRONG SELL" in val:
            return "background-color: #ffbfbf; font-weight:700; color:red;"
        elif "SELL" in val:
            return "background-color: #ffe7e7;"
        elif "HOLD" in val:
            return "background-color: #fff3e0;"
        elif "ERROR" in val or "NO DATA" in val:
            return "background-color: #f8d7da; color:red;"
        return ""


    # --- Show STRONG BUY tickers first ---
    # STRONG BUY
    strong_buys = df_results[df_results["Enhanced Signal"] == "STRONG BUY"]
    with st.expander(f"🚀 STRONG BUY SIGNALS ({len(strong_buys)})", expanded=True):
        if not strong_buys.empty:
            st.balloons()
            st.dataframe(
                strong_buys.style.applymap(highlight_signal, subset=["Enhanced Signal"]),
                use_container_width=True
            )

        else:
            st.info("No strong buy signals currently.")

    # POTENTIAL BUY
    potential_buys = df_results[df_results["Enhanced Signal"] == "POTENTIAL BUY"]
    with st.expander(f"💡 POTENTIAL BUY SIGNALS ({len(potential_buys)})"):
        if not potential_buys.empty:
            st.dataframe(
                potential_buys.style.applymap(highlight_signal, subset=["Enhanced Signal"]),
                use_container_width=True
            )
        else:
            st.info("No potential buy signals currently.")

    # MODERATE BUY
    moderate_buys = df_results[df_results["Enhanced Signal"] == "MODERATE BUY"]
    with st.expander(f"⚠️ MODERATE BUY SIGNALS ({len(moderate_buys)})"):
        if not moderate_buys.empty:
            st.dataframe(
                moderate_buys.style.applymap(highlight_signal, subset=["Enhanced Signal"]),
                use_container_width=True
            )
        else:
            st.info("No moderate buy signals currently.")

    # Optional: full watchlist in an expander
    with st.expander("Full Watchlist (All signals)"):
        st.dataframe(df_results.style.applymap(highlight_signal, subset=["Enhanced Signal"]), use_container_width=True)

else:
    st.info("Watchlist is empty. Add tickers above to begin.")







# swing_ui_enhanced_levels.py
import streamlit as st
import pandas as pd
import yfinance as yf
import time
from typing import Optional, Dict

# ---------------------------------------------------------
# Streamlit Title
# ---------------------------------------------------------

st.title("📉 52-Week High Drop Analyzer — Watchlist")

# ================== User config ==================
WATCHLIST_CSV = "watchlist.csv"
HISTORY_PERIOD = "2y"
DOWNLOAD_SLEEP = 0.15

CATEGORIES = {
    "<10%": (0.0, 10.0),
    "10-20%": (10.0, 20.0),
    "20-30%": (20.0, 30.0),
    "30-40%": (30.0, 40.0),
    "40-50%": (40.0, 50.0001),
}

CATEGORY_COLORS = {
    "<10%": "#d9f0ff",
    "10-20%": "#b3e6ff",
    "20-30%": "#ffeeb3",
    "30-40%": "#ffd6cc",
    "40-50%": "#ffb3b3",
}
# =================================================


# ---------------------------------------------------------
# Load Watchlist
# ---------------------------------------------------------
try:
    watch_df = pd.read_csv(WATCHLIST_CSV)
except Exception as e:
    st.error(f"Could not read watchlist CSV: {e}")
    st.stop()

# CSV MUST contain: Rank, Country, Company Name, Yahoo Ticker
if "Yahoo Ticker" not in watch_df.columns or "Company Name" not in watch_df.columns:
    st.error("CSV file must contain 'Yahoo Ticker' and 'Company Name'.")
    st.stop()

ticker_col = 'Yahoo Ticker'
company_col = 'Company Name'

# Extract tickers
tickers = [t.strip().upper() for t in watch_df['Yahoo Ticker'].dropna() if t.strip()]
companies = {row['Yahoo Ticker'].strip().upper(): row['Company Name'] for _, row in watch_df.iterrows()}



if len(tickers) == 0:
    st.warning("No tickers found in watchlist.")
    st.stop()

st.sidebar.markdown(f"Loaded **{len(tickers)}** tickers from `{WATCHLIST_CSV}`.")
st.sidebar.write("Example:", tickers[:5])


# ---------------------------------------------------------
# yfinance downloader (cached)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def download_single_ticker(ticker: str, period: str = HISTORY_PERIOD) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)  # remove threads=False
        if df is None or df.empty:
            return None

        df.index = pd.to_datetime(df.index)

        # Fix MultiIndex columns
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        return df
    except Exception:
        return None


# ---------------------------------------------------------
# Compute 52-week drop
# ---------------------------------------------------------
def compute_latest_drop_from_df(df: pd.DataFrame) -> Optional[Dict]:
    if df is None or df.empty:
        return None

    if "Close" not in df.columns:
        return None

    high_series = df["High"] if "High" in df.columns else df["Close"]
    h52 = high_series.rolling(window=252, min_periods=1).max()

    latest_close = float(df["Close"].iloc[-1])
    latest_h52 = float(h52.iloc[-1])

    if latest_h52 == 0:
        return None

    drop_pct = (latest_h52 - latest_close) / latest_h52 * 100

    return {
        "Close": latest_close,
        "52W_High": latest_h52,
        "Drop_pct": drop_pct
    }


# ---------------------------------------------------------
# Main Loop
# ---------------------------------------------------------
st.info("Fetching price data for watchlist — please wait (cached).")
progress = st.progress(0)

collected = []
for i, ticker in enumerate(tickers):
    df_hist = download_single_ticker(ticker)
    rec = compute_latest_drop_from_df(df_hist)

    if rec:
        collected.append({
            "Company Name": companies.get(ticker, ticker),
            "Ticker": ticker,
            "Current Price": round(rec["Close"], 2),
            "52-Week High": round(rec["52W_High"], 2),
            "Drop %": round(rec["Drop_pct"], 2),
        })

    progress.progress((i + 1) / len(tickers))
    time.sleep(DOWNLOAD_SLEEP)

progress.empty()

if len(collected) == 0:
    st.error("No price data could be loaded for any ticker.")
    st.stop()

df_all = pd.DataFrame(collected)
df_all["Drop %"] = pd.to_numeric(df_all["Drop %"], errors="coerce")

# ---------------------------------------------------------
# Bucketize
# ---------------------------------------------------------
buckets = {
    label: df_all[(df_all["Drop %"] >= rng[0]) & (df_all["Drop %"] < rng[1])].copy()
    for label, rng in CATEGORIES.items()
}

for label in buckets:
    buckets[label].sort_values("Drop %", ascending=False, inplace=True)


# ---------------------------------------------------------
# Display
# ---------------------------------------------------------
st.markdown("## 📊 Stocks grouped by % drop from 52-week high")

for label, df_bucket in buckets.items():
    count = len(df_bucket)
    color = CATEGORY_COLORS[label]

    with st.expander(f"{label} — {count} stocks", expanded=False):
        st.markdown(
            f"<div style='background:{color};padding:8px;border-radius:6px;'>"
            f"<b>{label}</b> — {count} stock(s)</div>",
            unsafe_allow_html=True
        )

        if count == 0:
            st.info("No stocks in this bucket.")
        else:
            d = df_bucket.copy()
            d["Current Price"] = d["Current Price"].map(lambda x: f"{x:.2f}")
            d["52-Week High"] = d["52-Week High"].map(lambda x: f"{x:.2f}")
            d["Drop %"] = d["Drop %"].map(lambda x: f"{x:.2f}%")
            st.dataframe(d, use_container_width=True)

st.caption("Live data cached. Clear cache or rerun to refresh.")

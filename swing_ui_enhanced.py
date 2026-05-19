import streamlit as st
import pandas as pd
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
import numpy as np
import os
import time


# ================== HELPER FUNCTIONS ==================

def load_watchlist():
    """Load tickers from CSV"""
    csv_path = "watchlist.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df['Yahoo Ticker'].tolist()
    return []


def save_watchlist(watchlist):
    """Save tickers to CSV"""
    df = pd.DataFrame({'Yahoo Ticker': watchlist})
    df.to_csv("watchlist.csv", index=False)


def load_ticker_to_name():
    """Load ticker to company name mapping"""
    csv_path = "watchlist.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return dict(zip(df['Yahoo Ticker'], df['Company Name']))
    return {}


# ================== PAGE CONFIG ==================

st.set_page_config(page_title="Swing Trading Enhanced Dashboard", layout="wide")
st.title("📈 Swing Trading Entry Screener (11 Indicators)")


# ================== INDICATOR FUNCTIONS ==================

def add_basic_indicators(df):
    """Add EMA50, EMA200, MACD, RSI"""
    if df.empty or len(df) < 50:
        return df

    df = df.copy()

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
    """Add ATR, ADX, Volume, Support, Pullback, MACD Histogram"""
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

    # ===== MACD HISTOGRAM =====
    df['MACD_H'] = df['MACD'] - df['MACD_Signal']

    # Fill NaN values
    df = df.bfill().ffill().fillna(0)

    return df


def detect_divergence(df):
    """Detect bullish RSI divergence"""
    if len(df) < 30:
        return False

    try:
        recent_data = df.tail(20)

        if len(recent_data) < 5:
            return False

        price_lows = recent_data['Low'].nsmallest(2)
        rsi_lows = recent_data['RSI'].nsmallest(2)

        if len(price_lows) < 2 or len(rsi_lows) < 2:
            return False

        if price_lows.iloc[0] < price_lows.iloc[1] and rsi_lows.iloc[0] > rsi_lows.iloc[1]:
            return True

        return False
    except:
        return False


def check_support_strength(df):
    """Check how many times support level has been tested"""
    if len(df) < 30:
        return 1

    try:
        swing_low = df['Low'].tail(20).min()

        touches = 0
        for i in range(len(df) - 20, len(df)):
            if abs(df.iloc[i]['Low'] - swing_low) / swing_low < 0.01:
                touches += 1

        return min(touches, 3)
    except:
        return 1


def check_macd_strength(df):
    """Check if MACD histogram is growing"""
    if len(df) < 5:
        return False

    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]

        hist_now = last.get('MACD_H', last['MACD'] - last['MACD_Signal'])
        hist_prev = prev.get('MACD_H', prev['MACD'] - prev['MACD_Signal'])
        hist_prev2 = prev2.get('MACD_H', prev2['MACD'] - prev2['MACD_Signal'])

        is_growing = hist_now > hist_prev > hist_prev2

        return is_growing and hist_now > 0
    except:
        return False


@st.cache_data(ttl=600)
def check_market_context():
    """Check if SPY is in uptrend"""
    try:
        spy_data = yf.download('SPY', period='3mo', progress=False)

        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = [col[0] if isinstance(col, tuple) else col for col in spy_data.columns]

        spy_data['EMA50'] = spy_data['Close'].ewm(span=50, adjust=False).mean()

        is_bullish = spy_data['Close'].iloc[-1] > spy_data['EMA50'].iloc[-1]

        return is_bullish, f"SPY: {'Bullish ✓' if is_bullish else 'Bearish ✗'}"
    except:
        return True, "Market check unavailable"


def enhanced_signal(df):
    """Generate enhanced trading signals - 11 indicators with tiered logic"""
    if df.empty or len(df) < 50:
        return "HOLD", "Insufficient data"

    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # ===== 5 URGENT CONDITIONS =====
        urgent = {
            "Trend Up (EMA50>EMA200)": float(last['EMA50']) > float(last['EMA200']),
            "MACD Bullish": float(last['MACD']) > float(last['MACD_Signal']),
            "ADX>25": float(last['ADX']) > 25,
            "Volume >1.5x": float(last.get('Volume_Ratio', 0)) > 1.5,
            "Support <2%": float(last.get('Distance_to_Support', 100)) < 2.0
        }

        # ===== 2 SECONDARY CONDITIONS =====
        secondary = {
            "RSI<70": float(last['RSI']) < 70,
            "Pullback Entry": (float(last.get('Distance_to_EMA50', 0)) >= -2.0) and \
                              (float(last.get('Distance_to_EMA50', 0)) <= 2.0)
        }

        # ===== 4 BONUS CONDITIONS =====
        divergence_present = detect_divergence(df)
        support_strength = check_support_strength(df)
        macd_strong = check_macd_strength(df)
        market_bullish, market_msg = check_market_context()

        bonus = {
            "Bullish Divergence": divergence_present,
            "Strong Support (2+ touches)": support_strength >= 2,
            "MACD Histogram Growing": macd_strong,
            "Market Context Bullish": market_bullish
        }

        urgent_count = sum(urgent.values())
        secondary_count = sum(secondary.values())
        bonus_count = sum(bonus.values())

        # ===== TIERED SIGNAL LOGIC =====
        if urgent_count >= 5 and secondary_count >= 2 and bonus_count >= 2:
            reason = f"✓ All 5 Urgent + 2 Secondary + 2 Bonus | {market_msg}"
            return "STRONG BUY", reason

        elif urgent_count >= 5 and secondary_count >= 2 and bonus_count >= 1:
            reason = f"✓ All 5 Urgent + 2 Secondary + 1 Bonus | {market_msg}"
            return "POTENTIAL BUY", reason

        elif urgent_count >= 5 and secondary_count >= 1:
            reason = f"✓ All 5 Urgent + 1 Secondary | {market_msg}"
            return "MODERATE BUY", reason

        else:
            reason = f"Urgent: {urgent_count}/5, Secondary: {secondary_count}/2, Bonus: {bonus_count}/4"
            return "HOLD", reason

    except Exception as e:
        return "ERROR", str(e)


@st.cache_data(ttl=300)
def fetch_safe(ticker):
    """Safely download stock data"""
    try:
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in required_cols if col in df.columns]

        if not available_cols:
            return pd.DataFrame()

        df = df[available_cols].copy()

        df = df.reset_index()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()

        return df
    except Exception as e:
        return pd.DataFrame()


def score_signal(row):
    """Score signals for ranking"""
    score = 0

    if row['Enhanced Signal'] == 'STRONG BUY':
        score += 20
    elif row['Enhanced Signal'] == 'POTENTIAL BUY':
        score += 15
    elif row['Enhanced Signal'] == 'MODERATE BUY':
        score += 10

    try:
        if isinstance(row['Volume_Ratio'], (int, float)) and row['Volume_Ratio'] > 2.0:
            score += 4
        elif isinstance(row['Volume_Ratio'], (int, float)) and row['Volume_Ratio'] > 1.5:
            score += 2
    except:
        pass

    try:
        if isinstance(row['Distance_to_Support'], (int, float)):
            if row['Distance_to_Support'] < 1.0:
                score += 3
            elif row['Distance_to_Support'] < 2.0:
                score += 2
    except:
        pass

    try:
        if isinstance(row['ADX'], (int, float)):
            if row['ADX'] > 35:
                score += 3
            elif row['ADX'] > 25:
                score += 1
    except:
        pass

    return score


def highlight_signal(val):
    """Color code signals"""
    val = str(val).upper()
    if "STRONG BUY" in val:
        return "background-color: #c6ffbf; font-weight:700; color:green;"
    elif "MODERATE BUY" in val:
        return "background-color: #d4f7c4; color:green;"
    elif "POTENTIAL BUY" in val:
        return "background-color: #e6ffe6; color:green;"
    elif "HOLD" in val:
        return "background-color: #fff3e0;"
    elif "NO DATA" in val or "ERROR" in val:
        return "background-color: #f8d7da; color:red;"
    return ""


# ================== STREAMLIT UI - SWING SCREENER ==================

# Initialize session state
# ================== STREAMLIT UI - SWING SCREENER ==================

# Initialize session state
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()

ticker_to_name = load_ticker_to_name()

# ===== MARKET STATUS (ADD THIS) =====
st.markdown("---")
spy_bullish, spy_msg = check_market_context()

if spy_bullish:
    st.success(f"📈 {spy_msg}")
else:
    st.error(f"📉 {spy_msg}")

st.markdown("---")
# ===== END MARKET STATUS =====


ticker_to_name = load_ticker_to_name()

# Auto-refresh every 3 minutes
st_autorefresh(interval=180000)

# Add new ticker
col1, col2 = st.columns([4, 1])
with col1:
    new_ticker = st.text_input("Enter ticker to add:", "").upper()
with col2:
    if st.button("Add Ticker"):
        if new_ticker:
            if new_ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_ticker)
                save_watchlist(st.session_state.watchlist)
                st.success(f"{new_ticker} added!")
            else:
                st.warning(f"{new_ticker} already in watchlist")

# Analysis
if st.session_state.watchlist:
    st.info(f"🔍 Analyzing {len(st.session_state.watchlist)} tickers with 11 indicators...")

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, ticker in enumerate(st.session_state.watchlist):
        status_text.write(f"⏳ Downloading {ticker}...")

        df = fetch_safe(ticker)

        if df.empty:
            results.append({
                'Ticker': ticker,
                'Company Name': ticker_to_name.get(ticker, "-"),
                'Price': '-',
                'Enhanced Signal': 'NO DATA',
                'RSI': '-',
                'Volume_Ratio': '-',
                'Distance_to_Support': '-',
                'ADX': '-',
                'Score': 0
            })
            progress_bar.progress((idx + 1) / len(st.session_state.watchlist))
            time.sleep(0.2)
            continue

        df = add_basic_indicators(df)
        df = add_extra_indicators(df)
        signal, reason = enhanced_signal(df)
        last = df.iloc[-1]


        def safe(val):
            return val.item() if hasattr(val, 'item') else val


        results.append({
            'Ticker': ticker,
            'Company Name': ticker_to_name.get(ticker, "-"),
            'Enhanced Signal': signal,
            'Price': round(safe(last['Close']), 2) if pd.notna(last['Close']) else "-",
            'RSI': round(safe(last['RSI']), 1) if pd.notna(last['RSI']) else "-",
            'Volume_Ratio': round(safe(last['Volume_Ratio']), 2) if pd.notna(last.get('Volume_Ratio')) else "-",
            'Distance_to_Support': round(safe(last['Distance_to_Support']), 2) if pd.notna(
                last.get('Distance_to_Support')) else "-",
            'ADX': round(safe(last['ADX']), 2) if pd.notna(last['ADX']) else "-",
            'Score': 0,
            'Reason': reason
        })

        progress_bar.progress((idx + 1) / len(st.session_state.watchlist))
        time.sleep(0.2)

    progress_bar.empty()
    status_text.empty()

    df_results = pd.DataFrame(results)
    df_results['Score'] = df_results.apply(score_signal, axis=1)

    # STRONG BUY
    strong = df_results[df_results['Enhanced Signal'] == 'STRONG BUY'].sort_values('Score', ascending=False)

    st.subheader(f"🚀 STRONG BUY ({len(strong)}) — 5 Urgent + 2 Secondary + 2 Bonus")
    if not strong.empty:
        st.balloons()
        st.success("✅ Perfect setup - All conditions aligned!")
        display_cols = ['Ticker', 'Company Name', 'Price', 'RSI', 'Volume_Ratio', 'Distance_to_Support', 'ADX', 'Score']
        st.dataframe(
            strong[display_cols].head(10).style.map(highlight_signal, subset=['Enhanced Signal']),
            use_container_width=True
        )
        st.warning(f"📌 Take top 3-5 by score")
    else:
        st.info("No strong buy signals currently")

    # POTENTIAL BUY
    potential = df_results[df_results['Enhanced Signal'] == 'POTENTIAL BUY'].sort_values('Score', ascending=False)

    with st.expander(f"💡 POTENTIAL BUY ({len(potential)}) — 5 Urgent + 2 Secondary + 1 Bonus"):
        if not potential.empty:
            st.info("Good setup - Missing 1 bonus filter")
            display_cols = ['Ticker', 'Company Name', 'Price', 'RSI', 'Volume_Ratio', 'Distance_to_Support', 'ADX',
                            'Score']
            st.dataframe(
                potential[display_cols].head(10).style.map(highlight_signal, subset=['Enhanced Signal']),
                use_container_width=True
            )
        else:
            st.info("No potential buy signals")

    # MODERATE BUY
    moderate = df_results[df_results['Enhanced Signal'] == 'MODERATE BUY'].sort_values('Score', ascending=False)

    with st.expander(f"⚠️ MODERATE BUY ({len(moderate)}) — 5 Urgent + 1 Secondary"):
        if not moderate.empty:
            st.warning("Decent setup - Missing secondary confirmations")
            display_cols = ['Ticker', 'Company Name', 'Price', 'RSI', 'Volume_Ratio', 'Distance_to_Support', 'ADX',
                            'Score']
            st.dataframe(
                moderate[display_cols].head(10).style.map(highlight_signal, subset=['Enhanced Signal']),
                use_container_width=True
            )
        else:
            st.info("No moderate buy signals")

    # Summary
    st.divider()
    st.subheader("📊 11 Indicators Breakdown")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("STRONG BUY", len(strong))
    with col2:
        st.metric("POTENTIAL BUY", len(potential))
    with col3:
        st.metric("MODERATE BUY", len(moderate))

    st.info("""
    **11 Indicators Used:**
    🔴 **5 URGENT:** EMA Trend + MACD Cross + ADX + Volume + Support
    🟡 **2 SECONDARY:** RSI + Pullback Entry
    💚 **4 BONUS:** Divergence + Support Strength + MACD Histogram + Market Context
    """)

else:
    st.info("📝 Watchlist is empty. Add tickers above to begin.")

# ================== 52-WEEK DROP ANALYZER ==================

st.divider()
st.title("📉 52-Week High Drop Analyzer")

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

try:
    watch_df = pd.read_csv("watchlist.csv")
except Exception as e:
    st.error(f"Could not read watchlist CSV: {e}")
    st.stop()

if "Yahoo Ticker" not in watch_df.columns or "Company Name" not in watch_df.columns:
    st.error("CSV file must contain 'Yahoo Ticker' and 'Company Name'.")
    st.stop()

ticker_col = 'Yahoo Ticker'
company_col = 'Company Name'

tickers_drop = [t.strip().upper() for t in watch_df['Yahoo Ticker'].dropna() if t.strip()]
companies_drop = {row['Yahoo Ticker'].strip().upper(): row['Company Name'] for _, row in watch_df.iterrows()}

if len(tickers_drop) == 0:
    st.warning("No tickers found in watchlist.")
    st.stop()

st.sidebar.markdown(f"Loaded **{len(tickers_drop)}** tickers from `watchlist.csv`.")


@st.cache_data(show_spinner=False)
def download_single_ticker(ticker: str, period: str = HISTORY_PERIOD):
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df is None or df.empty:
            return None

        df.index = pd.to_datetime(df.index)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        return df
    except Exception:
        return None


def compute_latest_drop_from_df(df):
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


st.info("Fetching price data for watchlist — please wait (cached).")
progress = st.progress(0)

collected = []
for i, ticker in enumerate(tickers_drop):
    df_hist = download_single_ticker(ticker)
    rec = compute_latest_drop_from_df(df_hist)

    if rec:
        collected.append({
            "Company Name": companies_drop.get(ticker, ticker),
            "Ticker": ticker,
            "Current Price": round(rec["Close"], 2),
            "52-Week High": round(rec["52W_High"], 2),
            "Drop %": round(rec["Drop_pct"], 2),
        })

    progress.progress((i + 1) / len(tickers_drop))
    time.sleep(DOWNLOAD_SLEEP)

progress.empty()

if len(collected) == 0:
    st.error("No price data could be loaded for any ticker.")
    st.stop()

df_all = pd.DataFrame(collected)
df_all["Drop %"] = pd.to_numeric(df_all["Drop %"], errors="coerce")

buckets = {
    label: df_all[(df_all["Drop %"] >= rng[0]) & (df_all["Drop %"] < rng[1])].copy()
    for label, rng in CATEGORIES.items()
}

for label in buckets:
    buckets[label].sort_values("Drop %", ascending=False, inplace=True)

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
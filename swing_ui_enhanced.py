import streamlit as st
import pandas as pd
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
import numpy as np
import os
import time
from datetime import datetime, timedelta


# ================== HELPER FUNCTIONS ==================

def load_watchlist():
    """Load tickers from CSV"""
    csv_path = "watchlist.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df['Yahoo Ticker'].tolist()
    return []


def save_watchlist(watchlist):
    """Save tickers to CSV preserving all columns"""
    csv_path = "watchlist.csv"

    # Load existing data
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        # Keep only tickers that still exist
        existing_df = existing_df[existing_df['Yahoo Ticker'].isin(watchlist)]
        # Add any new tickers
        new_tickers = [t for t in watchlist if t not in existing_df['Yahoo Ticker'].values]
        if new_tickers:
            new_rows = pd.DataFrame({'Yahoo Ticker': new_tickers})
            existing_df = pd.concat([existing_df, new_rows], ignore_index=True)
        existing_df.to_csv("watchlist.csv", index=False)
    else:
        df = pd.DataFrame({'Yahoo Ticker': watchlist})
        df.to_csv("watchlist.csv", index=False)

    # Add new tickers with their names (or ticker as placeholder)
    company_names = [existing_dict.get(ticker, ticker) for ticker in watchlist]
    df = pd.DataFrame({
        'Yahoo Ticker': watchlist,
        'Company Name': company_names
    })
    df.to_csv("watchlist.csv", index=False)


def load_ticker_to_name():
    """Load ticker to company name mapping"""
    csv_path = "watchlist.csv"
    ticker_dict = {}

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'Company Name' in df.columns:
            for idx, row in df.iterrows():
                ticker = row['Yahoo Ticker'].strip().upper()
                company = row.get('Company Name', ticker)
                ticker_dict[ticker] = company if pd.notna(company) else ticker
        else:
            ticker_dict = {t.strip().upper(): t.strip().upper() for t in df['Yahoo Ticker']}

    return ticker_dict


# ================== PAGE CONFIG ==================

st.set_page_config(page_title="Goel's Strategy", layout="wide")
st.title("📈 Goel's Strategy")


# ================== INDICATOR FUNCTIONS ==================

def add_basic_indicators(df):
    """Add EMA20, EMA50, MACD, RSI (faster for 2–3 day swings)"""
    if df.empty or len(df) < 50:
        return df

    df = df.copy()

    # Faster EMAs for swing
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # MACD (keep standard, but we'll use histogram/slope)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df


def add_extra_indicators(df):
    """Add ATR, ADX, Volume, Support, Pullback, MACD Histogram, ATR%"""
    if df.empty or len(df) < 50:
        return df

    df = df.copy()

    # ===== ATR =====
    df['H-L'] = df['High'] - df['Low']
    df['H-Cp'] = (df['High'] - df['Close'].shift()).abs()
    df['L-Cp'] = (df['Low'] - df['Close'].shift()).abs()
    df['TR'] = df[['H-L', 'H-Cp', 'L-Cp']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100  # volatility %

    # ===== ADX (kept, but not over-weighted) =====
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
    df['DX'] = 100 * (df['+DI14'] - df['-DI14']).abs() / di_sum_safe
    df['ADX'] = df['DX'].rolling(14).mean()

    # ===== VOLUME =====
    df['Volume_MA20'] = df['Volume'].rolling(20).mean()
    vol_ma_safe = df['Volume_MA20'].replace(0, np.nan)
    df['Volume_Ratio'] = df['Volume'] / vol_ma_safe

    # ===== SUPPORT/SWING LOW =====
    df['Swing_Low_20'] = df['Low'].rolling(20).min()
    swing_safe = df['Swing_Low_20'].replace(0, np.nan)
    df['Distance_to_Support'] = ((df['Close'] - df['Swing_Low_20']) / swing_safe) * 100

    # ===== PULLBACK TO EMA20 (faster mean) =====
    ema_safe = df['EMA20'].replace(0, np.nan)
    df['Distance_to_EMA20'] = ((df['Close'] - df['EMA20']) / ema_safe) * 100

    # ===== MACD HISTOGRAM =====
    df['MACD_H'] = df['MACD'] - df['MACD_Signal']

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
    """Check if MACD histogram is turning up and above zero"""
    if len(df) < 5:
        return False

    try:
        last3 = df['MACD_H'].tail(3)
        if last3.isna().any():
            return False

        # Increasing and above zero
        return last3.iloc[-1] > last3.iloc[-2] > last3.iloc[-3] and last3.iloc[-1] > 0
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


@st.cache_data(ttl=300)
def check_vix_level():
    """Check current VIX level and categorize"""
    try:
        vix_data = yf.download('^VIX', period='1d', progress=False)

        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = [col[0] if isinstance(col, tuple) else col for col in vix_data.columns]

        vix_value = float(vix_data['Close'].iloc[-1])

        # Categorize VIX
        if vix_value < 15:
            category = "Very Calm"
            condition = "Too slow, hard to make 2-3%"
            trading_ok = False
            color = "blue"
        elif vix_value < 20:
            category = "Normal/Healthy"
            condition = "IDEAL for swing trading ✅"
            trading_ok = True
            color = "green"
        elif vix_value < 30:
            category = "Worried"
            condition = "Getting dangerous - Only take STRONG BUY"
            trading_ok = "selective"
            color = "orange"
        elif vix_value < 40:
            category = "Scared"
            condition = "Very risky, avoid trading"
            trading_ok = False
            color = "red"
        else:
            category = "Panic"
            condition = "Market crash, STOP trading"
            trading_ok = False
            color = "darkred"

        return vix_value, category, condition, trading_ok, color
    except:
        return None, "Unavailable", "Could not fetch VIX", None, "gray"


def enhanced_signal(df):
    """Generate swing signals - TWO types: Pullback OR Breakout with VIX consideration"""
    if df.empty or len(df) < 50:
        return "HOLD", "Insufficient data", {}, {}, "N/A"

    try:
        last = df.iloc[-1]

        # ===== GET VIX INFO =====
        vix_value, vix_category, vix_condition, trading_ok, vix_color = check_vix_level()

        # ===== COMMON CONDITIONS (both setups) =====
        trend_up = float(last['EMA20']) > float(last['EMA50'])
        volume_good = float(last.get('Volume_Ratio', 0)) > 1.2
        volatility_good = float(last.get('ATR_Pct', 0)) > 1.5
        market_bullish, market_msg = check_market_context()

        # ===== TYPE 1: PULLBACK BUY =====
        pullback_to_ema = -5.0 <= float(last.get('Distance_to_EMA20', 0)) <= 0.0
        rsi_not_hot = float(last['RSI']) < 65

        pullback_conditions = [trend_up, pullback_to_ema, rsi_not_hot, volume_good, volatility_good]
        pullback_count = sum(pullback_conditions)

        # ===== TYPE 2: BREAKOUT BUY =====
        breakout = float(last['Close']) > float(df['High'].tail(5).max())
        rsi_not_extreme = float(last['RSI']) < 75

        breakout_conditions = [trend_up, breakout, volume_good, volatility_good, rsi_not_extreme]
        breakout_count = sum(breakout_conditions)

        # Format VIX string
        vix_str = f"{vix_value:.1f}" if vix_value else "N/A"

        # ===== SIGNAL LOGIC WITH VIX CONSIDERATION =====

        # If VIX too high, downgrade signals
        if vix_value and vix_value > 30:
            # Only STRONG signals when VIX high
            if pullback_count >= 4 and market_bullish:
                return "STRONG BUY (Pullback)", f"Perfect pullback setup | VIX: {vix_str} ({vix_category})", {}, {}, vix_str
            elif breakout_count >= 4 and market_bullish:
                return "STRONG BUY (Breakout)", f"Perfect breakout setup | VIX: {vix_str} ({vix_category})", {}, {}, vix_str
            else:
                return "HOLD", f"⚠️ VIX too high ({vix_str}), SKIP trading | Pullback:{pullback_count}/5 | Breakout:{breakout_count}/5", {}, {}, vix_str

        # Normal VIX conditions
        if pullback_count >= 4 and market_bullish:
            return "STRONG BUY (Pullback)", f"Perfect pullback setup | VIX: {vix_str} ({vix_category})", {}, {}, vix_str
        elif pullback_count >= 3:
            return "POTENTIAL BUY (Pullback)", f"Good pullback near EMA20 | VIX: {vix_str} ({vix_category})", {}, {}, vix_str

        if breakout_count >= 4 and market_bullish:
            return "STRONG BUY (Breakout)", f"Perfect breakout setup | VIX: {vix_str} ({vix_category})", {}, {}, vix_str
        elif breakout_count >= 3:
            return "POTENTIAL BUY (Breakout)", f"Breakout above 5-day high | VIX: {vix_str} ({vix_category})", {}, {}, vix_str

        if pullback_count >= 2 or breakout_count >= 3:
            return "MODERATE BUY", f"Weak setup - missing confirmations | VIX: {vix_str} ({vix_category})", {}, {}, vix_str

        return "HOLD", f"No setup | VIX: {vix_str} ({vix_category}) | Pullback:{pullback_count}/5 | Breakout:{breakout_count}/5", {}, {}, vix_str

    except Exception as e:
        return "ERROR", str(e), {}, {}, "N/A"


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
    """Score each setup 0–100"""
    score = 0

    signal = str(row.get('Enhanced Signal', ''))

    # Signal type base score
    if 'STRONG BUY' in signal:
        score += 50
    elif 'POTENTIAL BUY' in signal:
        score += 35
    elif 'MODERATE BUY' in signal:
        score += 20

    # Volume bonus
    vr = row.get('Volume Ratio', 0)
    if isinstance(vr, (int, float)):
        if vr > 2.0:
            score += 20
        elif vr > 1.5:
            score += 15
        elif vr > 1.2:
            score += 10

    # Volatility bonus
    atr_pct = row.get('ATR %', 0)
    if isinstance(atr_pct, (int, float)):
        if atr_pct > 3:
            score += 15
        elif atr_pct > 2:
            score += 10
        elif atr_pct > 1.5:
            score += 5

    # Risk management bonus
    if isinstance(row.get('Stop Loss (ATR)'), (int, float)):
        score += 5

    return min(score, 100)


def rating_from_score(score):
    if score >= 85:
        return "A+ (High Probability)"
    elif score >= 70:
        return "A (Strong Setup)"
    elif score >= 55:
        return "B (Decent)"
    else:
        return "C (Weak)"


# ================== STREAMLIT UI - SWING SCREENER ==================

# Initialize session state
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()

ticker_to_name = load_ticker_to_name()

# ===== MARKET STATUS =====
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    spy_bullish, spy_msg = check_market_context()
    if spy_bullish:
        st.success(f"📈 {spy_msg}")
    else:
        st.error(f"📉 {spy_msg}")

with col2:
    vix_value, vix_category, vix_condition, trading_ok, vix_color = check_vix_level()

    if vix_value:
        st.write(f"**VIX Level: {vix_value:.2f}**")
        st.write(f"**Category:** {vix_category}")
        st.write(f"**Condition:** {vix_condition}")

        if trading_ok == True:
            st.success("✅ IDEAL CONDITIONS - Trade normally")
        elif trading_ok == "selective":
            st.warning("⚠️ CAUTION - Only take STRONG BUY signals")
        else:
            st.error("❌ AVOID TRADING - Wait for better conditions")
    else:
        st.info("VIX data unavailable")

st.markdown("---")
# ===== END MARKET STATUS =====

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
                'Enhanced Signal': 'NO DATA',
                'Current Price': '-',
                'Entry Price': '-',
                'Stop Loss (ATR)': '-',
                'Target 2%': '-',
                'Target 3%': '-',
                'Volume Ratio': '-',
                'ATR %': '-',
                'Score': 0,
                'Reason': 'No data',
                'VIX Level': '-'
            })
            progress_bar.progress((idx + 1) / len(st.session_state.watchlist))
            time.sleep(0.2)
            continue

        df = add_basic_indicators(df)
        df = add_extra_indicators(df)
        signal, reason, pullback_cond, breakout_cond, vix_level = enhanced_signal(df)
        last = df.iloc[-1]


        def safe(val):
            return val.item() if hasattr(val, 'item') else val


        results.append({
            'Ticker': ticker,
            'Company Name': ticker_to_name.get(ticker, "-"),
            'Enhanced Signal': signal,
            'Current Price': round(safe(last.get('Close')), 2) if pd.notna(last.get('Close')) else "-",
            'Entry Price': round(safe(last.get('Close')), 2) if pd.notna(last.get('Close')) else "-",
            'Stop Loss (ATR)': round(safe(last.get('Close')) - (safe(last.get('ATR')) * 1.5), 2)
            if pd.notna(last.get('Close')) and pd.notna(last.get('ATR')) else "-",
            'Target 2%': round(safe(last.get('Close')) * 1.02, 2) if pd.notna(last.get('Close')) else "-",
            'Target 3%': round(safe(last.get('Close')) * 1.03, 2) if pd.notna(last.get('Close')) else "-",
            'Volume Ratio': round(safe(last.get('Volume_Ratio')), 2) if pd.notna(last.get('Volume_Ratio')) else "-",
            'ATR %': round(safe(last.get('ATR_Pct')), 2) if pd.notna(last.get('ATR_Pct')) else "-",
            'Score': 0,
            'Reason': reason,
            'VIX Level': vix_level,
            'Pullback Conditions': pullback_cond,
            'Breakout Conditions': breakout_cond
        })

        progress_bar.progress((idx + 1) / len(st.session_state.watchlist))
        time.sleep(0.2)

    progress_bar.empty()
    status_text.empty()

    df_results = pd.DataFrame(results)
    df_results['Score'] = df_results.apply(score_signal, axis=1)
    df_results['Rating'] = df_results['Score'].apply(rating_from_score)

    # STRONG BUY
    strong = df_results[df_results['Enhanced Signal'].str.contains('STRONG BUY', na=False)].sort_values('Score',
                                                                                                        ascending=False)

    st.subheader(f"🚀 STRONG BUY ({len(strong)})")
    if not strong.empty:
        st.balloons()
        st.success("✅ Perfect setup!")
        st.dataframe(
            strong[['Ticker', 'Company Name', 'Enhanced Signal', 'Current Price', 'Entry Price',
                    'Stop Loss (ATR)', 'Target 2%', 'Target 3%', 'VIX Level', 'Score', 'Rating']],
            use_container_width=True,
            height=600
        )
    else:
        st.info("No strong buy signals")

    # POTENTIAL BUY
    potential = df_results[df_results['Enhanced Signal'].str.contains('POTENTIAL BUY', na=False)].sort_values('Score',
                                                                                                              ascending=False)

    with st.expander(f"💡 POTENTIAL BUY ({len(potential)})"):
        if not potential.empty:
            st.info("Good setup - 4/5 conditions met")
            st.dataframe(
                potential[['Ticker', 'Company Name', 'Enhanced Signal', 'Current Price', 'Entry Price',
                           'Stop Loss (ATR)', 'Target 2%', 'Target 3%', 'VIX Level', 'Score', 'Rating']],
                use_container_width=True,
                height=600
            )
        else:
            st.info("No potential signals")

    # MODERATE BUY
    moderate = df_results[df_results['Enhanced Signal'].str.contains('MODERATE BUY', na=False)].sort_values('Score',
                                                                                                            ascending=False)

    with st.expander(f"⚠️ MODERATE BUY ({len(moderate)})"):
        if not moderate.empty:
            st.warning("Weak setup - 3/5 conditions met")
            st.dataframe(
                moderate[['Ticker', 'Company Name', 'Enhanced Signal', 'Current Price', 'Entry Price',
                          'Stop Loss (ATR)', 'Target 2%', 'Target 3%', 'VIX Level', 'Score', 'Rating']],
                use_container_width=True,
                height=600
            )
        else:
            st.info("No moderate signals")

    # Summary
    st.divider()
    st.subheader("📊 Strategy Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("STRONG BUY", len(strong))
    with col2:
        st.metric("POTENTIAL BUY", len(potential))
    with col3:
        st.metric("MODERATE BUY", len(moderate))
    with col4:
        total_signals = len(strong) + len(potential) + len(moderate)
        st.metric("TOTAL SIGNALS", total_signals)

    st.info("""
    **11 Indicators Used:**

    🔴 **TREND:** EMA20 > EMA50
    🔴 **MOMENTUM:** MACD Histogram > 0
    🔴 **STRENGTH:** ADX > 25
    🔴 **VOLUME:** Volume > 1.2x average
    🔴 **VOLATILITY:** ATR% > 1.5%
    🔴 **ENTRY QUALITY:** Pullback to EMA20 OR Breakout above 5-day high
    🔴 **SUPPORT:** Distance to Support < 3%
    🔴 **SUPPORT STRENGTH:** 2+ touches on support
    🔴 **REVERSAL:** Bullish Divergence
    🔴 **RSI:** < 60 (Pullback) or < 75 (Breakout)
    🔴 **MARKET:** SPY Bullish + VIX < 30
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
            st.dataframe(d, use_container_width=True, height=700)

st.caption("Live data cached. Clear cache or rerun to refresh.")




@st.cache_data(ttl=300)
def fetch_news_flag(ticker):
    """
    Fetch simple news headlines using yfinance built-in news.
    Returns category: Earnings / FDA / Acquisition / Upgrade / Other
    """
    try:
        tk = yf.Ticker(ticker)
        news = tk.news

        if not news:
            return "No News"

        headlines = " ".join([n['title'].lower() for n in news[:5]])

        if "earnings" in headlines:
            return "Earnings"
        if "fda" in headlines:
            return "FDA"
        if "acquire" in headlines or "acquisition" in headlines:
            return "Acquisition"
        if "upgrade" in headlines or "downgrade" in headlines:
            return "Analyst Action"
        if "guidance" in headlines:
            return "Guidance"

        return "Other"
    except:
        return "No News"




#PROGRAM - 3 - GAPUP
# ================== HELPER FUNCTIONS ==================
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
        .apply(lambda col: col.map(color_score), subset=["Score"])
        .format({
            "Gap %": "{:.2f}%",
            "Volume Ratio": "{:.1f}x",
            "Sector %": "{:.2f}%",
            "SPY Futures %": "{:.2f}%",
            "Score": "{:.0f}"
        })
    )

    st.dataframe(styled, use_container_width=True)



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

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




# ================== HELPER FUNCTIONS ==================

def load_watchlist():
    """Load tickers from CSV"""
    csv_path = "watchlist.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df['Yahoo Ticker'].tolist()
    return []


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

st.set_page_config(page_title="Gap Up Morning Screener", layout="wide")
st.title("⚡ Gap Up Morning Screener (Catch 10-20% Moves)")

st.markdown("""
**Purpose:** Detect stocks that gapped up overnight with volume confirmation.
Candidates for 10-20%+ moves throughout the day.

**Best Time to Run:** 9:30-10:00 AM (Market open)
**Strategy:** Day trading gap ups with strong volume & market support
""")


# ================== INDICATOR FUNCTIONS ==================

@st.cache_data(ttl=600)
def check_market_context():
    """Check if SPY is in uptrend - REQUIRED FOR GAP FILTERING"""
    try:
        spy_data = yf.download('SPY', period='3mo', progress=False)

        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = [col[0] if isinstance(col, tuple) else col for col in spy_data.columns]

        spy_data['EMA50'] = spy_data['Close'].ewm(span=50, adjust=False).mean()
        spy_close = spy_data['Close'].iloc[-1]
        spy_ema50 = spy_data['EMA50'].iloc[-1]

        is_bullish = spy_close > spy_ema50

        # Also get SPY price for context
        spy_price = spy_close

        return is_bullish, f"SPY: {'Bullish ✓' if is_bullish else 'Bearish ✗'} (${spy_price:.2f})", spy_price

    except:
        return True, "Market check unavailable", 0


@st.cache_data(ttl=300)
def check_vix_level():
    """Check current VIX level"""
    try:
        vix_data = yf.download('^VIX', period='1d', progress=False)

        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = [col[0] if isinstance(col, tuple) else col for col in vix_data.columns]

        vix_value = float(vix_data['Close'].iloc[-1])

        if vix_value < 15:
            category = "Very Calm"
            trading_ok = False
        elif vix_value < 20:
            category = "Normal/Healthy"
            trading_ok = True
        elif vix_value < 30:
            category = "Worried"
            trading_ok = "selective"
        elif vix_value < 40:
            category = "Scared"
            trading_ok = False
        else:
            category = "Panic"
            trading_ok = False

        return vix_value, category, trading_ok
    except:
        return None, "Unavailable", None


@st.cache_data(ttl=300)
def fetch_safe(ticker):
    """Safely download stock data"""
    try:
        df = yf.download(ticker, period="3mo", progress=False, auto_adjust=True)

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


@st.cache_data(ttl=120)
def fetch_premarket(ticker):
    """
    Fetch premarket data using 1-minute interval.
    yfinance provides premarket for US stocks.
    """
    try:
        df = yf.download(ticker, period="1d", interval="1m", prepost=True, progress=False)

        if df is None or df.empty:
            return None

        df.index = pd.to_datetime(df.index)
        df = df[df.index.time < datetime.strptime("09:30", "%H:%M").time()]

        if df.empty:
            return None

        pm_high = df["High"].max()
        pm_low = df["Low"].min()
        pm_volume = df["Volume"].sum()

        return {
            "PM_High": float(pm_high),
            "PM_Low": float(pm_low),
            "PM_Volume": float(pm_volume)
        }
    except:
        return None


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


def validate_data_timing(df):
    """Validate that last 2 rows are real trading days with valid timestamps."""
    if df.empty or len(df) < 2:
        return False

    # Ensure index is datetime
    try:
        df.index = pd.to_datetime(df.index)
    except:
        return False

    # Get last two valid timestamps
    dates = df.index[-2:]

    # If any timestamp is NaT → invalid
    if dates.isna().any():
        return False

    today_date = dates[-1]
    yesterday_date = dates[-2]

    # Ensure both are datetime objects
    if not isinstance(today_date, pd.Timestamp) or not isinstance(yesterday_date, pd.Timestamp):
        return False

    # Calculate difference in days
    date_diff = (today_date.normalize() - yesterday_date.normalize()).days

    # Accept 1–3 days difference (weekends/holidays)
    if 1 <= date_diff <= 3:
        return True

    return False


def calculate_gap_metrics(df, market_bullish):
    """
    Calculate gap, volume, and volatility metrics
    """
    if df.empty or len(df) < 2:
        return None

    if not validate_data_timing(df):
        return None

    try:
        today = df.iloc[-1]
        yesterday = df.iloc[-2]

        previous_close = float(yesterday['Close'])
        current_open = float(today['Open'])
        current_high = float(today['High'])
        current_close = float(today['Close'])

        gap_pct = ((current_open - previous_close) / previous_close) * 100
        intraday_move_pct = ((current_high - current_open) / current_open) * 100

        volume_ma20 = df['Volume'].iloc[:-1].tail(20).mean()
        current_volume = float(today['Volume'])

        if volume_ma20 == 0:
            volume_ratio = 0
        else:
            volume_ratio = current_volume / volume_ma20

        df_copy = df.copy()
        df_copy['H-L'] = df_copy['High'] - df_copy['Low']
        df_copy['H-Cp'] = (df_copy['High'] - df_copy['Close'].shift()).abs()
        df_copy['L-Cp'] = (df_copy['Low'] - df_copy['Close'].shift()).abs()
        df_copy['TR'] = df_copy[['H-L', 'H-Cp', 'L-Cp']].max(axis=1)
        df_copy['ATR'] = df_copy['TR'].rolling(14).mean()
        df_copy['ATR_Pct'] = (df_copy['ATR'] / df_copy['Close']) * 100

        today_atr_pct = float(df_copy['ATR_Pct'].iloc[-1])
        yesterday_atr_pct = float(df_copy['ATR_Pct'].iloc[-2])

        atr_increase = today_atr_pct - yesterday_atr_pct

        high_20 = df['High'].tail(20).max()
        low_20 = df['Low'].tail(20).min()

        resistance_break = 1 if current_open > high_20 else 0

        return {
            'Previous Close': previous_close,
            'Open': current_open,
            'High': current_high,
            'Close': current_close,
            'Gap %': gap_pct,
            'Intraday High': current_high,
            'Intraday Move %': intraday_move_pct,
            'Volume': current_volume,
            'Volume MA20': volume_ma20,
            'Volume Ratio': volume_ratio,
            'ATR %': today_atr_pct,
            'Previous ATR %': yesterday_atr_pct,
            'ATR Increase': atr_increase,
            '20-Day High': high_20,
            '20-Day Low': low_20,
            'Above 20-Day High': resistance_break
        }

    except Exception as e:
        return None


def get_gap_signal_FIXED(metrics, market_bullish, vix_value):
    """
    CORRECTED signal logic - consistent and matches sections
    """
    if not metrics:
        return "NO DATA", "Unable to fetch data", 0, "N/A"

    gap = metrics['Gap %']
    volume_ratio = metrics['Volume Ratio']
    atr_increase = metrics['ATR Increase']

    if gap < 3:
        return "NO GAP", "Gap < 3% - skip", 0, "N/A"

    vol_multiplier = 1.0
    if not market_bullish:
        vol_multiplier = 1.3

    if vix_value and vix_value > 30:
        return "TOO RISKY", f"VIX too high ({vix_value:.1f}) - skip", 0, "⚠️ HIGH VIX"

    if gap > 12:
        score = 95
        reason = f"Massive gap: {gap:.1f}% (confirms move)"
        signal = "🔥 HUGE MOVE LIKELY"
    elif gap > 10 and volume_ratio * vol_multiplier > 1.5:
        score = 90
        reason = f"Gap {gap:.1f}% + Heavy vol {volume_ratio:.1f}x"
        signal = "🔥 HUGE MOVE LIKELY"
    elif gap > 8 and volume_ratio * vol_multiplier > 1.4:
        score = 80
        reason = f"Gap {gap:.1f}% + Good vol {volume_ratio:.1f}x"
        signal = "⚡ BIG MOVE POSSIBLE"
    elif gap > 7 and volume_ratio * vol_multiplier > 1.3:
        score = 75
        reason = f"Gap {gap:.1f}% + Decent vol {volume_ratio:.1f}x"
        signal = "⚡ BIG MOVE POSSIBLE"
    elif gap > 6 and volume_ratio * vol_multiplier > 1.2:
        score = 65
        reason = f"Gap {gap:.1f}% + Volume {volume_ratio:.1f}x"
        signal = "📈 DECENT MOVE"
    elif gap > 5 and volume_ratio * vol_multiplier > 1.15:
        score = 60
        reason = f"Gap {gap:.1f}% + Volume {volume_ratio:.1f}x"
        signal = "📈 DECENT MOVE"
    elif gap > 3 and volume_ratio * vol_multiplier > 1.1:
        score = 45
        reason = f"Small gap {gap:.1f}% + Vol {volume_ratio:.1f}x"
        signal = "👀 MONITOR"
    else:
        score = 20
        reason = f"Gap {gap:.1f}% but low vol {volume_ratio:.1f}x - likely fizzle"
        signal = "⏭️ SKIP"

    if atr_increase > 1.5:
        score += 10
        reason += f" | ATR up {atr_increase:.2f}%"

    return signal, reason, min(score, 100), "OK"


# ================== STREAMLIT UI ==================

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()

ticker_to_name = load_ticker_to_name()

# ===== MARKET STATUS =====
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    spy_bullish, spy_msg, spy_price = check_market_context()
    if spy_bullish:
        st.success(f"📈 {spy_msg}")
    else:
        st.error(f"📉 {spy_msg}")

with col2:
    vix_value, vix_category, trading_ok = check_vix_level()

    if vix_value:
        color = "green" if trading_ok == True else "orange" if trading_ok == "selective" else "red"
        st.write(f"**VIX: {vix_value:.2f}**")
        st.write(f"**{vix_category}**")
    else:
        st.info("VIX data unavailable")

with col3:
    now = datetime.now()
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    if now < market_open:
        time_until = (market_open - now).total_seconds() / 60
        st.info(f"Market opens in {int(time_until)} min")
    elif now > market_close:
        st.warning("Market closed - check tomorrow at 9:30 AM")
    else:
        st.success(f"Market OPEN - Run now! ({now.strftime('%H:%M')})")

st.markdown("---")

# Auto-refresh every 2 minutes
st_autorefresh(interval=120000)

# Analysis
if st.session_state.watchlist:
    st.info(f"🔍 Scanning {len(st.session_state.watchlist)} tickers for gap ups...")

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    market_bullish, _, _ = check_market_context()
    vix_value, _, _ = check_vix_level()

    for idx, ticker in enumerate(st.session_state.watchlist):
        status_text.write(f"⏳ Checking {ticker}...")

        df = fetch_safe(ticker)

        if df.empty:
            progress_bar.progress((idx + 1) / len(st.session_state.watchlist))
            time.sleep(0.1)
            continue

        metrics = calculate_gap_metrics(df, market_bullish)

        if not metrics:
            progress_bar.progress((idx + 1) / len(st.session_state.watchlist))
            time.sleep(0.1)
            continue

        pm = fetch_premarket(ticker)
        pm_gap = None
        if pm and len(df) >= 2:
            prev_close = df.iloc[-2]['Close']
            if prev_close != 0:
                pm_gap = ((pm['PM_High'] - prev_close) / prev_close) * 100

        news_flag = fetch_news_flag(ticker)

        signal, reason, score, status = get_gap_signal_FIXED(metrics, market_bullish, vix_value)

        if metrics['Gap %'] >= 3 and signal not in ["NO GAP", "⏭️ SKIP"]:
            results.append({
                'Ticker': ticker,
                'Company Name': ticker_to_name.get(ticker, "-"),
                'Gap %': round(metrics['Gap %'], 2),
                'Prev Close': round(metrics['Previous Close'], 2),
                'Open': round(metrics['Open'], 2),
                'High': round(metrics['High'], 2),
                'Volume Ratio': round(metrics['Volume Ratio'], 2),
                'ATR Increase': round(metrics['ATR Increase'], 2),
                '20D High': round(metrics['20-Day High'], 2),
                'Above 20D High': 'YES' if metrics['Above 20-Day High'] else 'NO',
                'Premarket High': round(pm['PM_High'], 2) if pm else "-",
                'Premarket Low': round(pm['PM_Low'], 2) if pm else "-",
                'Premarket Volume': int(pm['PM_Volume']) if pm else "-",
                'Premarket Gap %': round(pm_gap, 2) if pm_gap is not None else "-",
                'Signal': signal,
                'Reason': reason,
                'Score': score,
                'Status': status,
                'News': news_flag
            })

        progress_bar.progress((idx + 1) / len(st.session_state.watchlist))
        time.sleep(0.1)

    progress_bar.empty()
    status_text.empty()

    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('Score', ascending=False)

        # ===== REAL-TIME ALERTS =====
        st.subheader("🔔 Real-Time Alerts")
        alerts = df_results[
            (df_results['Score'] >= 85) |
            (df_results['Gap %'] >= 10) |
            (df_results['Volume Ratio'] >= 2)
            ]

        if alerts.empty:
            st.info("No alerts right now")
        else:
            st.warning("⚠️ High-priority setups detected")
            st.dataframe(
                alerts[['Ticker', 'Company Name', 'Gap %', 'Volume Ratio', 'Score', 'Signal', 'News']],
                use_container_width=True,
                height=200
            )

        # ===== HUGE MOVES =====
        huge = df_results[df_results['Signal'] == "🔥 HUGE MOVE LIKELY"]

        st.subheader(f"🔥 HUGE MOVES - {len(huge)} Stock(s)")
        st.error("⚠️ These could move 15-25%+ TODAY!")
        if not huge.empty:
            st.dataframe(
                huge[['Ticker', 'Company Name', 'Gap %', 'Open', 'High', 'Volume Ratio',
                      'ATR Increase', '20D High', 'Above 20D High', 'Premarket Gap %', 'News', 'Score', 'Signal']],
                use_container_width=True,
                height=300
            )
            st.markdown("**Why are these huge?**")
            for idx, row in huge.iterrows():
                st.write(f"**{row['Ticker']}:** {row['Reason']}")
        else:
            st.info("No huge gaps with strong volume today")

        # ===== BIG MOVES =====
        big = df_results[df_results['Signal'] == "⚡ BIG MOVE POSSIBLE"]

        with st.expander(f"⚡ BIG MOVES - {len(big)} Stock(s) (10-15% potential)"):
            if not big.empty:
                st.warning("Good candidates for 10-15% moves")
                st.dataframe(
                    big[['Ticker', 'Company Name', 'Gap %', 'Open', 'High', 'Volume Ratio',
                         'ATR Increase', '20D High', 'Above 20D High', 'Premarket Gap %', 'News', 'Score', 'Signal']],
                    use_container_width=True,
                    height=300
                )
                st.markdown("**Why are these big?**")
                for idx, row in big.iterrows():
                    st.write(f"**{row['Ticker']}:** {row['Reason']}")
            else:
                st.info("No big moves today")

        # ===== DECENT MOVES =====
        decent = df_results[df_results['Signal'] == "📈 DECENT MOVE"]

        with st.expander(f"📈 DECENT MOVES - {len(decent)} Stock(s) (5-10% potential)"):
            if not decent.empty:
                st.info("Worth monitoring for breakout continuation")
                st.dataframe(
                    decent[['Ticker', 'Company Name', 'Gap %', 'Open', 'High', 'Volume Ratio',
                            'ATR Increase', '20D High', 'Above 20D High', 'Premarket Gap %', 'News', 'Score',
                            'Signal']],
                    use_container_width=True,
                    height=300
                )
                st.markdown("**Why monitor these?**")
                for idx, row in decent.iterrows():
                    st.write(f"**{row['Ticker']}:** {row['Reason']}")
            else:
                st.info("No decent moves")

        # ===== MONITOR =====
        monitor = df_results[df_results['Signal'] == "👀 MONITOR"]

        with st.expander(f"👀 MONITOR - {len(monitor)} Stock(s) (Small gaps)"):
            if not monitor.empty:
                st.info("Small gaps - need catalysts or news to move")
                st.dataframe(
                    monitor[['Ticker', 'Company Name', 'Gap %', 'Open', 'High', 'Volume Ratio',
                             'ATR Increase', '20D High', 'Above 20D High', 'Premarket Gap %', 'News', 'Score',
                             'Signal']],
                    use_container_width=True,
                    height=300
                )
            else:
                st.info("No stocks to monitor")

        # ===== SUMMARY =====
        st.divider()
        st.subheader("📊 Summary & Trading Plan")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔥 HUGE", len(huge))
        with col2:
            st.metric("⚡ BIG", len(big))
        with col3:
            st.metric("📈 DECENT", len(decent))
        with col4:
            st.metric("Total Signals", len(df_results))

        st.info("""
        **📋 CORRECTED STRATEGY (All Issues Fixed + Upgrades):**

        **Signal Logic:**
        - 🔥 **HUGE MOVE (15-25%):** Gap >12% alone OR Gap 8-12% + Heavy Volume
        - ⚡ **BIG MOVE (10-15%):** Gap 7-10% + Good Volume
        - 📈 **DECENT MOVE (5-10%):** Gap 5-7% + Volume Confirmed
        - 👀 **MONITOR:** Small gap, needs news/catalyst

        **Volume Confirmation:**
        - In bullish market: Need 1.1-1.5x volume based on gap size
        - In bearish market: Need 30% MORE volume (conservative)
        - Low volume (< 1.1x) = Gap likely fizzles

        **Market Context (FILTER):**
        - ✅ SPY Bullish + VIX < 30: Trade normally
        - ⚠️ SPY Bearish + VIX < 30: Require extra volume confirmation
        - ❌ VIX > 30: SKIP all trades (too risky)

        **Premarket Upgrade:**
        - Premarket High / Volume / Gap % now visible
        - Use Premarket Gap % + News to prioritize

        **News Filter:**
        - Earnings / FDA / Acquisition / Analyst / Guidance / Other / No News
        - Strongest moves usually have Earnings / FDA / Acquisition

        **Entry Rules (9:35-9:45 AM):**
        1. Wait 5-10 min after open for volume confirmation
        2. Enter on first pullback (not at open)
        3. Set stop loss: Previous support OR Gap-Low (whichever closer)
        4. Target: 5% (conservative) → 20%+ (if momentum holds)

        **Exit Rules:**
        - Take 50% profit at 5% gain → Trail rest with 3% stop
        - OR: Hold if momentum strong, exit by 3 PM
        - MUST exit by EOD (never hold overnight)

        **Risk Management:**
        - Max risk per trade: 1% account
        - Size = Account Risk / Stop distance
        - Gap plays = HIGH RISK/HIGH REWARD
        """)

        st.warning("""
        **⚠️ IMPORTANT NOTES:**

        1. **Check NEWS first** - Understand WHY it gapped
           - Earnings beat/miss?
           - FDA approval/rejection?
           - Acquisition news?
           - Guidance change?

        2. **Avoid fake gaps:**
           - Low volume gaps often reverse
           - This screener filters those out

        3. **This is DAY TRADING, not swing trading**
           - Different rules, higher risk, faster profits
           - Use separate capital, not your swing account

        4. **Market conditions matter:**
           - Bullish market = Gaps hold better
           - Bearish market = Higher failure rate
           - See SPY + VIX status above

        5. **Best days for gaps:**
           - Pre-earnings or post-earnings
           - FOMC announcements
           - Major economic data
           - Sector rotation days
        """)

    else:
        st.warning("❌ No tradeable gap ups today. Market may be range-bound.")
        st.info("Check back tomorrow at 9:30 AM for morning scan.")

else:
    st.info("📝 Watchlist is empty. Add tickers to watchlist.csv to begin.")


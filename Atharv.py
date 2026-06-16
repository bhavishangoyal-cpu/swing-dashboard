"""
dip_buy_screener.py

Local watchlist dip-buy scanner (target 5%).
- Reads watchlist.csv (column 'Yahoo Ticker' or first column)
- Uses yfinance for intraday (5m) and daily data
- Scores setups and suggests entry/stop/targets (no order execution)
- Configure parameters below
"""

import os
import io
import time
import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# -----------------------
# CONFIG
# -----------------------
WATCHLIST_CSV = "watchlist.csv"   # local file (first column or 'Yahoo Ticker')
POLL_INTERVAL_SEC = 300           # how often to refresh (seconds)
MIN_DOLLAR_VOLUME = 50_000_000    # liquidity gate ($)
MIN_ATR_PCT = 0.008               # 0.8% -> 0.008
MIN_MARKETCAP = 10_000_000_000    # optional gate (10B)
MIN_SIGNALS_SAMPLE = 30           # minimum bars for intraday indicators
TARGET_PROFIT_PCT = 0.05          # user goal: 5% profit target (changed from 55%)
MAX_GAP_DOWN_PCT = 0.08           # ignore >8% gap downs (panic)
MIN_GAP_DOWN_PCT = 0.01           # consider gap down >=1%
USE_REALTIME_LOOP = False         # set True to run continuously

OUTPUT_CSV = "dip_signals.csv"

# -----------------------
# UTILITIES / INDICATORS
# -----------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd_hist(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd - macd_signal

def true_range(df):
    high = df['High']
    low = df['Low']
    prev = df['Close'].shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr

def atr(df, period=14):
    tr = true_range(df)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    cum_vol = df['Volume'].cumsum()
    cum_vp = (tp * df['Volume']).cumsum()
    return cum_vp / cum_vol

def beta_vs_market(daily_df, market_symbol="SPY"):
    try:
        m = yf.download(market_symbol, period="120d", progress=False)['Close']
        s = daily_df['Close'].dropna()
        common_index = s.index.intersection(m.index)
        if len(common_index) < 30:
            return np.nan
        r_stock = s.loc[common_index].pct_change().dropna()
        r_mkt = m.loc[common_index].pct_change().dropna()
        cov = np.cov(r_stock, r_mkt)
        beta = cov[0,1] / (cov[1,1] + 1e-12)
        return float(beta)
    except Exception:
        return np.nan

# -----------------------
# DATA FETCHING
# -----------------------
def load_watchlist(path=WATCHLIST_CSV):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if 'Yahoo Ticker' in df.columns:
        tickers = df['Yahoo Ticker'].astype(str).str.strip().str.upper().dropna().tolist()
    else:
        tickers = df.iloc[:,0].astype(str).str.strip().str.upper().dropna().tolist()
    return tickers

def fetch_intraday_5m(ticker, days=7):
    try:
        df = yf.download(ticker, period=f"{days}d", interval="5m", progress=False, threads=False)
        if df is None or df.empty:
            return None
        return df.dropna()
    except Exception:
        return None

def fetch_daily(ticker, days=365):
    try:
        df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False, threads=False)
        if df is None or df.empty:
            return None
        return df.dropna()
    except Exception:
        return None

# -----------------------
# FILTERS & SCORING
# -----------------------
def liquidity_and_volatility_gate(daily_df):
    try:
        avg_dollar_vol = (daily_df['Close'] * daily_df['Volume']).tail(20).mean()
        atr14 = atr(daily_df, period=14).iloc[-1]
        atr_pct = atr14 / daily_df['Close'].iloc[-1] if atr14 is not None else 0
        return avg_dollar_vol, atr_pct
    except Exception:
        return 0, 0

def score_dip_setup(ticker, intraday_df, daily_df, qqq_intraday=None):
    reasons = []
    score = 0

    avg_dollar_vol, atr_pct = liquidity_and_volatility_gate(daily_df)
    if avg_dollar_vol < MIN_DOLLAR_VOLUME:
        reasons.append(f"low dollar vol ${avg_dollar_vol:,.0f}")
        return _result(ticker, 0, "AVOID", None, None, None, None, reasons, avg_dollar_vol, atr_pct)

    if atr_pct < MIN_ATR_PCT:
        reasons.append(f"low ATR% {atr_pct:.4f}")
        return _result(ticker, 0, "AVOID", None, None, None, None, reasons, avg_dollar_vol, atr_pct)

    if intraday_df is None or len(intraday_df) < MIN_SIGNALS_SAMPLE:
        reasons.append("no intraday data")
        return _result(ticker, 0, "AVOID", None, None, None, None, reasons, avg_dollar_vol, atr_pct)

    df = intraday_df.copy()
    price = df['Close'].iloc[-1]

    df['EMA20'] = ema(df['Close'], span=20)
    df['EMA50'] = ema(df['Close'], span=50)
    df['RSI14'] = rsi(df['Close'], period=14)
    df['MACD_H'] = macd_hist(df['Close'])
    df['VWAP'] = vwap(df)

    avg_5min_vol = df['Volume'].tail(60).mean() if len(df) >= 60 else df['Volume'].mean()
    recent_breakout_vol = df['Volume'].iloc[-5:].max()
    dip_volume = df['Volume'].iloc[-20:-1].mean() if len(df) > 20 else df['Volume'].mean()

    try:
        today = df.index.normalize()[-1]
        day_df = df[df.index.normalize() == today]
        first_bar = day_df.iloc[0]
        prev_close = df['Close'].shift(1).loc[first_bar.name]
        gap_pct = (first_bar['Open'] - prev_close) / prev_close if prev_close and not math.isnan(prev_close) else 0.0
    except Exception:
        gap_pct = 0.0

    near_ema20 = abs(price - df['EMA20'].iloc[-1]) / price <= 0.008
    reclaim_vwap = price > df['VWAP'].iloc[-1]
    rsi_val = df['RSI14'].iloc[-1]
    macd_up = df['MACD_H'].iloc[-1] > df['MACD_H'].iloc[-2] if len(df) >= 2 else False
    rs_vs_qqq = None
    if qqq_intraday is not None and not qqq_intraday.empty:
        try:
            stock_ret = price / df['Close'].iloc[-6] - 1
            qqq_ret = qqq_intraday['Close'].iloc[-1] / qqq_intraday['Close'].iloc[-6] - 1
            rs_vs_qqq = stock_ret - qqq_ret
        except Exception:
            rs_vs_qqq = None

    # Price context (20)
    if near_ema20:
        score += 10; reasons.append("near 5m EMA20")
    if price > ema(daily_df['Close'], span=50).iloc[-1]:
        score += 10; reasons.append("above daily 50EMA")

    # Momentum (25)
    if 30 <= rsi_val <= 40:
        score += 10; reasons.append(f"RSI {rsi_val:.1f} in 30-40")
    elif 40 < rsi_val <= 50:
        score += 5; reasons.append(f"RSI {rsi_val:.1f} in 40-50")
    if macd_up:
        score += 10; reasons.append("MACD hist rising")
    if reclaim_vwap:
        score += 5; reasons.append("above VWAP")

    # Volume (20)
    if recent_breakout_vol >= 1.5 * avg_5min_vol:
        score += 10; reasons.append("recent breakout vol >=1.5x avg")
    if dip_volume < recent_breakout_vol:
        score += 10; reasons.append("dip volume lower than breakout vol")

    # Relative strength (15)
    if rs_vs_qqq is not None and rs_vs_qqq > 0:
        score += 15; reasons.append("outperforming QQQ recently")
    elif rs_vs_qqq is None:
        if price > df['Close'].iloc[-30] if len(df) > 30 else False:
            score += 7; reasons.append("intraday uptrend (partial RS)")

    # News placeholder (10)
    score += 10; reasons.append("news assumed neutral (no API)")

    # Gap down check
    if gap_pct <= -MIN_GAP_DOWN_PCT and gap_pct >= -MAX_GAP_DOWN_PCT:
        score += 5; reasons.append(f"gap down {gap_pct*100:.1f}% (candidate)")
    elif gap_pct < -MAX_GAP_DOWN_PCT:
        reasons.append(f"large gap down {gap_pct*100:.1f}% (skip)"); return _result(ticker, 0, "AVOID", None, None, None, None, reasons, avg_dollar_vol, atr_pct)

    beta = beta_vs_market(daily_df)
    if not math.isnan(beta) and beta >= 1.5 and beta <= 3.0:
        score += 5; reasons.append(f"beta {beta:.2f} in sweet spot")

    score = min(100, int(round(score)))
    if score >= 75:
        signal = "STRONG"
    elif score >= 55:
        signal = "CONSIDER"
    elif score >= 35:
        signal = "WATCH"
    else:
        signal = "AVOID"

    entry = float(df['EMA20'].iloc[-1]) if near_ema20 else float(price)
    atr5 = atr(df, period=14).iloc[-1] if 'ATR' not in df.columns else df['ATR'].iloc[-1]
    if math.isnan(atr5) or atr5 == 0:
        atr5 = (df['High'] - df['Low']).tail(10).mean()
    recent_swing_low = df['Low'].tail(20).min()
    stop = float(max(recent_swing_low - 0.002 * price, entry - 2 * atr5))
    target1 = round(entry * (1 + 0.03), 6)   # quick 3% partial
    target2 = round(entry * (1 + TARGET_PROFIT_PCT), 6)  # main target (5%)
    return _result(ticker, score, signal, entry, stop, target1, target2, reasons, avg_dollar_vol, atr_pct, beta=beta)

def _result(ticker, score, signal, entry, stop, t1, t2, reasons, avg_dollar_vol, atr_pct, beta=None):
    return {
        "ticker": ticker,
        "score": score,
        "signal": signal,
        "entry": entry,
        "stop": stop,
        "target1": t1,
        "target2": t2,
        "reasons": reasons,
        "avg_dollar_vol": avg_dollar_vol,
        "atr_pct": atr_pct,
        "beta": beta
    }

# -----------------------
# MAIN SCAN LOOP
# -----------------------
def run_scan_once():
    tickers = load_watchlist(WATCHLIST_CSV)
    qqq = fetch_intraday_5m("QQQ", days=7)
    results = []
    errors = []

    for t in tickers:
        try:
            intraday = fetch_intraday_5m(t, days=7)
            daily = fetch_daily(t, days=365)
            if daily is None or daily.empty:
                print(f"{t}: no daily data, skipping")
                # still append a consistent result so DataFrame columns exist
                results.append(_result(t, 0, "AVOID", None, None, None, None, ["no daily data"], 0, 0, beta=None))
                continue

            res = score_dip_setup(t, intraday, daily, qqq_intraday=qqq)
            # ensure res is a dict and contains expected keys
            if not isinstance(res, dict):
                raise ValueError(f"score_dip_setup returned non-dict for {t}")
            # normalize missing keys by re-creating via _result if needed
            expected_keys = {"ticker","score","signal","entry","stop","target1","target2","reasons","avg_dollar_vol","atr_pct","beta"}
            if not expected_keys.issubset(set(res.keys())):
                # fill missing keys with safe defaults
                safe = {k: res.get(k, None) for k in expected_keys}
                res = safe
            results.append(res)

            # print summary for strong signals
            if res.get('signal') in ("STRONG", "CONSIDER"):
                print(f"[{res.get('signal')}] {res.get('ticker')} score={res.get('score')} entry={res.get('entry')} stop={res.get('stop')} target={res.get('target2')}")
                for r in res.get('reasons', []):
                    print("   -", r)

        except Exception as e:
            # log error and append a safe AVOID row so DataFrame columns remain consistent
            err_msg = f"Error scanning {t}: {e}"
            print(err_msg)
            errors.append(err_msg)
            results.append(_result(t, 0, "AVOID", None, None, None, None, [str(e)], 0, 0, beta=None))

    # Build DataFrame safely
    if not results:
        # no results at all: create empty DataFrame with expected columns
        cols = ["ticker","score","signal","entry","stop","target1","target2","reasons","avg_dollar_vol","atr_pct","beta"]
        df_out = pd.DataFrame(columns=cols)
    else:
        df_out = pd.DataFrame(results)

    # Ensure 'score' column exists and is numeric
    if 'score' not in df_out.columns:
        df_out['score'] = 0
    df_out['score'] = pd.to_numeric(df_out['score'], errors='coerce').fillna(0).astype(int)

    # Sort safely
    try:
        df_out = df_out.sort_values(by="score", ascending=False).reset_index(drop=True)
    except Exception as e:
        print("Warning: could not sort by score:", e)

    # Save CSV
    try:
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"Scan complete. Results saved to {OUTPUT_CSV}")
    except Exception as e:
        print("Warning: failed to save CSV:", e)

    # Optionally print top candidates
    top = df_out[df_out['signal'].isin(['STRONG','CONSIDER'])].head(20)
    if not top.empty:
        print("\nTop candidates:")
        print(top[['ticker','score','signal','entry','stop','target2','avg_dollar_vol','atr_pct','beta']].to_string(index=False))
    else:
        print("No strong/consider signals right now.")

    # return DataFrame for further use
    return df_out

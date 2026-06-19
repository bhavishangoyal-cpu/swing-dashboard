import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os


# =========================
# CONFIG
# =========================

WATCHLIST_PATH = "watchlist.csv"

REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}


# =========================
# INDICATORS
# =========================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series = series.astype(float)
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI14"] = rsi(df["Close"], 14)
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    return df


# =========================
# HELPERS
# =========================

def get_last_values(df: pd.DataFrame):
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
        "day_range_pos": (float(df["Close"].iloc[-1]) - float(df["Low"].iloc[-1])) / (float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1]) + 1e-9),
    }


def detect_trend(ema20_last: float, ema50_last: float) -> str:
    if np.isnan(ema20_last) or np.isnan(ema50_last):
        return "UNKNOWN"
    if ema20_last > ema50_last:
        return "UP"
    elif ema20_last < ema50_last:
        return "DOWN"
    return "SIDEWAYS"


def detect_support_resistance(df: pd.DataFrame, lookback: int = 40):
    recent = df.tail(lookback)

    support = recent['Low'].rolling(5).min().iloc[-1]
    resistance = recent['High'].rolling(5).max().iloc[-1]

    return support, resistance



def is_near_level(price: float, level: float, tolerance: float = 0.02) -> bool:
    if level <= 0:
        return False
    return abs(price - level) / level <= tolerance


def bullish_engulfing(o_prev, c_prev, o_last, c_last) -> bool:
    return (c_prev < o_prev) and (c_last > o_last) and (c_last >= o_prev) and (o_last <= c_prev)


def bearish_engulfing(o_prev, c_prev, o_last, c_last) -> bool:
    return (c_prev > o_prev) and (c_last < o_last) and (c_last <= o_prev) and (o_last >= c_prev)


def hammer(o_last, h_last, l_last, c_last) -> bool:
    body = abs(c_last - o_last)
    rng = h_last - l_last
    if rng == 0:
        return False
    lower_shadow = min(o_last, c_last) - l_last
    return (lower_shadow > 2 * body) and (body / rng < 0.4)


def volume_strong(v_last: float, vol_avg20: float, factor: float = 1.5) -> bool:
    if np.isnan(vol_avg20) or vol_avg20 == 0:
        return False
    return v_last > factor * vol_avg20


# =========================
# SAFE HISTORY LOADER
# =========================

def safe_history(ticker: str, interval: str, period: str = "7d") -> pd.DataFrame | None:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval, prepost=False)
    except:
        return None

    if df is None or df.empty:
        return None

    if not REQUIRED_COLS.issubset(df.columns):
        return None

    df = df.dropna(subset=list(REQUIRED_COLS))
    if df.empty:
        return None

    return df


# =========================
# ANALYSIS ENGINE
# =========================

def analyze_ticker(ticker: str, interval: str) -> dict:
    data = safe_history(ticker, interval=interval, period="7d")
    if data is None or len(data) < 60:
        return {"ticker": ticker, "status": "NO_DATA", "interval": interval}

    df = compute_indicators(data).dropna()
    if len(df) < 30:
        return {"ticker": ticker, "status": "NO_DATA", "interval": interval}

    vals = get_last_values(df)

    # =========================
    # FIXED TREND LOGIC (STRONG)
    # =========================
    trend = "UP" if (vals["c_last"] > vals["ema20_last"] > vals["ema50_last"]) else "DOWN"

    # =========================
    # FIXED SUPPORT/RESISTANCE
    # =========================
    support, resistance = detect_support_resistance(df)

    near_support = is_near_level(vals["c_last"], support, tolerance=0.02)
    near_resistance = is_near_level(vals["c_last"], resistance, tolerance=0.02)

    rsi_up = vals["rsi_last"] > vals["rsi_prev"]
    rsi_down = vals["rsi_last"] < vals["rsi_prev"]

    bull_eng = bullish_engulfing(vals["o_prev"], vals["c_prev"], vals["o_last"], vals["c_last"])
    bear_eng = bearish_engulfing(vals["o_prev"], vals["c_prev"], vals["o_last"], vals["c_last"])
    is_hammer = hammer(vals["o_last"], vals["h_last"], vals["l_last"], vals["c_last"])

    vol_ok = volume_strong(vals["v_last"], vals["vol_avg20"], factor=1.5)

    # =========================
    # LONG SCORE
    # =========================
    long_score = sum([
        trend == "UP",
        near_support,
        (28 <= vals["rsi_last"] <= 70) and rsi_up,
        bull_eng or is_hammer,
        vol_ok,
        vals["day_range_pos"] >= 0.75,  # closing in top 25% of day's range
    ])

    # =========================
    # SHORT SCORE
    # =========================
    short_score = sum([
        trend == "DOWN",
        near_resistance,
        (55 <= vals["rsi_last"] <= 75) and rsi_down,
        bear_eng,
        vol_ok,
    ])

    # =========================
    # FIXED DECISION LOGIC
    # =========================
    if long_score >= 4:
        decision = "BUY"
    elif short_score >= 4:
        decision = "SHORT"
    elif (trend in ["UP", "DOWN"]) and (near_support or near_resistance):
        decision = "WAIT"
    else:
        decision = "NO ENTER"

    # =========================
    # CONFIRMED LOGIC (SAFE)
    # =========================
    confirmed = (
        (decision == "BUY") and
        (trend == "UP") and
        (long_score >= 3) and
        (28 <= vals["rsi_last"] <= 70) and
        near_support and
        (vals["v_last"] > 1.5 * vals["vol_avg20"])
    )

    confirmed_label = "CONFIRMED" if confirmed else "NOT CONFIRMED"

    # =========================
    # NEW: SIGNAL STRENGTH (0–10)
    # =========================
    strength = 0

    # Trend strength
    if vals["c_last"] > vals["ema20_last"] > vals["ema50_last"]:
        strength += 2
    elif vals["ema20_last"] > vals["ema50_last"]:
        strength += 1

    # RSI strength
    if 35 <= vals["rsi_last"] <= 50:
        strength += 2
    elif 28 <= vals["rsi_last"] <= 70:
        strength += 1

    # Support strength
    dist = abs(vals["c_last"] - support) / support
    if dist <= 0.01:
        strength += 2
    elif dist <= 0.02:
        strength += 1

    # Volume strength
    if vals["v_last"] > 1.3 * vals["vol_avg20"]:
        strength += 2
    elif vals["v_last"] > 1.1 * vals["vol_avg20"]:
        strength += 1

    # Candle strength
    if bull_eng or is_hammer:
        strength += 1

    # Multi-timeframe alignment
    if confirmed:
        strength += 1

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


# =========================
# STREAMLIT UI
# =========================

def load_watchlist(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Watchlist file not found: {path}")
        return pd.DataFrame(columns=["Yahoo Ticker", "Company Name"])

    df = pd.read_csv(path)
    df["Yahoo Ticker"] = df["Yahoo Ticker"].astype(str).str.strip()
    df["Company Name"] = df["Company Name"].astype(str).str.strip()

    return df[["Yahoo Ticker", "Company Name"]]



def decision_color(val: str) -> str:
    if val == "BUY":
        return "background-color:#2ECC71;color:black;"
    elif val == "WAIT":
        return "background-color:#F1C40F;color:black;"
    elif val == "NO ENTER":
        return "background-color:#E74C3C;color:white;"
    return ""


def confirmed_color(val: str) -> str:
    if val == "CONFIRMED":
        return "background-color:#27AE60;color:white;"
    else:
        return "background-color:#AAB7B8;color:black;"


def main():
    st.set_page_config(page_title="Atharv Swing Scanner", layout="wide")
    st.title("Atharv – Swing Trading Scanner (5m + 15m)")

    watchlist_df = load_watchlist(WATCHLIST_PATH)
    tickers = watchlist_df["Yahoo Ticker"].tolist()
    st.write(f"Loaded **{len(tickers)}** tickers from watchlist.csv")

    intervals = ["5m", "15m"]

    if st.button("Run Scanner"):
        for interval in intervals:
            st.subheader(f"Interval: {interval}")

            rows = []
            for _, row in watchlist_df.iterrows():
                t = row["Yahoo Ticker"]
                company_name = row["Company Name"]
                res = analyze_ticker(t, interval)

                if res["status"] != "OK":
                    rows.append({
                        "Ticker": t,
                        "Company (Ticker)": f"{company_name} ({t})",
                        "Decision": "NO ENTER",
                        "Trend": "",
                        "Close": "",
                        "Support": "",
                        "Resistance": "",
                        "RSI": "",
                        "LongScore": "",
                        "ShortScore": "",
                        "CONFIRMED": "",
                        "RangePos%": "",
                    })
                else:
                    rows.append({
                        "Ticker": res["ticker"],
                        "Company (Ticker)": f"{company_name} ({res['ticker']})",
                        "Decision": res["decision"],
                        "Trend": res["trend"],
                        "Close": round(res["close"], 2),
                        "Support": round(res["support"], 2),
                        "Resistance": round(res["resistance"], 2),
                        "RSI": round(res["rsi"], 1),
                        "LongScore": res["long_score"],
                        "ShortScore": res["short_score"],
                        "CONFIRMED": res["confirmed"],
                        "Strength": res["strength"],
                        "RangePos%": res["range_pos"],
                    })

            df_res = pd.DataFrame(rows)

            df_res["DecisionRank"] = df_res["Decision"].map({
                "BUY": 0,
                "WAIT": 1,
                "NO ENTER": 2
            }).fillna(3)

            df_res["ConfirmedRank"] = df_res["CONFIRMED"].map({
                "CONFIRMED": 0,
                "NOT CONFIRMED": 1
            }).fillna(2)

            # Sort: BUY first, then WAIT, then NO ENTER
            order = {"BUY": 0, "WAIT": 1, "NO ENTER": 2}
            df_res["Rank"] = df_res["Decision"].map(order).fillna(3)
            df_res = df_res.sort_values(
                ["DecisionRank", "ConfirmedRank", "Strength", "LongScore"],
                ascending=[True, True, False, False]
            ).drop(columns=["DecisionRank", "ConfirmedRank"])

            # Style decision + confirmed columns
            styled = df_res.style.apply(
                lambda col: [decision_color(v) for v in col],
                subset=["Decision"]
            ).apply(
                lambda col: [confirmed_color(v) for v in col],
                subset=["CONFIRMED"]
            )

            st.dataframe(styled, use_container_width=True)


if __name__ == "__main__":
    main()

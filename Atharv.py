import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(layout="wide")
st.title("🦅 Institutional Alpha Matrix (Stable)")

sector_map = {
    "Tech": {"core": "QQQ", "bull": "TQQQ", "bear": "SQQQ"},
    "S&P 500": {"core": "SPY", "bull": "UPRO", "bear": "SPXU"},
    "Semis": {"core": "SOXX", "bull": "SOXL", "bear": "SOXS"}
}


@st.fragment(run_every=30)
def scan():
    # 1. Get all tickers
    tickers = list(set([cfg[k] for cfg in sector_map.values() for k in ["core", "bull", "bear"]]))

    # 2. Download - keeping default index structure
    data = yf.download(tickers, period="5d", interval="15m", progress=False)

    rows = []
    for label, cfg in sector_map.items():
        core = cfg["core"]

        # Robust check: Data is a MultiIndex (Price, Ticker)
        if core not in data['Close'].columns: continue

        # Extract series
        close = data['Close'][core]
        vol = data['Volume'][core]

        # Calculate VWAP manually
        vwap = (close * vol).cumsum() / vol.cumsum()
        z = (close.iloc[-1] - vwap.iloc[-1]) / (close.rolling(20).std().iloc[-1] + 1e-9)

        # Signals
        signal = "⏳ MONITORING"
        if z < -2.0:
            signal = f"🔥 BUY LONG: {cfg['bull']}"
        elif z > 2.0:
            signal = f"🚨 BUY SHORT: {cfg['bear']}"

        rows.append({
            "Sector": label,
            "Price": f"${close.iloc[-1]:.2f}",
            "Z-Score": round(z, 2),
            "ACTION": signal
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.warning("Data stream active, but processing error.")


scan()
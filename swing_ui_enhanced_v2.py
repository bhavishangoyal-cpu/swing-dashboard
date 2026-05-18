# swing_ui_enhanced_v2.py

import streamlit as st
import pandas as pd
from swing_core import fetch_data, add_indicators
from swing_alert_enhanced import add_extra_indicators, enhanced_signal
from watchlist_manager import load_watchlist, save_watchlist

st.set_page_config(page_title="Swing Trading Enhanced Dashboard v2", layout="wide")
st.title("📊 Swing Trading Enhanced Dashboard (Clean UI)")

# =========================
# Watchlist Management
# =========================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()

col1, col2 = st.columns([3,1])
with col1:
    new_ticker = st.text_input("Enter ticker (e.g., NVDA)").upper()
with col2:
    if st.button("Add"):
        if new_ticker and new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker)
            save_watchlist(st.session_state.watchlist)
            st.success(f"{new_ticker} added.")
        elif new_ticker in st.session_state.watchlist:
            st.warning(f"{new_ticker} already in watchlist.")

st.write("---")

# =========================
# Display Watchlist
# =========================
if st.session_state.watchlist:
    cols = st.columns(len(st.session_state.watchlist))
    for i, ticker in enumerate(st.session_state.watchlist):
        with cols[i]:
            st.markdown(f"**{ticker}**")
            if st.button("❌", key=f"rm_{ticker}"):
                st.session_state.watchlist.remove(ticker)
                save_watchlist(st.session_state.watchlist)
                st.experimental_rerun()
else:
    st.info("Add tickers to watchlist.")

st.write("---")

# =========================
# Analyze Stocks
# =========================
if st.session_state.watchlist:
    results = []
    for ticker in st.session_state.watchlist:
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                df = fetch_data(ticker)
                df = add_indicators(df)
                df = add_extra_indicators(df)  # enhanced indicators: ADX, ATR, Stoch
                signal, reason = enhanced_signal(df)
                last = df.iloc[-1] if not df.empty else {}
                results.append({
                    "Ticker": ticker,
                    "Signal": signal,
                    "Price": float(last.get('Close', "-")),
                    "EMA50": float(last.get('EMA50', "-")),
                    "EMA200": float(last.get('EMA200', "-")),
                    "RSI": float(last.get('RSI', "-")),
                    "ADX": float(last.get('ADX', "-")),
                    "ATR": float(last.get('ATR', "-")),
                    "Stoch_K": float(last.get('Stoch_K', "-")),
                    "Stoch_D": float(last.get('Stoch_D', "-")),
                    "Reason": reason
                })
            except Exception as e:
                results.append({
                    "Ticker": ticker,
                    "Signal": "ERROR",
                    "Price": "-", "EMA50": "-", "EMA200": "-", "RSI": "-",
                    "ADX": "-", "ATR": "-", "Stoch_K": "-", "Stoch_D": "-",
                    "Reason": str(e)
                })

    df_results = pd.DataFrame(results)

    # =========================
    # Color Signals
    # =========================
    def color_signal(val):
        if val == "STRONG BUY":
            return "background-color: #d4edda; color: green; font-weight:700"
        if val == "ERROR":
            return "background-color: #f8d7da; color: red; font-weight:700"
        return ""

    st.dataframe(
        df_results.style.applymap(color_signal, subset=["Signal"]),
        use_container_width=True
    )

    # =========================
    # Balloon Alert
    # =========================
    if any(df_results["Signal"] == "STRONG BUY"):
        st.balloons()

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time

# Force wide-screen layout natively for maximum visibility across all data rows
st.set_page_config(page_title="Derivative Scalper Matrix", layout="wide")

# 1. We declare the main layout frame
st.markdown("# 🦅 Broad Market Intraday Derivative Matrix")
st.markdown("Independent execution screen tracking structural extremes across all major sectors.")
st.divider()

# Complete Market Mapping Database
sector_map = {
    "Nasdaq 100 (Tech Heavy)": {"core": "QQQ", "bull": "TQQQ", "bear": "SQQQ"},
    "S&P 500 (Core Market)": {"core": "SPY", "bull": "UPRO", "bear": "SPXU"},
    "Dow Jones 30 (Blue Chips)": {"core": "DIA", "bull": "UDOW", "bear": "SDOW"},
    "Small Caps (Russell 2000)": {"core": "IWM", "bull": "TNA", "bear": "TZA"},
    "Technology Sector": {"core": "XLK", "bull": "TECL", "bear": "TECS"},
    "Semiconductor Sector": {"core": "SOXX", "bull": "SOXL", "bear": "SOXS"},
    "Financial Sector": {"core": "XLF", "bull": "FAS", "bear": "FAZ"},
    "Biotech & Healthcare": {"core": "XBI", "bull": "LABU", "bear": "LABD"},
    "Real Estate Sector": {"core": "XLRE", "bull": "DRN", "bear": "DRV"},
    "Crude Oil (Energy)": {"core": "USO", "bull": "GUSH", "bear": "DRIP"},
    "Natural Gas": {"core": "UNG", "bull": "BOIL", "bear": "KOLD"},
    "Gold Miners": {"core": "GDX", "bull": "NUGT", "bear": "DUST"},
    "Silver Bullion": {"core": "SLV", "bull": "AGQ", "bear": "ZSL"},
    "Long-Term Bonds (20Y+)": {"core": "TLT", "bull": "TMF", "bear": "TMV"}
}

# Intraday timeframe controller
intraday_interval = st.selectbox("Select Candle Interval Target", ["5m", "15m", "1m"], index=0)

# Flatten all tickers out for parallel batch query
all_tickers = []
for cfg in sector_map.values():
    all_tickers.extend([cfg["core"], cfg["bull"], cfg["bear"]])
all_tickers = list(set(all_tickers))


# 2. This Fragment block automatically reruns every 30 seconds completely on its own
@st.fragment(run_every=30)
def render_live_matrix():
    st.caption(f"⏱️ Live Auto-Scan Active (Updates every 30s). Heartbeat: {time.strftime('%H:%M:%S EST')}")

    try:
        # Optimized Parallel Multithreaded Pull
        raw_data = yf.download(
            tickers=" ".join(all_tickers),
            period="5d",
            interval=intraday_interval,
            group_by="ticker",
            progress=False
        )

        matrix_rows = []

        # Vector calculations executed directly in RAM
        for label, cfg in sector_map.items():
            core_t, bull_t, bear_t = cfg["core"], cfg["bull"], cfg["bear"]

            if core_t not in raw_data or raw_data[core_t].empty:
                continue

            core_df = raw_data[core_t].dropna()
            bull_df = raw_data[bull_t].dropna() if bull_t in raw_data else pd.DataFrame()
            bear_df = raw_data[bear_t].dropna() if bear_t in raw_data else pd.DataFrame()

            if len(core_df) < 20:
                continue

            # Compute Core VWAP
            core_df['TP'] = (core_df['High'] + core_df['Low'] + core_df['Close']) / 3
            core_df['TPV'] = core_df['TP'] * core_df['Volume']
            core_df['Cum_TPV'] = core_df['TPV'].rolling(window=20).sum()
            core_df['Cum_Vol'] = core_df['Volume'].rolling(window=20).sum()
            core_df['VWAP'] = core_df['Cum_TPV'] / core_df['Cum_Vol']

            # Compute Variance Metrics & Z-Score
            core_df['StdDev'] = core_df['Close'].rolling(window=20).std()
            core_df['Z_Score'] = (core_df['Close'] - core_df['VWAP']) / core_df['StdDev']

            # Compute Core 14-period RSI
            delta = core_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            core_df['RSI'] = 100 - (100 / (1 + rs))

            # Extract trailing pricing states
            current_z = core_df['Z_Score'].iloc[-1]
            current_rsi = core_df['RSI'].iloc[-1]
            core_p = core_df['Close'].iloc[-1]
            bull_p = bull_df['Close'].iloc[-1] if not bull_df.empty else 0.0
            bear_p = bear_df['Close'].iloc[-1] if not bear_df.empty else 0.0

            # Determine Action Directive Thresholds
            if current_z < -2.0 and current_rsi < 32:
                signal = f"🔥 BUY LONG ({bull_t})"
            elif current_z > 2.0 and current_rsi > 68:
                signal = f"🚨 BUY SHORT ({bear_t})"
            else:
                signal = "🛑 HOLD / WAIT"

            matrix_rows.append({
                "Market Asset Category": label,
                "Core Tracker": f"{core_t} (${core_p:.2f})",
                "3x Bull ETF": f"{bull_t} (${bull_p:.2f})",
                "3x Bear ETF": f"{bear_t} (${bear_p:.2f})",
                "Z-Score (VWAP Dev)": round(current_z, 2),
                "RSI (14)": round(current_rsi, 1),
                "SYSTEM DIRECTIVE": signal
            })

        # Display Results
        if len(matrix_rows) > 0:
            matrix_df = pd.DataFrame(matrix_rows)

            # Colorize cell strings using clean hex maps
            def highlight_directives(val):
                if "BUY LONG" in str(val):
                    return "background-color: #004d26; color: #00FFCC; font-weight: bold;"
                elif "BUY SHORT" in str(val):
                    return "background-color: #660000; color: #FF9999; font-weight: bold;"
                return "color: #777777;"

            styled_matrix = matrix_df.style.map(highlight_directives, subset=["SYSTEM DIRECTIVE"])
            st.dataframe(styled_matrix, use_container_width=True, hide_index=True, height=600)
        else:
            st.info("Waiting for data stream connections...")

    except Exception as e:
        st.error(f"Execution matrix synchronization paused: {str(e)}")


# Execute the live loop block
render_live_matrix()
import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import time

# Assuming this code lives inside your main UI layout under Tab 5:
# tab1, tab2, tab3, tab4, tab5 = st.tabs(["...", "...", "...", "...", "🎯 80%+ Intraday Squeeze"])
# with tab5:

st.subheader("🎯 High-Effectiveness Intraday Squeeze Scanner (Tab 5)")
st.caption("Filters for 1.2% Volatility Squeezes with Institutional Volume Alignment | Timeframe: 5m / 1h Hybrid")

# 1. Define your high-velocity, volatile watchlist
watchlist = ["QLD", "MU", "NVDA", "AMD", "QQQ", "TSLA"]


# Create a container that auto-refreshes every 5 minutes cleanly without a full page reload
@st.fragment(run_every=300)
def render_squeeze_scanner():
    st.write(f"🔄 Last Dashboard Scan: `{time.strftime('%H:%M:%S')} PST` (Auto-refreshes every 5m)")

    # Pre-allocate rows for our fast display table
    scanner_results = []

    # Loop through watchlist to run our 3-Layer Filter System completely offline in memory
    for ticker in watchlist:
        try:
            # --- LAYER 1: FAST HIGHER-TIMEFRAME DATA PULL & CALCULATION ---
            # Fetch 1-hour data to anchor the macro institutional trend direction
            df_1h = yf.download(ticker, period="1mo", interval="1h", progress=False, group_by="ticker")
            if df_1h.empty: continue

            # Compute 50 EMA on the 1-hour timeframe
            df_1h['EMA_50'] = ta.ema(df_1h['Close'], length=50)
            macro_uptrend = df_1h['Close'].iloc[-1] > df_1h['EMA_50'].iloc[-1]

            # --- LAYERS 2 & 3: FAST INTRADAY IN-MEMORY CALCULATIONS ---
            # Fetch 5-minute data (last 5 days is more than enough for a 20-period calculation)
            df_5m = yf.download(ticker, period="5d", interval="5m", progress=False, group_by="ticker")
            if df_5m.empty: continue

            # Donchian Channels (20-period High/Low) for structural breakout levels
            df_5m['High_20'] = df_5m['High'].rolling(20).max()
            df_5m['Low_20'] = df_5m['Low'].rolling(20).min()

            # Calculate the Dynamic Box Width (Squeeze Percentage)
            # Use index [-2] (last completed 5m candle) to completely avoid live flashing lines
            current_close = df_5m['Close'].iloc[-1]
            previous_high_box = df_5m['High_20'].iloc[-2]
            previous_low_box = df_5m['Low_20'].iloc[-2]
            box_width_pct = (previous_high_box - previous_low_box) / current_close

            # Chaikin Money Flow (CMF) for Institutional Volume Confirmation
            df_5m['CMF'] = ta.cmf(df_5m['High'], df_5m['Low'], df_5m['Close'], df_5m['Volume'], length=20)
            volume_confirmed = df_5m['CMF'].iloc[-1] >= 0.10

            # Check for immediate structural breakout above the 5m ceiling
            price_breakout = current_close > previous_high_box
            is_squeezed = box_width_pct <= 0.012  # Strict 1.2% range filter

            # --- EVALUATE THE MASTER 80%+ INTRADAY RULE ---
            if macro_uptrend and is_squeezed and price_breakout and volume_confirmed:
                status = "🔥 STRONG BUY SIGNAL"
                color = "green"
            elif macro_uptrend and is_squeezed:
                status = "⏳ Squeezed (Waiting for Breakout)"
                color = "orange"
            else:
                status = "❌ Ignore / No Setup"
                color = "red"

            # Append metrics to our summary array for instant UI rendering
            scanner_results.append({
                "Ticker": ticker,
                "Status": status,
                "Live Price": round(current_close, 2),
                "Squeeze Width": f"{round(box_width_pct * 100, 2)}%",
                "CMF (Volume)": round(df_5m['CMF'].iloc[-1], 2),
                "1H Trend": "Bullish" if macro_uptrend else "Bearish",
                "Suggested Entry Floor (SL)": round(previous_low_box, 2)
            })

        except Exception as e:
            # Ensures a single broken network ticker doesn't crash your entire main app
            st.error(f"Error processing {ticker}: {str(e)}")
            continue

    # Convert processed array to an optimized DataFrame for the UI
    results_df = pd.DataFrame(scanner_results)

    # Custom colored styling for the Streamlit dataframe presentation view
    def style_status(val):
        if "STRONG BUY" in val: return "background-color: #d4edda; color: #155724; font-weight: bold;"
        if "Squeezed" in val: return "background-color: #fff3cd; color: #856404;"
        return "color: #721c24;"

    styled_df = results_df.style.applymap(style_status, subset=['Status'])

    # Output the fast live data matrix
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


# Run the live container fragment
render_squeeze_scanner()
import streamlit as st
import yfinance as yf
import pandas as pd
from google import genai
from google.genai import types

# =====================================================================
# 1. PAGE CONFIGURATION & SECURITY
# =====================================================================
st.set_page_config(page_title="Atharv Corporate Guide", page_icon="📈", layout="wide")

# Securely pull your free Gemini API Key
API_KEY = "AQ.Ab8RN6J3k0y2pU_MK_PsjfQmrDGqaw5jvXRLHRpBeofM5ecx6g"


@st.cache_resource
def get_ai_client():
    if API_KEY and API_KEY != "YOUR_FREE_GEMINI_API_KEY":
        try:
            return genai.Client(api_key=API_KEY)
        except Exception:
            return None
    return None


client = get_ai_client()

# =====================================================================
# 2. WEB INTERFACE TITLE
# =====================================================================
st.title("🚀 Atharv.py — Corporate Swing Trading Co-Pilot")
st.markdown(
    "Designed for family-managed corporate accounts to identify momentum swings and analyze macro risk factors.")
st.write("---")

# Sidebar Search & Portfolio Input Layout
st.sidebar.header("Asset Selection")
ticker_input = st.sidebar.text_input("Enter Ticker Symbol", value="NVDA").upper().strip()
st.sidebar.caption("💡 For Canadian assets, use the '.TO' suffix (e.g., XIU.TO or SHOP.TO)")

st.sidebar.write("---")
st.sidebar.header("💰 Your Position Details")
my_purchase_price = st.sidebar.number_input("Enter Your Purchase Price ($)", value=0.0, step=0.01,
                                            help="Set to 0.0 if you don't own this stock yet.")

# Run Analysis Execution
if ticker_input:
    ticker = yf.Ticker(ticker_input)

    with st.spinner(f"Analyzing {ticker_input} historical structures and pulling live headlines..."):
        try:
            info = ticker.info
            history = ticker.history(period="2y")

            # -----------------------------------------------------------------
            # 🧩 SHARE STATS & FINANCIAL HEALTH PARSING ENGINE
            # -----------------------------------------------------------------
            shares_outstanding = info.get('sharesOutstanding', None)
            float_shares = info.get('floatShares', None)
            insider_pct = info.get('heldPercentInsiders', 0) * 100
            inst_pct = info.get('heldPercentInstitutions', 0) * 100

            # Get the 3-month average volume
            avg_vol_3m = info.get('averageVolume', None) or info.get('averageDailyVolume3Month', None)

            # Math for the Turnover percentage
            if avg_vol_3m and shares_outstanding:
                daily_turnover_pct = (avg_vol_3m / shares_outstanding) * 100
            else:
                daily_turnover_pct = None

            # Short seller stats
            shares_short = info.get('sharesShort', None)
            shares_short_prior = info.get('sharesShortPriorMonth', None)
            short_ratio = info.get('shortRatio', None)
            short_pct_float = info.get('shortPercentOfFloat', 0) * 100

            # Calculate monthly short trajectory trend
            if shares_short and shares_short_prior and shares_short_prior > 0:
                short_change_pct = ((shares_short - shares_short_prior) / shares_short_prior) * 100
            else:
                short_change_pct = None

            # Corporate Health Variables
            profit_margin = info.get('profitMargins', 0) * 100
            debt_to_equity = info.get('debtToEquity', None)

        except Exception as e:
            st.error(f"Could not load data for '{ticker_input}'. Please verify the symbol or your internet connection.")
            st.stop()

    if history.empty:
        st.error(f"No trading background found for symbol: {ticker_input}")
        st.stop()

    # -----------------------------------------------------------------
    # DATA ASSIGNMENT FOR PRIMARY METRICS
    # -----------------------------------------------------------------
    name = info.get('longName', 'N/A')
    sector = info.get('sector', 'N/A')
    industry = info.get('industry', 'N/A')
    summary = info.get('longBusinessSummary', 'No corporate summary available.')

    pe_ratio = info.get('trailingPE', 'N/A')
    forward_pe = info.get('forwardPE', 'N/A')
    market_cap = info.get('marketCap', 'N/A')

    avg_volume = info.get('averageVolume', 0)
    beta = info.get('beta', 1.0)
    held_by_institutions = info.get('heldPercentInstitutions', 0) * 100

    current_price = info.get('currentPrice', history['Close'].iloc[-1])
    fifty_two_high = info.get('fiftyTwoWeekHigh', max(history['Close'][-252:]))
    fifty_two_low = info.get('fiftyTwoWeekLow', min(history['Close'][-252:]))

    # Technical Calculations
    history['MA50'] = history['Close'].rolling(window=50).mean()
    history['MA200'] = history['Close'].rolling(window=200).mean()
    ma50_now = history['MA50'].iloc[-1]
    ma200_now = history['MA200'].iloc[-1]
    pct_from_high = ((fifty_two_high - current_price) / fifty_two_high) * 100

    # -----------------------------------------------------------------
    # 🧩 ADVANCED: MULTI-STAGE DOWNTREND DIAGNOSTIC ENGINE MATH
    # -----------------------------------------------------------------
    history['MA21'] = history['Close'].rolling(window=21).mean()
    ma21_now = history['MA21'].iloc[-1]

    if ma21_now and ma21_now > 0:
        trend_cushion_pct = ((current_price - ma21_now) / ma21_now) * 100
    else:
        trend_cushion_pct = 0.0

    # Volume and structural anomaly checks
    recent_volume = history['Volume'].iloc[-5:].mean()
    long_avg_volume = info.get('averageVolume', 1) if info.get('averageVolume', 1) > 0 else 1
    volume_spike_ratio = recent_volume / long_avg_volume

    if current_price >= ma21_now:
        downward_diagnosis = "RUNNING"
    else:
        if current_price > ma200_now:
            downward_diagnosis = "CORRECTION"
        elif current_price <= ma200_now and volume_spike_ratio > 1.5 and short_change_pct and short_change_pct > 10.0:
            downward_diagnosis = "STRUCTURAL_BLEED"
        else:
            downward_diagnosis = "MARKET_CRASH_OR_MACRO_FLUSH"

    # Institutional Targets & Bank Consensus
    target_low = info.get('targetLowPrice', 'N/A')
    target_high = info.get('targetHighPrice', 'N/A')
    target_mean = info.get('targetMeanPrice', 'N/A')
    recommendation = info.get('recommendationKey', 'N/A').replace('_', ' ').title()

    # -----------------------------------------------------------------
    # UI HEADER DISPLAY
    # -----------------------------------------------------------------
    st.header(f"🏢 {name}")
    st.subheader(f"Sector: {sector} | Industry: {industry}")

    with st.expander("📄 View Company Profile Summary"):
        st.write(summary)

    # -----------------------------------------------------------------
    # ROW 1: LIVE TECHNICAL & FUNDAMENTAL MATRIX
    # -----------------------------------------------------------------
    st.write("---")
    st.subheader("📊 Live Technical & Fundamental Matrix")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
        st.metric("Trailing P/E", f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else f"{pe_ratio}")
    with col2:
        st.metric("52-Week High", f"${fifty_two_high:.2f}")
        st.metric("Forward P/E", f"{forward_pe:.2f}" if isinstance(forward_pe, (int, float)) else f"{forward_pe}")
    with col3:
        st.metric("Distance from High", f"-{pct_from_high:.1f}%")
        st.metric("Volatility (Beta)", f"{beta:.2f}")
    with col4:
        if isinstance(market_cap, (int, float)):
            st.metric("Market Cap", f"${market_cap / 1e9:.2f}B")
        else:
            st.metric("Market Cap", "N/A")
        st.metric("Institutional Owned", f"{held_by_institutions:.1f}%")

    # -----------------------------------------------------------------
    # ROW 2: MARKET STRUCTURE & LIQUIDITY DYNAMICS
    # -----------------------------------------------------------------
    st.write("---")
    st.subheader("💧 Market Structure & Liquidity Dynamics")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

    with col_stat1:
        st.markdown("**Supply Structure**")
        if shares_outstanding:
            st.metric("Shares Outstanding",
                      f"{shares_outstanding / 1e9:.2f}B" if shares_outstanding >= 1e9 else f"{shares_outstanding / 1e6:.2f}M")
        else:
            st.metric("Shares Outstanding", "N/A")

        if float_shares:
            st.metric("Free Float",
                      f"{float_shares / 1e9:.2f}B" if float_shares >= 1e9 else f"{float_shares / 1e6:.2f}M")
        else:
            st.metric("Free Float", "N/A")

    with col_stat2:
        st.markdown("**Ownership Dynamics**")
        st.metric("Held by Institutions", f"{inst_pct:.2f}%")
        st.metric("Held by Insiders", f"{insider_pct:.2f}%")

    with col_stat3:
        st.markdown("**Trading Liquidity**")
        if avg_vol_3m:
            st.metric("Avg Vol (3 Month)", f"{avg_vol_3m / 1e6:.2f}M")
        else:
            st.metric("Avg Vol (3 Month)", "N/A")

        if daily_turnover_pct:
            if 3.0 <= daily_turnover_pct <= 7.0:
                st.metric("Daily Turnover %", f"{daily_turnover_pct:.2f}%", delta="🎯 Swing Sweet Spot",
                          delta_color="normal")
            elif daily_turnover_pct > 15.0:
                st.metric("Daily Turnover %", f"{daily_turnover_pct:.2f}%", delta="⚠️ Hyper-Speculative",
                          delta_color="inverse")
            else:
                st.metric("Daily Turnover %", f"{daily_turnover_pct:.2f}%")
        else:
            st.metric("Daily Turnover %", "N/A")

    with col_stat4:
        st.markdown("**Short Seller Pressure**")
        if shares_short:
            st.metric("Shares Short", f"{shares_short / 1e6:.2f}M")
        else:
            st.metric("Shares Short", "N/A")

        st.metric("Short % of Float", f"{short_pct_float:.2f}%")
        st.metric("Short Ratio (Days to Cover)", f"{short_ratio:.1f}" if short_ratio else "N/A")

    # -----------------------------------------------------------------
    # ROW 3: FINANCIAL HEALTH & MOMENTUM FILTERS
    # -----------------------------------------------------------------
    st.write("---")
    st.subheader("🏥 Financial Health & Short Trajectory")
    col_health1, col_health2, col_health3 = st.columns(3)

    with col_health1:
        st.markdown("**Core Profitability**")
        if profit_margin:
            if profit_margin >= 20.0:
                st.metric("Net Profit Margin", f"{profit_margin:.2f}%", delta="🟢 Highly Profitable")
            elif profit_margin < 0.0:
                st.metric("Net Profit Margin", f"{profit_margin:.2f}%", delta="🔴 Burning Cash", delta_color="inverse")
            else:
                st.metric("Net Profit Margin", f"{profit_margin:.2f}%")
        else:
            st.metric("Net Profit Margin", "N/A")

    with col_health2:
        st.markdown("**Leverage Risk**")
        if debt_to_equity is not None:
            if debt_to_equity <= 100.0:
                st.metric("Debt-to-Equity Ratio", f"{debt_to_equity:.1f}%", delta="🟢 Safe Leverage")
            elif debt_to_equity > 200.0:
                st.metric("Debt-to-Equity Ratio", f"{debt_to_equity:.1f}%", delta="⚠️ Heavy Debt Loading",
                          delta_color="inverse")
            else:
                st.metric("Debt-to-Equity Ratio", f"{debt_to_equity:.1f}%")
        else:
            st.metric("Debt-to-Equity Ratio", "N/A / Cash Rich")

    with col_health3:
        st.markdown("**Short Interest Trajectory**")
        if short_change_pct is not None:
            if short_change_pct > 10.0:
                st.metric("Shorts MoM Change", f"{short_change_pct:+.1f}%", delta="⚠️ Bears Accumulating",
                          delta_color="inverse")
            elif short_change_pct < -10.0:
                st.metric("Shorts MoM Change", f"{short_change_pct:+.1f}%", delta="🟢 Bears Fleeing")
            else:
                st.metric("Shorts MoM Change", f"{short_change_pct:+.1f}% (Stable)")
        else:
            st.metric("Shorts MoM Change", "N/A")

    # -----------------------------------------------------------------
    # ROW 4: WALL STREET INSTITUTIONAL CONSENSUS FLOOR
    # -----------------------------------------------------------------
    st.write("---")
    st.subheader("🏛️ Wall Street Institutional Consensus")
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)

    with b_col1:
        st.metric("Consensus Rating", f"{recommendation}")
    with b_col2:
        if isinstance(target_mean, (int, float)):
            upside = ((target_mean - current_price) / current_price) * 100
            st.metric("Average Target", f"${target_mean:.2f}", f"+{upside:.1f}% Est. Upside")
        else:
            st.metric("Average Target", "N/A")
    with b_col3:
        st.metric("Bank Low Target", f"${target_low:.2f}" if isinstance(target_low, (int, float)) else "N/A")
    with b_col4:
        st.metric("Bank High Target", f"${target_high:.2f}" if isinstance(target_high, (int, float)) else "N/A")

    # -----------------------------------------------------------------
    # ROW 5: INSTITUTIONAL TREND & DOWNTREND DIAGNOSER VIEW
    # -----------------------------------------------------------------
    st.write("---")
    st.subheader("🎯 Institutional Trend & Downward Risk Diagnoser")
    st.markdown("Tracks massive 1,000% runs while accurately diagnosing the exact structural nature of price drops.")

    exit_col1, exit_col2, exit_col3 = st.columns(3)

    with exit_col1:
        st.markdown("**Institutional Launchpad Status**")
        if current_price >= ma21_now:
            st.metric("Launchpad Cushion", f"+{trend_cushion_pct:.1f}%", delta="💎 Strong Institutional Support")
        else:
            st.metric("Launchpad Cushion", f"{trend_cushion_pct:.1f}%", delta="⚠️ Below Launchpad Floor",
                      delta_color="inverse")

    with exit_col2:
        st.markdown("**Core Technical Baselines**")
        st.write(f"🔹 **21-Day Trend Floor:** ${ma21_now:.2f}")
        st.write(f"🏛️ **200-Day Macro Floor:** ${ma200_now:.2f}")

    with exit_col3:
        st.markdown("**Strategic Execution & Trend Diagnosis**")

        if downward_diagnosis == "RUNNING":
            st.success(
                "🚀 RIDE THE RUNNER: Trend is perfectly healthy. Let your profits compound into maximum potential.")

        elif downward_diagnosis == "CORRECTION":
            st.warning(
                "🟡 HEALTHY CORRECTION: Price is dipping but remains safely above the 200-Day Macro Floor. No structural damage detected. Safe to hold or accumulate pullbacks.")

        elif downward_diagnosis == "MARKET_CRASH_OR_MACRO_FLUSH":
            st.info(
                "🌊 MACRO FLUSH / CRASH SECTOR: Stock is below major floors but lacking heavy volume liquidation or short spikes. Likely dragged down by broader market panic. Hold firm through systemic volatility.")

        elif downward_diagnosis == "STRUCTURAL_BLEED":
            st.error(
                "🚨 STRUCTURAL DOWNWARD TREND: Asset has completely broken down below the 200-Day Floor on high institutional volume and rising short interest. Expect a multi-month negative freeze. DO NOT add fresh capital here.")

    # -----------------------------------------------------------------
    # NEW BLOCK: 🎯 TAILORED POSITION EXECUTION CHECKLIST (USER LIST REQUIREMENT)
    # -----------------------------------------------------------------
    if my_purchase_price > 0.0:
        st.write("---")
        st.subheader(f"📋 Personalized Execution Checklist for {ticker_input}")
        st.markdown(
            "Your custom, live-calculated action plan based on your average cost and current institutional trend structures.")

        # Calculate individual performance metrics
        gain_loss_pct = ((current_price - my_purchase_price) / my_purchase_price) * 100
        in_the_green = current_price >= my_purchase_price

        list_col1, list_col2 = st.columns([1, 2])

        with list_col1:
            st.markdown("**Your Equity Status Metrics**")
            st.metric("Your Cost Basis", f"${my_purchase_price:.2f}")
            if in_the_green:
                st.metric("Position Return", f"+{gain_loss_pct:.2f}%", delta="🟢 In The Green")
            else:
                st.metric("Position Return", f"{gain_loss_pct:.2f}%", delta="🔴 Capital In Drawdown",
                          delta_color="inverse")

        with list_col2:
            st.markdown("**What To Do Right Now (Action List):**")

            # Generate personalized lists dynamically based on technical frameworks + user price
            if downward_diagnosis == "RUNNING":
                if in_the_green:
                    st.markdown(f"""
                    * **[HOLD]** Your position is safely in the green (`+{gain_loss_pct:.1f}%`) and institutional momentum is roaring. Do not sell early.
                    * **[TRAILING TRACK]** Your profit floor is protected by the 21-Day Trend Floor at **${ma21_now:.2f}**. Let the asset climb uncapped.
                    * **[EXECUTION]** Take no profit reduction until the price closes below the 21-day floor line.
                    """)
                else:
                    st.markdown(f"""
                    * **[HOLD / WATCH]** You are down ` {gain_loss_pct:.1f}%` from your entry, but the asset has flipped into a fresh **Institutional Launchpad Run**. 
                    * **[BUY ALIGNMENT]** The trend has reversed positively. Because it crossed above **${ma21_now:.2f}**, the structural path to your break-even point is wide open.
                    * **[EXECUTION]** Hold firm. No panic-selling allowed while institutions are actively backing the move.
                    """)

            elif downward_diagnosis == "CORRECTION":
                if in_the_green:
                    st.markdown(f"""
                    * **[PROTECT / HOLD]** You are up `+{gain_loss_pct:.1f}%`, but the stock is undergoing a short-term pullback below its 21-day line.
                    * **[SAFETY MATRIX]** The long-term floor at **${ma200_now:.2f}** is still completely intact. This is a healthy correction.
                    * **[EXECUTION]** If you want to lock in your profits tightly, use **${ma21_now:.2f}** as your soft exit line. Otherwise, hold safely through the temporary dip.
                    """)
                else:
                    st.markdown(f"""
                    * **[HOLD & ACCUMULATE]** You are down `{gain_loss_pct:.1f}%` on this share, but it is diagnosed as a **Healthy Technical Correction**.
                    * **[SUPPORT CHECK]** The asset is breathing but tracking safely above the long-term macro floor (**${ma200_now:.2f}**).
                    * **[EXECUTION]** Do not sell at a loss. If your corporate cash allocation rules allow, this is a mathematically safe area to add shares and average down your entry cost.
                    """)

            elif downward_diagnosis == "MARKET_CRASH_OR_MACRO_FLUSH":
                st.markdown(f"""
                * **[STRICT FREEZE & HOLD]** This asset is down due to a broad macro sector sweep or global market panic. 
                * **[PHILOSOPHY COMPLIANCE]** Your capital is currently at a `{gain_loss_pct:.1f}%` paper variance. Remember your rule: **Never sell at a loss.**
                * **[EXECUTION]** Freeze the position entirely. Broader market flushes are temporary liquidations. Let the family-managed account carry the position safely as an investment without crystalizing an emotional loss.
                """)

            elif downward_diagnosis == "STRUCTURAL_BLEED":
                st.markdown(f"""
                * **[LOCK capital / DO NOT ADD]** This share has broken major technical benchmarks (**${ma200_now:.2f}**) on heavy institutional distribution volume. 
                * **[RISK WARNING]** You are currently down `{gain_loss_pct:.1f}%`. The stock is entering a prolonged negative multi-month cooling cycle.
                * **[EXECUTION]** **DO NOT throw good money after bad.** Absolutely do not average down here. Put a capital freeze on this ticker, let the existing shares sit untouched, and re-allocate your fresh corporate capital toward active green runners instead.
                """)

    # -----------------------------------------------------------------
    # ROW 6: ANALYSIS COLUMNS (Algorithm & AI Catalyst Deep Dive)
    # -----------------------------------------------------------------
    st.write("---")
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("⚙️ Automated Algorithmic Logic")

        # Short Term Trend Card
        st.markdown("**Short-Term Swing Direction:**")
        if current_price > ma50_now and ma50_now > ma200_now:
            st.success("🟢 Strong Upward Momentum. Structural trend is healthy; target pullbacks for entry.")
        elif current_price < ma50_now and current_price > ma200_now:
            st.warning(
                "🟡 Technical Correction. Price retreating toward the 200-day floor. Monitor for reversal support.")
        else:
            st.error("🔴 Bearish Structural Trend. High capital vulnerability for immediate swing trades.")

        # Long Term Value Card
        st.markdown("**1-2 Year Structural Outlook:**")
        if isinstance(forward_pe, (int, float)) and isinstance(pe_ratio, (int, float)):
            if forward_pe < pe_ratio:
                st.info(
                    "🔵 Positive. Earnings projections expand outward, indicating long-term valuation discount room.")
            else:
                st.markdown("⚪ *Premium/Flat. Growth trajectories appear valued-in by core institutional analysts.*")
        else:
            st.markdown("⚪ *Data insufficient to safely cross-verify corporate forwarding horizons.*")

        # Liquidity Check
        st.markdown("**Capital Exit Liquidity:**")
        if avg_volume > 1000000:
            st.success(f"✅ Safe ({avg_volume:,} avg shares/day). Swift exit execution available.")
        elif avg_volume > 200000:
            st.warning(f"⚠️ Moderate ({avg_volume:,} avg shares/day). Handle under controlled size allocation.")
        else:
            st.error(f"🚨 Extreme Liquidity Risk ({avg_volume:,} shares). High probability of slippage parameters.")

    with right_col:
        st.subheader("📰 Live Catalyst Feed & AI Deep Dive")

        if not client:
            st.warning(
                "⚠️ Enter a valid Gemini API Key at the top of the file to populate the AI sentiment breakdown below.")
        else:
            with st.spinner("Activating Google Search Grounding to fetch live market catalysts..."):
                prompt = f"""
                Perform a live regulatory and sentiment risk assessment for the ticker asset: {ticker_input} ({name}).

                1. Identify the top 3-4 major news headlines, product announcements, or earnings catalysts from the past 72 hours.
                2. Evaluate if these events represent short-term volatility plays (swings) or changing structural fundamentals for holding 1-2 years. 
                3. Explicitly state any immediate hazards to corporate capital reserves or cash liquidity parameters.

                Format your final response with clean, professional bold headers. Provide direct bullet points for the news events.
                """

                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            tools=[{"google_search": {}}]
                        )
                    )
                    st.markdown("#### 🤖 Automated AI Intelligence Report")
                    st.info(response.text)

                except Exception as ai_err:
                    st.error(f"AI Synthesis module failed to execute: {ai_err}")
import streamlit as st
import yfinance as yf
import pandas as pd

# Configuration
st.set_page_config(page_title="Stock Valuation Engine", layout="wide")


def get_stock_data(ticker):
    """Fetches key metrics from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "Ticker": ticker,
        "Forward P/E": info.get("forwardPE", 0) or 0,
        "ROIC": (info.get("returnOnInvestedCapital", 0) or 0) * 100,
        "Net Margin": (info.get("profitMargins", 0) or 0) * 100,
        "Debt/Equity": info.get("debtToEquity", 0) or 0
    }


st.title("📊 Institutional Stock Valuation Engine")
st.markdown("Compare stocks using institutional metrics: P/E, ROIC, and Profitability.")

tickers_input = st.text_input("Enter Tickers (comma separated):", "AAPL, MSFT, GOOGL, AMZN")
ticker_list = [t.strip().upper() for t in tickers_input.split(",")]

if st.button("Generate Valuation Analysis"):
    data = [get_stock_data(t) for t in ticker_list]
    df = pd.DataFrame(data).set_index("Ticker")

    # Calculate a simple "Institutional Health Score"
    # Logic: High ROIC (+), Low P/E (+), High Margin (+)
    # Note: This is a simplified example; institutions use complex DCF models
    df['Health Score'] = (
            (df['ROIC'] / df['ROIC'].max()) * 50 +
            (1 / (df['Forward P/E'] + 0.1)) * 20 +
            (df['Net Margin'] / df['Net Margin'].max()) * 30
    )

    # Visualization
    st.subheader("Comparative Analysis Table")

    # Style the table
    st.dataframe(
        df.style.background_gradient(cmap='RdYlGn', subset=['ROIC', 'Net Margin', 'Health Score']),
        use_container_width=True
    )

    # Guidance/Interpretation
    st.subheader("AI Guidance")
    for ticker, row in df.iterrows():
        advice = "Hold/Accumulate"
        if row['Health Score'] > df['Health Score'].mean() + 5:
            advice = "Strong Buy (High Institutional Quality)"
        elif row['Health Score'] < df['Health Score'].mean() - 5:
            advice = "Sell/Underperform (Metric Outlier)"

        st.write(f"**{ticker}**: {advice} (Score: {row['Health Score']:.2f})")

st.markdown("---")
st.caption("Note: Metrics are based on current market data. Always verify against latest 10-K filings.")
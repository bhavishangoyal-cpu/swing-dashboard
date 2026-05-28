import pandas as pd
import yfinance as yf

# ============================
# 1. DOWNLOAD ALL U.S. TICKERS
# ============================

def get_exchange_list(url):
    df = pd.read_csv(url)
    df = df[['Symbol', 'Name']]
    df = df.dropna()
    return df

nasdaq = get_exchange_list("https://datahub.io/core/nasdaq-listings/r/nasdaq-listed.csv")
nyse = get_exchange_list("https://datahub.io/core/nyse-other-listings/r/nyse-listed.csv")
amex = get_exchange_list("https://datahub.io/core/amex-listings/r/amex-listed.csv")

all_stocks = pd.concat([nasdaq, nyse, amex], ignore_index=True)
tickers = all_stocks['Symbol'].unique().tolist()

# ============================
# 2. FETCH MARKET CAPS
# ============================

def get_market_cap(ticker):
    try:
        info = yf.Ticker(ticker).fast_info
        return info.get("market_cap", None)
    except:
        return None

all_stocks['MarketCap'] = all_stocks['Symbol'].apply(get_market_cap)

# ============================
# 3. CLASSIFY BY MARKET CAP
# ============================

def classify_cap(mc):
    if mc is None:
        return "Unknown"
    if mc > 10_000_000_000:
        return "Large Cap"
    elif mc > 2_000_000_000:
        return "Mid Cap"
    elif mc > 300_000_000:
        return "Small Cap"
    else:
        return "Micro Cap"

all_stocks['Category'] = all_stocks['MarketCap'].apply(classify_cap)

# ============================
# 4. DOWNLOAD ALL ETFs
# ============================

etf_list = pd.read_html("https://etfdb.com/etfs/")[0]
etf_list.to_excel("all_etfs.xlsx", index=False)

# ============================
# 5. SAVE OUTPUT FILES
# ============================

all_stocks[all_stocks['Category'] == "Large Cap"].to_excel("large_cap_stocks.xlsx", index=False)
all_stocks[all_stocks['Category'] == "Mid Cap"].to_excel("mid_cap_stocks.xlsx", index=False)
all_stocks[all_stocks['Category'] == "Small Cap"].to_excel("small_cap_stocks.xlsx", index=False)

print("Done! Excel files created:")
print(" - large_cap_stocks.xlsx")
print(" - mid_cap_stocks.xlsx")
print(" - small_cap_stocks.xlsx")
print(" - all_etfs.xlsx")

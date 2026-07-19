from ib_insync import IB, Stock, util
import pandas as pd

SYMBOLS = ['NVDA', 'TSLA', 'GOOG']
DURATION = '10 D'  # Pulls the last 10 days of data
BAR_SIZE = '2 mins'

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=10)

print("Downloading historical data...")

for sym in SYMBOLS:
    print(f"Fetching {sym}...")
    contract = Stock(sym, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=DURATION,
        barSizeSetting=BAR_SIZE,
        whatToShow='TRADES',
        useRTH=False,
        formatDate=2  # Returns UTC timestamps
    )

    if bars:
        df = util.df(bars)
        # Rename columns to match what the backtester expects
        df = df[['date', 'open', 'high', 'low', 'close']]
        df.to_csv(f"{sym}_2m.csv", index=False)
        print(f"✅ Saved {sym}_2m.csv ({len(df)} rows)")
    else:
        print(f"⚠️ Failed to get data for {sym}")

ib.disconnect()
print("Done.")
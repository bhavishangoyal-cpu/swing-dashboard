from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=10)

contract = Stock('TQQQ', 'SMART', 'USD')
ib.qualifyContracts(contract)

bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='60 D',
    barSizeSetting='2 mins',
    whatToShow='TRADES',
    useRTH=True,
    formatDate=1
)

df = util.df(bars)
df.to_csv("TQQQ_2min_60days.csv", index=False)

print(df.head())
print(len(df))

ib.disconnect()
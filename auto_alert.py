# auto_alert.py
import yfinance as yf
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import csv
import os

# === GMAIL SETUP ===
EMAIL = "bhavishangoyal@gmail.com"
APP_PASSWORD = "fkoc qhpo cmok nvzn"
TO_EMAIL = "bhavishangoyal@gmail.com"

# === VALID STOCKS ===
TICKERS = [
    "NVDA", "TSLA", "AAPL", "AMD", "SMCI", "PLTR", "MSFT", "GOOGL", "AMZN", "META",
    "QQQ", "SPY", "IWM", "ARKK", "SOXX", "XLF", "XLE", "XLK", "XLY", "XLP",
    "NFLX", "ADBE", "CRM", "ORCL", "INTC", "CSCO", "AVGO", "QCOM", "TXN", "MU"
]

# === LOG SETUP ===
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"buys_{datetime.now().year}.csv")

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Ticker", "Price", "Stop", "Target", "RSI", "Reason"])

# === INDICATORS ===
# === INDICATORS ===
def add_indicators(df):
    if 'Close' not in df.columns or 'Volume' not in df.columns:
        return df
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    df['EMA200'] = df['Close'].ewm(span=200).mean()
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Volume_Avg'] = df['Volume'].rolling(20).mean()   # ← FIXED LINE
    return df
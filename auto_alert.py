from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import winsound
import time
import threading
import pyttsx3

# --- Core High-Frequency Settings ---
SYMBOL = 'SOXL'
BAR_SIZE = '1 min'  # Drop to 1-minute bars for high trade volume
HISTORY_DURATION = '2 D'
TRADE_QTY = 100

IB_HOST = '127.0.0.1'
IB_PORT = 7497
CLIENT_ID = 3
MARKET_DATA_TYPE = 3  # Delayed = 3, Live = 1

# --- Calibrated Scalping Risk Parameters ---
HARD_STOP_LOSS_PCT = -0.0035  # Relaxed to 0.35% to survive minor 1-min noise
TARGET1_PCT = 0.0050  # Take 50% profits at +0.50%
TARGET2_PCT = 0.0100  # Take final profits at +1.00%
TRAIL_ACTIVATE = 0.0040  # Activate trailing at +0.40%
TRAIL_PULLBACK = 0.0015  # Trail distance of 0.15%

state = {
    'df': pd.DataFrame(),
    'order_pending': False,
    'order_pending_since': None,
    'buy_price': None,
    'peak_pnl_pct': 0.0,
    'entry_time': None,
    'sold_target1': False,
    'sold_target2': False,
    'last_direction': 0,
    'last_exit_reason': None,
    'last_exit_price': 0.0
}

def speak_alert(text):
    """Speaks the text on a separate thread so it doesn't freeze the scalper."""
    def target():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except:
            pass # Fails silently if the audio engine is busy
    threading.Thread(target=target, daemon=True).start()

def calculate_scalping_indicators(df):
    """Computes fast MACD and ATR for high-frequency momentum tracking."""
    df = df.copy()
    for col in ('high', 'low', 'close'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    close = df['close']

    # 1. Fast MACD Setup (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['hist'] = df['macd'] - df['signal']

    # 2. Average True Range (ATR) for volatility threshold tracking
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - close.shift(1)).abs()
    tr3 = (df['low'] - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    return df


def get_position_info():
    try:
        positions = ib.positions()
        pos = next((p for p in positions if p.contract.symbol == SYMBOL), None)
        if pos:
            return int(pos.position), float(pos.avgCost)
        return 0, 0.0
    except:
        return 0, 0.0


def submit_order(action, qty, limit_price=None, note=''):
    qty_held, _ = get_position_info()
    if action == 'SELL':
        state['last_exit_reason'] = note
        state['last_exit_price'] = get_position_info()[1]

    if action == 'SELL' and qty_held <= 0:
        return

    # Using Market Orders for entry/exit to guarantee execution during fast scalps
    order = MarketOrder(action, qty, tif='DAY')
    print(f"📡 Executing {action} {qty} MARKET ({note})")

    state['order_pending'] = True
    state['order_pending_since'] = pd.Timestamp.now()
    trade = ib.placeOrder(contract, order)

    def on_status(trade_):
        status = trade_.orderStatus.status
        if status == 'Filled':
            print(f"✅ ORDER FILLED [{note}]: {action} {qty} @ {trade_.orderStatus.avgFillPrice}")
            speak_alert(action)
            new_qty_held, _ = get_position_info()

            if action == 'BUY':
                state['buy_price'] = trade_.orderStatus.avgFillPrice
                state['peak_pnl_pct'] = 0.0
                state['last_direction'] = 1
                state['sold_target1'] = False
                state['sold_target2'] = False
            elif new_qty_held == 0:
                state['buy_price'] = None
                state['peak_pnl_pct'] = 0.0
                state['last_direction'] = 0
                state['sold_target1'] = False
                state['sold_target2'] = False

            state['order_pending'] = False
            state['order_pending_since'] = None
        elif status in ('Cancelled', 'ApiCancelled', 'Inactive'):
            state['order_pending'] = False
            state['order_pending_since'] = None

    trade.statusEvent += on_status
    ib.sleep(0.5)


def process_latest_data(bars, ticker):
    now_price = ticker.last if ticker.last > 0 else ticker.close
    try:
        df = util.df(bars)
        if df.empty: return

        df = df.set_index('date')
        df.index = pd.to_datetime(df.index, utc=True)
        df = calculate_scalping_indicators(df)
        state['df'] = df

        qty_held, avg_cost = get_position_info()
        effective_buy_price = state['buy_price'] if state['buy_price'] is not None else avg_cost

        if len(df) < 30:
            print("Warming up indicators...")
            return

        curr = df.iloc[-2]
        prev = df.iloc[-3]
        pnl_pct = 0.0

        # --- Live PnL Log ---
        if qty_held != 0:
            direction = 1 if qty_held > 0 else -1
            pnl_pct = direction * ((now_price - effective_buy_price) / effective_buy_price)
            pnl_dollars = direction * (now_price - effective_buy_price) * abs(qty_held)
            print(f"⏳ ACTIVE | Entry: ${effective_buy_price:.2f} | Pr: ${now_price:.2f} | PnL: {pnl_pct * 100:.2f}%")
        else:
            print(f"👀 FLAT | Price: ${now_price:.2f} | MACD Hist: {curr['hist']:.4f} | ATR: {curr['atr']:.2f}")

        # --- Circuit Breaker ---
        if state['order_pending'] and state['order_pending_since']:
            if (pd.Timestamp.now() - state['order_pending_since']).total_seconds() > 15:
                state['order_pending'] = False
                state['order_pending_since'] = None

        if state['order_pending']:
            return

        # --- ENTRY CONTEXT (FLAT POSITION) ---
        if qty_held == 0:
            # Trade trigger: MACD Histogram flips from negative to positive
            macd_flip_positive = (prev['hist'] <= 0) and (curr['hist'] > 0)

            # Volatility filter: Ensure asset isn't flatlining in zero volume
            has_volume_volatility = curr['atr'] > 0.05

            if macd_flip_positive and has_volume_volatility:
                print("🚀 MACRO MOMENTUM CROSSOVER — Firing Scalp Entry.")
                submit_order('BUY', TRADE_QTY, note='Velocity_Cross_Entry')
            return

        # --- EXIT CONTEXT (ACTIVE POSITION) ---
        # 1. Hard Stop Loss
        if pnl_pct <= HARD_STOP_LOSS_PCT:
            print("🛑 HARD STOP TRIGGERED.")
            submit_order('SELL', abs(qty_held), note='hard_stop')
            return

        # 2. Trailing Stop
        state['peak_pnl_pct'] = max(state.get('peak_pnl_pct', 0.0), pnl_pct)
        current_peak = state['peak_pnl_pct']

        if current_peak >= TRAIL_ACTIVATE:
            trail_level = current_peak - TRAIL_PULLBACK
            if pnl_pct <= trail_level:
                print(f"🔒 TRAIL TRIGGERED | Peak: {current_peak * 100:.2f}% | Floor: {trail_level * 100:.2f}%")
                submit_order('SELL', abs(qty_held), note='trailing')
                return

        # 3. Partial Target 1
        if not state.get('sold_target1', False) and pnl_pct >= TARGET1_PCT:
            print("🎯 TARGET 1 HIT — Scaling Out 50%.")
            submit_order('SELL', abs(qty_held) // 2, note='target1')
            state['sold_target1'] = True
            return

        # 4. Final Target 2
        if state.get('sold_target1', False) and not state.get('sold_target2', False) and pnl_pct >= TARGET2_PCT:
            print("💰 TARGET 2 HIT — Closing Position.")
            submit_order('SELL', abs(qty_held), note='target2')
            state['sold_target2'] = True
            return

    except Exception as e:
        print(f"Error Processing Data Stream: {e}")


if __name__ == '__main__':
    util.logToConsole()
    ib = IB()
    print("Connecting to Interactive Brokers...")
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
    ib.reqMarketDataType(MARKET_DATA_TYPE)

    contract = Stock(SYMBOL, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    print(f"High-Volume Velocity Scalper Initialized for {SYMBOL}...")
    ticker = ib.reqMktData(contract)
    ib.sleep(2)

    try:
        while True:
            bars = ib.reqHistoricalData(
                contract, endDateTime='', durationStr=HISTORY_DURATION,
                barSizeSetting=BAR_SIZE, whatToShow='TRADES', useRTH=True, keepUpToDate=False
            )
            if bars:
                process_latest_data(bars, ticker)
            ib.sleep(2)  # Speed up check loops to 2 seconds for high frequency updates

    except KeyboardInterrupt:
        print("Shutting down bot safely...")
    finally:
        ib.disconnect()
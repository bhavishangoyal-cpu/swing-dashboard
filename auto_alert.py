from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import winsound

# --- Core Settings ---
SYMBOL = 'TQQQ'
BAR_SIZE = '5 mins'  # Switched to a standard swing/intraday compression timeframe
HISTORY_DURATION = '2 D'  # Duration to calculate short-term indicators cleanly
TRADE_QTY = 100

IB_HOST = '127.0.0.1'
IB_PORT = 7497
CLIENT_ID = 3
MARKET_DATA_TYPE = 3  # Delayed = 3, Live = 1

# --- Risk Management Settings ---
HARD_STOP_LOSS_PCT = -0.0010  # Reduced to 1.0% to support a wider intraday swing structure
TARGET1_PCT = 0.005  # Sell 50% at +1.5%
TARGET2_PCT = 0.010  # Sell remaining at +3.0%
TRAIL_ACTIVATE = 0.003  # Trailing activates at +0.8% profit
TRAIL_PULLBACK = 0.001  # Trail distance

ib = None
contract = None

state = {
    'df': pd.DataFrame(),
    'order_pending': False,
    'order_pending_since': None,   # <-- new
    'buy_price': None,
    'peak_pnl_pct': 0.0,
    'entry_time': None,
    'sold_target1': False,
    'sold_target2': False,
    'last_direction': 0,
    'last_exit_reason': None,
    'last_exit_price': 0.0
}

def calculate_ttm_squeeze(df, period=20, bb_mult=2.0, kc_mult=1.5):
    """Calculates John Carter's TTM Squeeze parameters using standard retail math."""
    df = df.copy()
    for col in ('high', 'low', 'close'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    close = df['close']
    high = df['high']
    low = df['low']

    # 1. Bollinger Bands
    sma = close.rolling(window=period).mean()
    std_dev = close.rolling(window=period).std()
    df['bb_upper'] = sma + (bb_mult * std_dev)
    df['bb_lower'] = sma - (bb_mult * std_dev)

    # 2. Keltner Channels (using Wilder's style ATR calculation)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    df['kc_upper'] = sma + (kc_mult * atr)
    df['kc_lower'] = sma - (kc_mult * atr)

    # 3. Squeeze Condition: Red dot if Bollinger Bands are inside Keltner Channels
    df['squeeze_on'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])

    # 4. Momentum Histogram using Linear Regression of Price against High/Low/SMA midpoint
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    donchian_midpoint = (highest_high + lowest_low) / 2
    df_midpoint = (donchian_midpoint + sma) / 2

    fit_y = close - df_midpoint
    # Basic linear regression approximation for standard indicator histogram
    x = np.arange(period)

    def get_slope(y_series):
        if len(y_series) < period or y_series.isnull().any():
            return 0.0
        return np.polyfit(x, y_series, 1)[0]

    df['histogram'] = fit_y.rolling(window=period).apply(get_slope, raw=False)
    return df


def check_dr_paul_macro_filter():
    """Fetches Daily data to implement Dr. David Paul's 200-day SMA Bull Regime rule."""
    try:
        daily_bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='300 D',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            keepUpToDate=False
        )
        if not daily_bars:
            return False

        df_daily = util.df(daily_bars)
        close_series = pd.to_numeric(df_daily['close'], errors='coerce')

        if len(close_series) < 200:
            print("⚠️ Insufficient daily history for Dr. Paul's 200 SMA Filter.")
            return False

        sma_200 = close_series.rolling(window=200).mean().iloc[-1]
        current_daily_close = close_series.iloc[-1]

        is_bull_regime = current_daily_close > sma_200
        print(
            f"📋 Macro Check | Current Price: {current_daily_close:.2f} | 200 Day SMA: {sma_200:.2f} | Bull Regime: {is_bull_regime}")
        return is_bull_regime
    except Exception as e:
        print(f"❌ Error computing Macro Filter: {e}")
        return False


def get_position_info():
    try:
        positions = ib.positions()
        pos = next((p for p in positions if p.contract.symbol == SYMBOL), None)
        if pos:
            return int(pos.position), float(pos.avgCost)
        return 0, 0.0
    except:
        return 0, 0.0


def play_trade_alert(side):
    if side.upper() == 'BUY':
        winsound.Beep(1200, 400)
    else:
        winsound.Beep(600, 400)


def submit_order(action, qty, limit_price=None, note=''):
    qty_held, _ = get_position_info()

    if action == 'SELL':
        state['last_exit_reason'] = note
        state['last_exit_price'] = get_position_info()[1]

    if action == 'SELL' and qty_held <= 0:
        print("🛑 SHORTING BLOCKED")
        return

    if limit_price is not None:
        order = LimitOrder(action, qty, round(limit_price, 2), tif='DAY')
        print(f"📡 Sending {action} {qty} LIMIT @ {limit_price:.2f} ({note})")
    else:
        order = MarketOrder(action, qty, tif='DAY')
        print(f"📡 Sending {action} {qty} MARKET ({note})")

    state['order_pending'] = True
    state['order_pending_since'] = pd.Timestamp.now()
    trade = ib.placeOrder(contract, order)

    def on_status(trade_):
        status = trade_.orderStatus.status
        if status == 'Filled':
            print(f"✅ ORDER FILLED [{note}]: {action} {qty} @ {trade_.orderStatus.avgFillPrice}")
            new_qty_held, _ = get_position_info()

            if note == 'TTM_Squeeze_Macro_Confirmed':
                state['buy_price'] = trade_.orderStatus.avgFillPrice
                state['peak_pnl_pct'] = 0.0
                state['last_direction'] = 1 if new_qty_held > 0 else -1
                state['sold_target1'] = False
                state['sold_target2'] = False
            elif new_qty_held == 0:
                state['buy_price'] = None
                state['peak_pnl_pct'] = 0.0
                state['last_direction'] = 0
                state['sold_target1'] = False
                state['sold_target2'] = False

            state['order_pending'] = False
            state['order_pending_since'] = None  # <-- new
        elif status in ('Cancelled', 'ApiCancelled', 'Inactive'):
            state['order_pending'] = False
            state['order_pending_since'] = None  # <-- new

    trade.statusEvent += on_status
    ib.sleep(1.0)

    # --- Safety net (new) ---
    if trade.orderStatus.status in ('Filled', 'Cancelled', 'ApiCancelled', 'Inactive'):
        state['order_pending'] = False
        state['order_pending_since'] = None  # <-- new

def process_latest_data(bars, ticker):
    now_price = ticker.last if ticker.last > 0 else ticker.close
    try:
        df = util.df(bars)
        if df.empty: return

        df = df.set_index('date')
        df.index = pd.to_datetime(df.index, utc=True)
        df = calculate_ttm_squeeze(df)
        state['df'] = df

        qty_held, avg_cost = get_position_info()
        effective_buy_price = state['buy_price'] if state['buy_price'] is not None else avg_cost

        if len(df) < 25:
            print("Warming up indicators...")
            return

        curr = df.iloc[-2]
        prev = df.iloc[-3]

        pnl_pct = 0.0  # stays 0 while flat

        # --- Live PnL Log ---
        if qty_held != 0:
            direction = 1 if qty_held > 0 else -1
            pnl_pct = direction * ((now_price - effective_buy_price) / effective_buy_price)
            pnl_dollars = direction * (now_price - effective_buy_price) * abs(qty_held)
            pnl_string = (f"PROFIT: +${pnl_dollars:.2f} (+{pnl_pct * 100:.2f}%)"
                          if pnl_dollars >= 0
                          else f"LOSS: -${abs(pnl_dollars):.2f} ({pnl_pct * 100:.2f}%)")
            print(f"{pd.Timestamp.now().strftime('%H:%M:%S')} | Entry: ${effective_buy_price:.2f} | "
                  f"Pr: ${now_price:.2f} | Qty: {qty_held} | {pnl_string} | Pending: {state['order_pending']}")
        else:
            print(f"{pd.Timestamp.now().strftime('%H:%M:%S')} | Pr: ${now_price:.2f} | Qty: 0 | FLAT | "
                  f"SQZ: {'RED' if curr['squeeze_on'] else 'GRN'} (Hist: {curr['histogram']:.4f})")

            # --- Circuit breaker: force-clear a stuck pending flag ---
            if state['order_pending'] and state['order_pending_since']:
                stuck_seconds = (pd.Timestamp.now() - state['order_pending_since']).total_seconds()
                if stuck_seconds > 30:
                    print(f"⚠️ order_pending stuck for {stuck_seconds:.0f}s — force clearing.")
                    state['order_pending'] = False
                    state['order_pending_since'] = None

            if state['order_pending']:
                return

        # --- ENTRY CONTEXT (FLAT POSITION) ---
        if qty_held == 0:
            state['peak_pnl_pct'] = 0.0
            state['last_direction'] = 0

            squeeze_fired = (prev['squeeze_on'] == True) and (curr['squeeze_on'] == False)
            positive_momentum = curr['histogram'] > 0
            momentum_accelerating = curr['histogram'] > prev['histogram']

            if squeeze_fired and positive_momentum and momentum_accelerating:
                print("🎯 Squeeze fired visually on intraday chart. Evaluating Macro Regime filter...")
                if check_dr_paul_macro_filter():
                    print("🚀 MACRO BULL REGIME VALIDATED — Entering Position.")
                    submit_order('BUY', TRADE_QTY, note='TTM_Squeeze_Macro_Confirmed')
                else:
                    print("🛑 Entry signal blocked: Asset trading underneath the Daily 200 SMA.")
            return

        # --- EXIT CONTEXT (ACTIVE POSITION) ---
        current_dir = 1 if qty_held > 0 else -1
        if state.get('last_direction', 0) != current_dir:
            state['peak_pnl_pct'] = 0.0
        state['last_direction'] = current_dir

        buffer = 0.05  # Slippage cushion for order execution

        # 1. Hard Stop Loss
        if pnl_pct <= HARD_STOP_LOSS_PCT:
            print("🛑 HARD STOP LOSS TRIGGERED | Exiting Entire Position.")
            exit_action = 'SELL' if qty_held > 0 else 'BUY'
            submit_order(exit_action, abs(qty_held),
                         limit_price=(now_price - buffer if qty_held > 0 else now_price + buffer),
                         note='hard_stop')
            return

        # 2. Trailing Stop
        state['peak_pnl_pct'] = max(state.get('peak_pnl_pct', 0.0), pnl_pct)
        current_peak = state['peak_pnl_pct']

        if current_peak >= TRAIL_ACTIVATE:
            trail_level = current_peak - TRAIL_PULLBACK
            if pnl_pct <= trail_level:
                print(f"🔒 TRAILING STOP TRIGGERED | Peak: {current_peak * 100:.2f}% | "
                      f"Floor: {trail_level * 100:.2f}% | Current: {pnl_pct * 100:.2f}%")
                exit_action = 'SELL' if qty_held > 0 else 'BUY'
                submit_order(exit_action, abs(qty_held), note='trailing')
                state['peak_pnl_pct'] = 0.0
                state['last_direction'] = 0
                return

        # 3. Partial Target 1
        if not state.get('sold_target1', False) and pnl_pct >= TARGET1_PCT:
            sell_qty = abs(qty_held) // 2
            print(f"🎯 TARGET 1 HIT (+{pnl_pct * 100:.2f}%) - Scaling Out 50%.")
            exit_action = 'SELL' if qty_held > 0 else 'BUY'
            submit_order(exit_action, sell_qty,
                         limit_price=(now_price - buffer if qty_held > 0 else now_price + buffer),
                         note='target1')
            state['sold_target1'] = True
            return

        # 4. Final Target 2
        if state.get('sold_target1', False) and not state.get('sold_target2', False) and pnl_pct >= TARGET2_PCT:
            print(f"💰 TARGET 2 HIT (+{pnl_pct * 100:.2f}%) - Fully closing remaining units.")
            exit_action = 'SELL' if qty_held > 0 else 'BUY'
            submit_order(exit_action, abs(qty_held),
                         limit_price=(now_price - buffer if qty_held > 0 else now_price + buffer),
                         note='target2')
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

    print(f"Bot successfully structured around TTM Squeeze + 200-Day SMA Macro Filters...")
    ticker = ib.reqMktData(contract)
    ib.sleep(2)

    try:
        while True:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=HISTORY_DURATION,
                barSizeSetting=BAR_SIZE,
                whatToShow='TRADES',
                useRTH=True,
                keepUpToDate=False
            )

            if bars:
                process_latest_data(bars, ticker)

            ib.sleep(5)  # Polling interval slowed down slightly to reflect a cleaner swing chart execution loop

    except KeyboardInterrupt:
        print("Shutting down bot safely...")
    finally:
        ib.disconnect()
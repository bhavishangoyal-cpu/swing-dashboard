from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
import pandas as pd
import numpy as np
from datetime import datetime, timezone, time as dt_time
import winsound

# ==========================================
# CONFIG — same percentages for every symbol, nothing stock-specific
# ==========================================
SYMBOLS = ['NVDA', 'TSLA', 'GOOG']
CAPITAL = 10000.0
BAR_SIZE = '2 mins'
HISTORY_DURATION = '2 D'

IB_HOST = '127.0.0.1'
IB_PORT = 7497
CLIENT_ID = 3
MARKET_DATA_TYPE = 3  # 1 = Live (required before real trading), 3 = Delayed

# --- Percentage-based thresholds (identical across all symbols) ---
BREAKEVEN_TRIGGER_PCT = 0.001    # +0.10% peak profit triggers breakeven lock
BREAKEVEN_BUFFER_PCT = 0.0002    # +0.02% buffer above entry
TRAILING_DISTANCE_PCT = 0.002    # 0.20% trailing distance from peak
QUICK_SCALP_PCT = 0.003          # 0.30% quick profit target
MIN_ACCELERATION = 0.01          # minimum histogram acceleration to qualify as a candidate

# No hard stop-loss — per explicit instruction. A trade that never reaches
# BREAKEVEN_TRIGGER_PCT has no bounded downside. This is a known, accepted risk.

WINDOW_START_PT = dt_time(6, 30)
WINDOW_END_PT = dt_time(12, 0)
LOOP_SLEEP_SECONDS = 10  # compromise between responsiveness and IBKR historical-data pacing limits

ib = None
contracts = {}

state = {
    'held_symbol': None,
    'buy_price': None,
    'qty_held': 0,
    'highest_price_seen': 0.0,
    'breakeven_locked': False,
    'resting_trade': None,     # the live trailing-floor LIMIT order, if any
    'submitting': False,       # true only briefly while an order is actively being sent
    'entry_time': None,
    'last_sell_time': None,
    'daily_realized_pnl': 0.0,   # sum of closed-trade P&L so far today
    'daily_trade_count': 0,      # number of trades closed today
    'daily_date': None,          # tracks which trading day the counters belong to
}


def reset_daily_counters_if_new_day():
    """Resets the running daily P&L at the start of each new trading day (PT)."""
    today_pt = pd.Timestamp.now(tz='America/Los_Angeles').date()
    if state['daily_date'] != today_pt:
        if state['daily_date'] is not None:
            print(f"📅 New trading day detected — resetting daily P&L counter "
                  f"(yesterday ended at ${state['daily_realized_pnl']:.2f}).")
        state['daily_date'] = today_pt
        state['daily_realized_pnl'] = 0.0
        state['daily_trade_count'] = 0


def print_daily_pnl(tickers):
    """Prints running total P&L for the day — realized so far, plus unrealized if
    currently holding a position. Called every loop cycle regardless of state."""
    unrealized = 0.0
    if state['held_symbol'] is not None and state['buy_price'] is not None:
        symbol = state['held_symbol']
        now_price = tickers[symbol].last if tickers[symbol].last > 0 else tickers[symbol].close
        unrealized = (now_price - state['buy_price']) * state['qty_held']

    day_total = state['daily_realized_pnl'] + unrealized
    sign = "+" if day_total >= 0 else ""
    print(f"💰 TODAY | Realized: ${state['daily_realized_pnl']:.2f} "
          f"({state['daily_trade_count']} trades) | Unrealized: ${unrealized:.2f} | "
          f"TOTAL: {sign}${day_total:.2f}")


def calculate_ttm_squeeze(df, period=20, bb_mult=2.0, kc_mult=1.5):
    df = df.copy()
    for col in ('high', 'low', 'close'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    close, high, low = df['close'], df['high'], df['low']

    sma = close.rolling(window=period).mean()
    std_dev = close.rolling(window=period).std()
    df['bb_upper'] = sma + (bb_mult * std_dev)
    df['bb_lower'] = sma - (bb_mult * std_dev)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    df['kc_upper'] = sma + (kc_mult * atr)
    df['kc_lower'] = sma - (kc_mult * atr)
    df['squeeze_on'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])

    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    donchian_mid = (highest_high + lowest_low) / 2
    mid = (donchian_mid + sma) / 2
    fit_y = close - mid
    x = np.arange(period)

    def get_slope(y):
        if len(y) < period or y.isnull().any():
            return 0.0
        return np.polyfit(x, y, 1)[0]

    df['histogram'] = fit_y.rolling(window=period).apply(get_slope, raw=False)
    return df


def in_trading_window():
    now_pt = pd.Timestamp.now(tz='America/Los_Angeles').time()
    return WINDOW_START_PT <= now_pt < WINDOW_END_PT


def play_trade_alert(side):
    try:
        winsound.Beep(1200, 400) if side.upper() == 'BUY' else winsound.Beep(600, 400)
    except Exception:
        pass


def cancel_resting_order():
    """Cancels the active resting floor order, if one exists, and waits for confirmation."""
    trade = state['resting_trade']
    if trade and trade.isActive():
        print(f"🔄 Canceling resting floor order for {state['held_symbol']}...")
        ib.cancelOrder(trade.order)
        for _ in range(50):  # bounded wait, ~5s max
            if not trade.isActive():
                break
            ib.sleep(0.1)
    state['resting_trade'] = None


def submit_order(symbol, action, qty, limit_price=None, note=''):
    """For BUY entries and reactive SELL exits (quick-scalp, momentum-rollover).
    NOT used for the resting floor order — that's placed directly in manage_open_position
    so it doesn't block the main loop while it sits open waiting to be hit."""
    if qty <= 0:
        return None

    contract = contracts[symbol]
    if limit_price is not None:
        order = LimitOrder(action, qty, round(limit_price, 2), tif='DAY')
        print(f"📡 [{symbol}] Sending {action} {qty} LIMIT @ {limit_price:.2f} ({note})")
    else:
        order = MarketOrder(action, qty, tif='DAY')
        print(f"📡 [{symbol}] Sending {action} {qty} MARKET ({note})")

    state['submitting'] = True
    trade = ib.placeOrder(contract, order)

    def on_status(trade_):
        status = trade_.orderStatus.status
        if status == 'Filled':
            fill_price = trade_.orderStatus.avgFillPrice
            print(f"✅ [{symbol}] FILLED [{note}]: {action} {qty} @ {fill_price}")
            play_trade_alert(action)
            if action == 'BUY':
                state['held_symbol'] = symbol
                state['buy_price'] = fill_price
                state['qty_held'] = qty
                state['highest_price_seen'] = fill_price
                state['breakeven_locked'] = False
                state['resting_trade'] = None
                state['entry_time'] = datetime.now(timezone.utc)
            elif action == 'SELL':
                trade_pnl = (fill_price - state['buy_price']) * qty
                state['daily_realized_pnl'] += trade_pnl
                state['daily_trade_count'] += 1
                print(f"📊 [{symbol}] Trade P&L: ${trade_pnl:.2f} | Running today: ${state['daily_realized_pnl']:.2f}")
                state['held_symbol'] = None
                state['buy_price'] = None
                state['qty_held'] = 0
                state['highest_price_seen'] = 0.0
                state['breakeven_locked'] = False
                state['resting_trade'] = None
                state['last_sell_time'] = pd.Timestamp.now(tz='UTC')
            state['submitting'] = False
        elif status in ('Cancelled', 'ApiCancelled', 'Inactive'):
            state['submitting'] = False

    trade.statusEvent += on_status
    ib.sleep(1.0)
    return trade


def place_or_update_resting_floor(symbol, qty_held, floor_price):
    """Places the initial resting LIMIT SELL at the floor, or raises it if the floor moved up.
    Does NOT touch state['submitting'] — this order is meant to sit open, and must never
    block the main loop from continuing to check other exit conditions."""
    trade = state['resting_trade']

    if trade is None:
        print(f"🔒 [{symbol}] Placing resting LIMIT floor @ ${floor_price:.2f}")
        order = LimitOrder('SELL', qty_held, floor_price, tif='DAY')
        new_trade = ib.placeOrder(contracts[symbol], order)

        def on_status(trade_):
            if trade_.orderStatus.status == 'Filled':
                fill_price = trade_.orderStatus.avgFillPrice
                entry_price = state['buy_price']  # read before we reset it below
                trade_pnl = (fill_price - entry_price) * qty_held
                state['daily_realized_pnl'] += trade_pnl
                state['daily_trade_count'] += 1
                print(f"✅ [{symbol}] Resting floor FILLED @ {fill_price}")
                print(f"📊 [{symbol}] Trade P&L: ${trade_pnl:.2f} | Running today: ${state['daily_realized_pnl']:.2f}")
                play_trade_alert('SELL')
                state['held_symbol'] = None
                state['buy_price'] = None
                state['qty_held'] = 0
                state['highest_price_seen'] = 0.0
                state['breakeven_locked'] = False
                state['resting_trade'] = None
                state['last_sell_time'] = pd.Timestamp.now(tz='UTC')

        new_trade.statusEvent += on_status
        state['resting_trade'] = new_trade
        return

    if trade.isActive() and floor_price > trade.order.lmtPrice:
        print(f"📈 [{symbol}] Floor raised: ${trade.order.lmtPrice:.2f} → ${floor_price:.2f}")
        trade.order.lmtPrice = floor_price
        ib.placeOrder(trade.contract, trade.order)  # modifies the existing working order in place


def scan_for_entry(tickers):
    """Scans all symbols, ranks qualifying setups by histogram acceleration, buys the strongest."""
    if state['held_symbol'] is not None or state['submitting']:
        return

    if state['last_sell_time'] is not None:
        elapsed_min = (pd.Timestamp.now(tz='UTC') - state['last_sell_time']).total_seconds() / 60.0
        if elapsed_min < 1.0:
            print(f"⏸ Cooldown active ({1.0 - elapsed_min:.1f} min left)...")
            return

    candidates = []
    for symbol in SYMBOLS:
        try:
            ib.sleep(0.5)
            bars = ib.reqHistoricalData(
                contracts[symbol], endDateTime='', durationStr=HISTORY_DURATION,
                barSizeSetting=BAR_SIZE, whatToShow='TRADES', useRTH=False, keepUpToDate=False
            )
            if not bars:
                continue
            df = util.df(bars).set_index('date')
            df.index = pd.to_datetime(df.index, utc=True)
            df = calculate_ttm_squeeze(df)
            if len(df) < 25:
                continue

            curr = df.iloc[-2]
            prev = df.iloc[-3]
            now_price = tickers[symbol].last if tickers[symbol].last > 0 else tickers[symbol].close
            acceleration = curr['histogram'] - prev['histogram']

            squeeze_ok = (curr['squeeze_on'] == False)
            positive_momentum = curr['histogram'] > 0

            print(f"👀 [{symbol}] Pr: ${now_price:.2f} | Hist: {curr['histogram']:.4f} | Accel: {acceleration:.4f}")

            if squeeze_ok and positive_momentum and acceleration > MIN_ACCELERATION:
                candidates.append({'symbol': symbol, 'price': now_price, 'accel': acceleration})
        except Exception as e:
            print(f"⚠️ [{symbol}] Error scanning: {e}")

    if not candidates:
        return

    candidates.sort(key=lambda x: x['accel'], reverse=True)
    best = candidates[0]
    print(f"🏆 Strongest setup: {best['symbol']} (Accel: {best['accel']:.4f})")

    qty_to_buy = int(CAPITAL // best['price'])
    if qty_to_buy < 1:
        print(f"⚠️ [{best['symbol']}] Capital too small to buy 1 share at ${best['price']:.2f}.")
        return

    print(f"🚀 [{best['symbol']}] SIGNAL FIRED — Entering {qty_to_buy} shares.")
    submit_order(best['symbol'], 'BUY', qty_to_buy, note='Strongest_Squeeze_Entry')


def manage_open_position(tickers):
    """Checks quick-scalp and momentum-rollover FIRST (fast exits), then manages the
    resting trailing floor. Runs every loop cycle regardless of whether a floor order
    is currently resting — that's the whole point of the fix."""
    symbol = state['held_symbol']
    now_price = tickers[symbol].last if tickers[symbol].last > 0 else tickers[symbol].close
    buy_price = state['buy_price']
    qty_held = state['qty_held']

    state['highest_price_seen'] = max(state['highest_price_seen'], now_price)
    peak_price = state['highest_price_seen']
    current_return = (now_price - buy_price) / buy_price
    peak_return = (peak_price - buy_price) / buy_price
    dollar_pnl = qty_held * (now_price - buy_price)

    print(f"⏳ [{symbol}] Entry: ${buy_price:.2f} | Now: ${now_price:.2f} | Peak: ${peak_price:.2f} | "
          f"PnL: {current_return*100:.2f}% (${dollar_pnl:.2f}) | Locked: {state['breakeven_locked']}")

    if state['submitting']:
        return  # a reactive order (entry/exit) is actively being sent — wait for it, don't double-submit

    # 1. Quick scalp — fastest exit, checked first
    if current_return >= QUICK_SCALP_PCT:
        print(f"⚡ [{symbol}] Quick scalp hit ({current_return*100:.2f}%). Exiting.")
        cancel_resting_order()
        submit_order(symbol, 'SELL', qty_held, note='quick_scalp')
        return

    # 2. Momentum rollover
    try:
        bars = ib.reqHistoricalData(
            contracts[symbol], endDateTime='', durationStr=HISTORY_DURATION,
            barSizeSetting=BAR_SIZE, whatToShow='TRADES', useRTH=False, keepUpToDate=False
        )
        if bars:
            df = util.df(bars).set_index('date')
            df.index = pd.to_datetime(df.index, utc=True)
            df = calculate_ttm_squeeze(df)
            if len(df) >= 25:
                curr = df.iloc[-2]
                prev = df.iloc[-3]
                if current_return > 0 and curr['histogram'] < prev['histogram']:
                    print(f"⚠️ [{symbol}] MOMENTUM ROLLOVER — exiting to protect gain.")
                    cancel_resting_order()
                    submit_order(symbol, 'SELL', qty_held, note='momentum_rollover')
                    return
    except Exception as e:
        print(f"⚠️ [{symbol}] Error checking momentum: {e}")

    # 3. Trailing resting floor
    if not state['breakeven_locked'] and peak_return >= BREAKEVEN_TRIGGER_PCT:
        state['breakeven_locked'] = True
        print(f"🔐 [{symbol}] Breakeven lock activated at {peak_return*100:.2f}% peak.")

    if state['breakeven_locked']:
        breakeven_floor = buy_price * (1 + BREAKEVEN_BUFFER_PCT)
        trailing_floor = peak_price * (1 - TRAILING_DISTANCE_PCT)
        actual_floor = round(max(breakeven_floor, trailing_floor), 2)
        place_or_update_resting_floor(symbol, qty_held, actual_floor)


if __name__ == '__main__':
    util.logToConsole(level=30)
    ib = IB()
    print("Connecting to Interactive Brokers...")
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)

    live_positions = ib.positions()
    for pos in live_positions:
        if pos.contract.symbol in SYMBOLS and pos.position != 0:
            state['held_symbol'] = pos.contract.symbol
            state['qty_held'] = pos.position
            state['buy_price'] = pos.avgCost
            state['highest_price_seen'] = pos.avgCost
            print(f"🔄 SYNCED: Found {pos.position} shares of {pos.contract.symbol} in account.")
            break

    ib.reqMarketDataType(MARKET_DATA_TYPE)

    tickers = {}
    for sym in SYMBOLS:
        c = Stock(sym, 'SMART', 'USD')
        ib.qualifyContracts(c)
        contracts[sym] = c
        tickers[sym] = ib.reqMktData(c)
    ib.sleep(2)

    print(f"\n🚀 Live | Symbols: {SYMBOLS} | ${CAPITAL:.0f} pool | No hard stop")
    print(f"📊 Scalp: {QUICK_SCALP_PCT*100:.2f}% | Lock: {BREAKEVEN_TRIGGER_PCT*100:.2f}% | "
          f"Trail: {TRAILING_DISTANCE_PCT*100:.2f}%")
    print(f"⏰ Window: {WINDOW_START_PT} - {WINDOW_END_PT} PT\n")

    try:
        while True:
            reset_daily_counters_if_new_day()

            if state['held_symbol']:
                current_positions = ib.positions()
                found = any(p.contract.symbol == state['held_symbol'] and p.position != 0
                           for p in current_positions)
                if not found:
                    print(f"🔄 SYNC: Position {state['held_symbol']} closed externally. Resetting.")
                    if state['resting_trade'] and state['resting_trade'].isActive():
                        ib.cancelOrder(state['resting_trade'].order)
                    state['held_symbol'] = None
                    state['qty_held'] = 0
                    state['resting_trade'] = None

            print_daily_pnl(tickers)

            now_pt = pd.Timestamp.now(tz='America/Los_Angeles').time()
            is_outside = not in_trading_window()

            if is_outside and state['held_symbol'] is None:
                print(f"⏰ Outside trading window ({now_pt}). Standing by...")
                ib.sleep(30)
                continue

            try:
                if state['held_symbol'] is None:
                    if now_pt < WINDOW_END_PT:
                        scan_for_entry(tickers)
                    else:
                        print("🕒 Past cutoff. No new entries.")
                else:
                    manage_open_position(tickers)
            except Exception as e:
                print(f"⚠️ Error in loop: {e}")

            print("-" * 60)
            ib.sleep(LOOP_SLEEP_SECONDS)

    except KeyboardInterrupt:
        print("Shutting down bot safely...")
        if state['resting_trade'] and state['resting_trade'].isActive():
            print("Canceling active resting order...")
            ib.cancelOrder(state['resting_trade'].order)
    finally:
        ib.disconnect()
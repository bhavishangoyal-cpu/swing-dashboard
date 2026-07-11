from ib_insync import IB, Stock, MarketOrder, util
import pandas as pd
from datetime import datetime, timezone
import winsound
from ib_insync import MarketOrder, LimitOrder

SYMBOL = 'TQQQ'
PERIOD = 14
BAR_SIZE = '15 secs'
HISTORY_DURATION = '7200 S'
TRADE_QTY = 100

IB_HOST = '127.0.0.1'
IB_PORT = 7497
CLIENT_ID = 3
MARKET_DATA_TYPE = 3

HARD_STOP_LOSS_PCT = -0.0010
TARGET1_PCT = 0.0040   # Sell 50% at +0.4%
TARGET2_PCT = 0.0070   # Sell remaining at +0.7%

MIN_ADX_FOR_ENTRY = 20
MIN_DI_PLUS = 25
MIN_DI_SPREAD = 10
MAX_HOLD_SECONDS = 900
TRAIL_ACTIVATE = 0.0005
TRAIL_PULLBACK = 0.0001

ib = None
contract = None

state = {
    'df': pd.DataFrame(),
    'order_pending': False,
    'buy_price': None,
    'peak_pnl_pct': 0.0,
    'entry_time': None,
    'sold_target1': False,
    'sold_target2': False,        # ADDED
    'last_exit_reason': None,     # ADDED
    'last_exit_price': 0.0        # ADDED
}

def wilder_smooth(series, n):
    smoothed = series.copy()
    smoothed.iloc[n-1] = series.iloc[:n].mean()
    for i in range(n, len(series)):
        smoothed.iloc[i] = (smoothed.iloc[i-1] * (n - 1) + series.iloc[i]) / n
    return smoothed

def calculate_indicators(df, period=14):
    df = df.copy()
    for col in ('high', 'low', 'close', 'volume'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = wilder_smooth(tr, period)
    psm = wilder_smooth(plus_dm, period)
    msm = wilder_smooth(minus_dm, period)

    plus_di = 100 * psm / atr.replace(0, 1e-9)
    minus_di = 100 * msm / atr.replace(0, 1e-9)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    adx = wilder_smooth(dx, period)

    df['DI+'] = plus_di
    df['DI-'] = minus_di
    df['ADX'] = adx

    valid_vol = volume.clip(lower=0)
    df['VWAP'] = (close * valid_vol).cumsum() / valid_vol.cumsum().replace(0, 1e-9)

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

def play_trade_alert(side):
    if side.upper() == 'BUY':
        winsound.Beep(1000, 500)
    else:
        winsound.Beep(500, 500)


def submit_order(action, qty, limit_price=None, note=''):
    qty_held, _ = get_position_info()

    # Save exit info if selling
    if action == 'SELL':
        state['last_exit_reason'] = note
        state['last_exit_price'] = get_position_info()[1]  # capture exit price

    if action == 'SELL' and qty_held <= 0:
        print("🛑 SHORTING BLOCKED")
        return

    # Create the order object (Limit OR Market)
    if limit_price is not None:
        order = LimitOrder(action, qty, limit_price, tif='DAY')
        print(f"📡 Sending {action} {qty} LIMIT @ {limit_price:.2f} ({note})")
    else:
        order = MarketOrder(action, qty, tif='DAY')
        print(f"📡 Sending {action} {qty} MARKET ({note})")

    state['order_pending'] = True

    # Place the order using the 'order' object created above
    trade = ib.placeOrder(contract, order)
    print(f"📡 Sending {action} order for {qty} shares...")

    def on_status(trade_):
        status = trade_.orderStatus.status
        if status == 'Filled':
            print(f"✅ ORDER FILLED [{note}]: {action} {qty} @ {trade_.orderStatus.avgFillPrice}")
            play_trade_alert(action)
            # Reset states on BUY
            if action == 'BUY':
                state['buy_price'] = trade_.orderStatus.avgFillPrice
                state['peak_pnl_pct'] = 0.0
                state['entry_time'] = datetime.now(timezone.utc)
                state['sold_target1'] = False
                state['sold_target2'] = False
            state['order_pending'] = False
        elif status in ('Cancelled', 'ApiCancelled', 'Inactive'):
            state['order_pending'] = False

    trade.statusEvent += on_status
    ib.sleep(1.0)
def process_latest_data(bars, ticker):
    now_price = ticker.last if ticker.last > 0 else ticker.close
    try:
        df = util.df(bars)
        if df.empty: return

        df = df.set_index('date')
        df.index = pd.to_datetime(df.index, utc=True)
        df = calculate_indicators(df, PERIOD)
        state['df'] = df


        qty_held, avg_cost = get_position_info()
        effective_buy_price = state['buy_price'] if state['buy_price'] is not None else avg_cost

        if len(df) < 4:
            print("Warming up...")
            return

        curr = df.iloc[-2]
        prev = df.iloc[-3]

        adx = curr['ADX']
        di_plus = curr['DI+']
        di_minus = curr['DI-']
        vwap = curr['VWAP']

        print(f"{pd.Timestamp.now().strftime('%H:%M:%S')} | Pr: {now_price:.2f} | VWAP: {vwap:.2f} | "
              f"ADX: {adx:.1f} | DI+: {di_plus:.1f} | DI-: {di_minus:.1f} | Qty: {qty_held}")

        if state['order_pending']: return
        if state['order_pending']: return

        if qty_held == 0:

            # VWAP filter
            price_above_vwap = curr['close'] > vwap
            vwap_distance = (curr['close'] - vwap) / vwap
            not_extended = vwap_distance < 0.003

            # Momentum
            bullish_now = (
                    di_plus > di_minus
                    and adx >= prev['ADX']
            )

            bullish_prev = prev['DI+'] > prev['DI-']

            strong = (
                    di_plus >= MIN_DI_PLUS
                    and (di_plus - di_minus) >= MIN_DI_SPREAD
            )

            # Candle confirmation
            bullish_candle = curr['close'] > curr['open']

            if (
                    bullish_now
                    and bullish_prev
                    and adx >= MIN_ADX_FOR_ENTRY
                    and strong
                    and price_above_vwap
                    and not_extended
                    and bullish_candle
            ):
                print(
                    f"🎯 ENTRY SIGNAL | "
                    f"ADX:{adx:.1f} "
                    f"DI+:{di_plus:.1f} "
                    f"DI-:{di_minus:.1f} "
                    f"VWAP:{vwap:.2f}"
                )

                submit_order(
                    'BUY',
                    TRADE_QTY,
                    note='VWAP_ADX_momentum'
                )
        elif qty_held > 0:

            pnl_pct = (now_price - effective_buy_price) / effective_buy_price
            state['peak_pnl_pct'] = max(state['peak_pnl_pct'], pnl_pct)

            peak = state['peak_pnl_pct']
            print(
                f"📈 Peak: {peak * 100:.2f}% | "
                f"Current: {pnl_pct * 100:.2f}% | "
                f"Buy: {effective_buy_price:.2f} | "
                f"Price: {now_price:.2f}"
            )
            buffer = 0.03  # Limit order cushion

            # 1. Hard Stop Loss
            if pnl_pct <= HARD_STOP_LOSS_PCT:
                print(
                    f"🛑 HARD STOP LOSS | "
                    f"Current: {pnl_pct * 100:.2f}% "
                    f"Limit: {HARD_STOP_LOSS_PCT * 100:.2f}%"
                )

                submit_order(
                    'SELL',
                    qty_held,
                    limit_price=now_price - buffer,
                    note='hard_stop'
                )

                return

            # 1. Hard Stop (The floor)

            # Momentum failure exit
            if now_price < vwap and di_plus < di_minus:
                print(
                    f"⚠️ VWAP FAILURE EXIT | "
                    f"Price:{now_price:.2f} VWAP:{vwap:.2f} "
                    f"DI+:{di_plus:.1f} DI-:{di_minus:.1f}"
                )

                submit_order(
                    'SELL',
                    qty_held,
                    limit_price=now_price - buffer,
                    note='vwap_failure'
                )

                return
            # 2. Trailing Stop (Always Active - applies to the full or partial position)

            # 0.0005 represents 0.05%

            # Activate trailing only after trade is profitable
            # Trailing Stop
            if peak >= TRAIL_ACTIVATE:

                trail_level = peak - TRAIL_PULLBACK

                if pnl_pct <= trail_level:
                    print(
                        f"🔒 TRAILING EXIT | "
                        f"Peak {peak * 100:.2f}% → "
                        f"Current {pnl_pct * 100:.2f}%"
                    )

                    submit_order(
                        'SELL',
                        qty_held,
                        note='trailing'
                    )

                    return
            # 3. Target 1 (Sell 50%)

            if not state['sold_target1'] and pnl_pct >= TARGET1_PCT:
                sell_qty = qty_held // 2

                print(f"🎯 TARGET 1 (+{pnl_pct * 100:.2f}%) - Selling 50%")

                submit_order('SELL', sell_qty, limit_price=now_price - buffer, note='target1')

                state['sold_target1'] = True

                return

            # 4. Target 2 (Sell remaining)

            # Only triggers if Target 1 is already sold

            if state['sold_target1'] and not state['sold_target2'] and pnl_pct >= TARGET2_PCT:
                print(f"💰 TARGET 2 (+{pnl_pct * 100:.2f}%) - Selling remaining")

                submit_order('SELL', qty_held, limit_price=now_price - buffer, note='target2')

                state['sold_target2'] = True

                return
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    util.logToConsole()
    ib = IB()
    print("Connecting to IBKR...")
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
    ib.reqMarketDataType(MARKET_DATA_TYPE)

    contract = Stock(SYMBOL, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    print("Bot started with Scaled Targets (0.4% & 0.7%)...")
    ticker = ib.reqMktData(contract)
    ib.sleep(2)

    try:

        while True:

            bars = ib.reqHistoricalData(
                contract,
                '',
                HISTORY_DURATION,
                BAR_SIZE,
                'TRADES',
                useRTH=False,
                keepUpToDate=False
            )

            if bars:
                print("Bars loaded:", len(bars))
                process_latest_data(bars, ticker)

            ib.sleep(2)

    except KeyboardInterrupt:
        print("Shutting down...")

    finally:
        ib.disconnect()
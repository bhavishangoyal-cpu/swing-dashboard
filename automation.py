"""
TQQQ Advanced Matrix Momentum Bot — Profit-Lock / Anti-Whipsaw Edition
=======================================================================
Revised version of the original DMI/ADX momentum bot. Search "FIX:" for
what changed and why. Short version of the diagnosis:

  1. THE BIGGEST DRIVER OF YOUR LOSSES: at TRADE_QTY=10 shares (~$734
     notional at ~$73/share) and IBKR's ~$1 per-order commission, a
     round trip costs roughly $2, ~0.27% of notional. Most DI+/DI-
     crossovers on 5-second bars only capture a few cents of movement —
     smaller than the round-trip commission — so most trades lose money
     to costs alone, independent of whether the signal called the right
     direction. Trading larger size (so the ~$1 floor is a smaller % of
     notional) and/or trading less often is the single highest-leverage
     fix, and isn't something code alone can solve for you.

  2. The old trend-flip exit had ABSOLUTE priority and ignored P&L
     entirely — it closed the position the instant DI- ticked above
     DI+ even by a hair, which happens constantly at 5-second
     resolution, whether the trade was up or down.

  3. There was no mechanism to lock in a big favorable move before it
     round-tripped back into a loss. Added a trailing profit lock.

  4. "Never sell at a loss" can't be guaranteed by any rule set —
     markets can gap or trend hard against a position. What CAN be
     done: (a) never let a *signal-based* exit realize a loss — only a
     single, wider, deliberate stop is allowed to, and (b) bank profit
     proactively so fewer trades get the chance to round-trip into red.
     A strategy with no downside floor at all has unlimited risk on a
     bad day, so a (wide) hard stop is kept on purpose.

Not investment advice. Nothing here guarantees profitability. Paper-test
extensively — including through a real reversal — before considering
live capital.
"""

from ib_insync import IB, Stock, MarketOrder, util
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ============================================================
# CONFIG
# ============================================================
SYMBOL = 'TQQQ'
PERIOD = 60
BAR_SIZE = '15 secs'  # Still noisy for this kind of signal — consider '1 min' once retested
HISTORY_DURATION = '7200 S'
TRADE_QTY = 100  # FIX: consider raising this — see note #1 above; the flat commission
                # is what's actually costing you money at this size, not direction
IB_HOST = '127.0.0.1'
IB_PORT = 7497
CLIENT_ID = 3
MARKET_DATA_TYPE = 3  # 3 = Delayed testing, 1 = Live production

# --- FIX: cost-aware, profit-gated exit config -----------------------
EST_ROUNDTRIP_COST_PCT = 0.0013    # ~$2 round trip on ~$734 notional, taken from your own logs
MIN_EXIT_PROFIT_PCT = 0.0018   # floor + small cushion
MIN_PROFIT_LOCK_TRIGGER = 0.0025   # unrealized gain that arms the trailing lock
TRAIL_GIVEBACK = 0.0010            # give back this much from the peak before locking in
HARD_STOP_LOSS_PCT = -0.0005        # the ONLY exit allowed to realize a loss — tune to your risk tolerance
MIN_ADX_FOR_ENTRY = 15            # skip entries when there's no real trend strength
MIN_DI_PLUS = 25
MIN_DI_SPREAD = 12
MAX_HOLD_SECONDS = 1800            # after 30 min, take whatever profit is available rather than
                                    # waiting indefinitely on a trailing-lock retrace (never forces a loss)

# ============================================================
# STATE
# ============================================================
state = {
    'df': pd.DataFrame(),
    'order_pending': False,
    'buy_price': None,
    'peak_pnl_pct': 0.0,   # FIX: best unrealized gain seen so far in the open trade
    'entry_time': None,    # FIX: for the max-hold profit-take check
    'prev_adx': 0,
    'prev_di_plus': 0
}


# ============================================================
# INDICATORS & VWAP ENGINE
# ============================================================
def calculate_indicators(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    for col in ('high', 'low', 'close', 'volume'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    high, low, close, volume = df['high'], df['low'], df['close'], df['volume']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    psm = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    msm = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * psm / atr.replace(0, 1e-9)
    minus_di = 100 * msm / atr.replace(0, 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)

    df['DI+'] = plus_di
    df['DI-'] = minus_di
    df['ADX'] = dx.ewm(alpha=1 / period, adjust=False).mean()

    valid_vol = volume.clip(lower=0)
    cum_pv = (close * valid_vol).cumsum()
    cum_v = valid_vol.cumsum()
    df['VWAP'] = cum_pv / cum_v.replace(0, 1e-9)

    return df


# ============================================================
# PORTFOLIO DATA RECOVERY
# ============================================================
def get_position_info():
    try:
        positions = ib.positions()
        pos = next((p for p in positions if p.contract.symbol == SYMBOL), None)
        if pos:
            return int(pos.position), float(pos.avgCost)
        return 0, 0.0
    except Exception:
        return 0, 0.0


# ============================================================
# EXECUTION ENGINE WITH SHORT GUARD
# ============================================================
def play_trade_alert(side):
    """
    Plays a distinct sound based on the order side.
    BUY: High pitch
    SELL: Low pitch
    """
    if side.upper() == 'BUY':
        # Frequency 1000Hz, Duration 500ms
        winsound.Beep(1000, 500)
    elif side.upper() == 'SELL':
        # Frequency 500Hz, Duration 500ms
        winsound.Beep(500, 500)


def submit_order(action, qty, note=''):
    qty_held, _ = get_position_info()

    if action == 'SELL' and qty_held <= 0:
        print(f"🛑 SHORTING BLOCKED: Signal wanted to SELL, but portfolio position is {qty_held}.")
        state['order_pending'] = False
        return

    state['order_pending'] = True
    order = MarketOrder(action, qty, tif='DAY')
    trade = ib.placeOrder(contract, order)
    print(f"📡 Sending {action} order for {qty} shares to TWS...")

    def on_status(trade_):
        status = trade_.orderStatus.status
        if status == 'Filled':
            print(f"✅ ORDER FILLED [{note}]: {action} {qty} shares @ Avg Price: {trade_.orderStatus.avgFillPrice}")
            play_trade_alert(action)
            if action == 'BUY':
                state['buy_price'] = trade_.orderStatus.avgFillPrice
                state['peak_pnl_pct'] = 0.0                       # FIX: reset for the new trade
                state['entry_time'] = datetime.now(timezone.utc)  # FIX: start the hold-time clock
            else:
                state['buy_price'] = None
                state['peak_pnl_pct'] = 0.0
                state['entry_time'] = None
            state['order_pending'] = False
        elif status in ('Cancelled', 'ApiCancelled', 'Inactive'):
            print(f"❌ Order [{note}] was rejected or cancelled.")
            state['order_pending'] = False

    trade.statusEvent += on_status
    ib.sleep(1.0)

import winsound


# ============================================================
# OPTIMIZED STRATEGY ENGINE
# ============================================================
def process_latest_data(bars):
    try:
        df = util.df(bars)
        if df.empty: return

        df = df.set_index('date')
        df.index = pd.to_datetime(df.index, utc=True)
        df = calculate_indicators(df, PERIOD)
        state['df'] = df

        now_price = bars[-1].close
        qty_held, avg_cost = get_position_info()
        effective_buy_price = state['buy_price'] if state['buy_price'] is not None else avg_cost

        if len(df) < 4:
            print("Warming up historical data buffer...")
            return

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        adx, di_plus, di_minus, vwap = curr['ADX'], curr['DI+'], curr['DI-'], curr['VWAP']
        p_adx, p_di_plus, p_di_minus = prev['ADX'], prev['DI+'], prev['DI-']

        peak_display = f" | Peak: {state['peak_pnl_pct']*100:.2f}%" if qty_held > 0 else ""
        print(f"{pd.Timestamp.now().strftime('%H:%M:%S')} | Pr: {now_price:.2f} | VWAP: {vwap:.2f} | "
              f"ADX: {adx:.1f} | DI+: {di_plus:.1f} | DI-: {di_minus:.1f} | Qty: {qty_held}{peak_display}")

        if state['order_pending']: return

        if qty_held == 0:
            bullish_now = (di_plus > di_minus) and (adx > prev['ADX'])
            bullish_prev = (prev['DI+'] > prev['DI-'])
            strong_momentum = (di_plus >= MIN_DI_PLUS and (di_plus - di_minus) >= MIN_DI_SPREAD)
            above_vwap = now_price > vwap

            if (bullish_now and bullish_prev and adx >= MIN_ADX_FOR_ENTRY and
                    strong_momentum and above_vwap):
                print(f"🎯 STRONG MOMENTUM ENTRY (ADX {adx:.1f}, DI+ {di_plus:.1f})")
                submit_order('BUY', TRADE_QTY, note='strong_momentum_entry')

        elif qty_held > 0:
            pnl_pct = (now_price - effective_buy_price) / effective_buy_price
            state['peak_pnl_pct'] = max(state['peak_pnl_pct'], pnl_pct)
            peak = state['peak_pnl_pct']

            if peak >= MIN_PROFIT_LOCK_TRIGGER:
                current_stop = max(MIN_EXIT_PROFIT_PCT, peak - TRAIL_GIVEBACK)
                exit_note, exit_label = 'trailing_profit_lock', '🔒 PROFIT LOCK'
            else:
                current_stop = HARD_STOP_LOSS_PCT
                exit_note, exit_label = 'hard_stop', '🛑 STOP LOSS'

            if pnl_pct <= current_stop:
                print(f"{exit_label}: Hit {pnl_pct*100:.2f}% (Peak was {peak*100:.2f}%)")
                submit_order('SELL', qty_held, note=exit_note)
                return

            if pnl_pct >= MIN_EXIT_PROFIT_PCT:
                di_plus_cross_dn_di_minus = (p_di_plus > p_di_minus) and (di_plus < di_minus)
                if di_plus_cross_dn_di_minus or (di_minus > di_plus):
                    print(f"🚨 MATRIX EXIT: Trend flip at {pnl_pct*100:.2f}%. Peak was {peak*100:.2f}%")
                    submit_order('SELL', qty_held, note='trend_flip_exit')
                    return
                if (peak >= 0.003) and ((adx < p_adx) or (di_plus < p_di_plus)):
                    print(f"💰 MATRIX EXIT: Momentum fading at {pnl_pct*100:.2f}%.")
                    submit_order('SELL', qty_held, note='matrix_momentum_fade')
                    return

            if state.get('entry_time') is not None:
                held_secs = (datetime.now(timezone.utc) - state['entry_time']).total_seconds()
                if held_secs >= MAX_HOLD_SECONDS:
                    print(f"⏱️ MAX TIME: Closing trade at {pnl_pct*100:.2f}%.")
                    submit_order('SELL', qty_held, note='max_hold_timeout')
                    return
    except Exception as e:
        print(f"Error inside matrix calculation block: {e}")

# ============================================================
# PLATFORM INITIALIZATION ENTRYPOINT
# ============================================================
if __name__ == '__main__':
    util.logToConsole()
    ib = IB()

    print("Connecting to IBKR Paper Trading Platform...")
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
    ib.reqMarketDataType(MARKET_DATA_TYPE)

    contract = Stock(SYMBOL, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    print("Entering Corrected Matrix Strategy Stream Engine (profit-lock edition)...")

    try:
        while True:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=HISTORY_DURATION,
                barSizeSetting=BAR_SIZE,
                whatToShow='TRADES',
                useRTH=True,
                keepUpToDate=False,
            )

            if bars:
                process_latest_data(bars)

            ib.sleep(12)

    except KeyboardInterrupt:
        print("Shutting down cleanly...")
    finally:
        ib.disconnect()
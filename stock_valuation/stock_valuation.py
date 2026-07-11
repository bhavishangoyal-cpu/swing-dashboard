import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="Institutional Stock Terminal", page_icon="🏛️")


def check_password():
    """Optional gate. If APP_PASSWORD is set in Streamlit secrets (Settings ->
    Secrets on Streamlit Cloud, or .streamlit/secrets.toml locally), the app
    requires it before rendering anything. If it's not set, the app stays
    fully open — so this works with zero configuration too."""
    try:
        app_password = st.secrets.get("APP_PASSWORD", None)
    except Exception:
        app_password = None

    if not app_password:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("🏛️ Institutional Stock Terminal")
    pw = st.text_input("Password", type="password")
    if pw == app_password:
        st.session_state["authenticated"] = True
        st.rerun()
    elif pw:
        st.error("Wrong password.")
    return False


if not check_password():
    st.stop()

# ============================================================================
# 1. PILLAR MAP — raw yfinance line item -> (Pillar, Category)
# ============================================================================
PILLAR_MAP = {
    "Pillar 1: Quality": {
        "Performance": ['Total Revenue', 'Operating Revenue', 'Gross Profit', 'Operating Income',
                         'EBIT', 'EBITDA', 'Normalized EBITDA', 'ROCE %'],
        "Bottom Line": ['Net Income', 'Net Income Common Stockholders', 'Normalized Income'],
        "Efficiency": ['Operating Expense', 'Total Expenses', 'Research And Development',
                        'Selling General And Administration', 'Cost Of Revenue'],
        "Shareholder Value": ['Basic EPS', 'Diluted EPS', 'Basic Average Shares', 'Diluted Average Shares'],
        "Tax & Non-Operating": ['Tax Provision', 'Pretax Income', 'Interest Expense', 'Interest Income'],
    },
    "Pillar 2: Safety": {
        "Debt Load": ['Total Debt', 'Net Debt', 'Long Term Debt', 'Current Debt'],
        "Obligations": ['Current Liabilities', 'Accounts Payable', 'Total Liabilities Net Minority Interest'],
    },
    "Pillar 3: Assets & Liquidity": {
        "Liquidity": ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments',
                       'Accounts Receivable', 'Inventory', 'Current Assets', 'Working Capital'],
        "Hard Assets": ['Total Assets', 'Net PPE', 'Gross PPE', 'Investments And Advances', 'Goodwill'],
        "Cash Flow": ['Operating Cash Flow', 'Capital Expenditure', 'Free Cash Flow'],
    },
    "Pillar 4: Capital Structure": {
        "Equity": ['Stockholders Equity', 'Common Stock Equity', 'Total Capitalization',
                   'Tangible Book Value', 'Retained Earnings'],
        "Shares": ['Share Issued', 'Ordinary Shares Number', 'Treasury Shares Number'],
    }
}

PER_SHARE_ITEMS = {'Basic EPS', 'Diluted EPS'}
SHARE_COUNT_ITEMS = {'Basic Average Shares', 'Diluted Average Shares', 'Share Issued',
                      'Ordinary Shares Number', 'Treasury Shares Number'}
RATIO_ITEMS = {'Tax Rate For Calcs'}
PERCENT_ITEMS = {'ROCE %'}

ALL_ITEMS = [item for pillar in PILLAR_MAP.values() for cat in pillar.values() for item in cat]


# ============================================================================
# 2. DATA FETCH
# ============================================================================
def inject_roce(df):
    df = df.copy()
    if 'EBIT' in df.index and 'Total Assets' in df.index and 'Current Liabilities' in df.index:
        capital_employed = df.loc['Total Assets'] - df.loc['Current Liabilities']
        roce = (df.loc['EBIT'] / capital_employed.replace(0, np.nan)) * 100
        df.loc['ROCE %'] = roce
    return df


@st.cache_data(ttl=3600)
def fetch_statements(ticker):
    stock = yf.Ticker(ticker)
    annual = pd.concat([stock.financials, stock.balance_sheet, stock.cashflow])
    quarterly = pd.concat([stock.quarterly_financials, stock.quarterly_balance_sheet, stock.quarterly_cashflow])
    annual = annual[~annual.index.duplicated(keep='first')]
    quarterly = quarterly[~quarterly.index.duplicated(keep='first')]
    annual = annual.loc[:, ~annual.columns.duplicated()]
    quarterly = quarterly.loc[:, ~quarterly.columns.duplicated()]
    annual = inject_roce(annual)
    quarterly = inject_roce(quarterly)
    return annual, quarterly


@st.cache_data(ttl=3600)
def fetch_price_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2y")
    try:
        info = stock.info
    except Exception:
        info = {}
    return hist, info


@st.cache_data(ttl=3600)
def fetch_extended_data(ticker):
    stock = yf.Ticker(ticker)
    out = {}
    for attr, key in [('insider_transactions', 'insider_tx'),
                       ('institutional_holders', 'inst_holders'),
                       ('earnings_history', 'earnings_hist'),
                       ('earnings_dates', 'earnings_dates')]:
        try:
            out[key] = getattr(stock, attr)
        except Exception:
            out[key] = pd.DataFrame()
    return out


@st.cache_data(ttl=1800)
def fetch_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news if news else []
    except Exception:
        return []


@st.cache_data(ttl=3600)
def fetch_options_snapshot(ticker):
    try:
        stock = yf.Ticker(ticker)
        expiries = stock.options
        if not expiries:
            return None
        nearest = expiries[0]
        chain = stock.option_chain(nearest)
        calls, puts = chain.calls, chain.puts
        call_oi = calls['openInterest'].fillna(0).sum()
        put_oi = puts['openInterest'].fillna(0).sum()
        pc_ratio = round(put_oi / call_oi, 2) if call_oi else np.nan
        avg_iv_call = round(calls['impliedVolatility'].mean() * 100, 1) if 'impliedVolatility' in calls else np.nan
        avg_iv_put = round(puts['impliedVolatility'].mean() * 100, 1) if 'impliedVolatility' in puts else np.nan
        return {'Nearest Expiry': nearest, 'Put/Call OI Ratio': pc_ratio,
                'Avg Call IV %': avg_iv_call, 'Avg Put IV %': avg_iv_put,
                'Call OI': int(call_oi), 'Put OI': int(put_oi)}
    except Exception:
        return None


@st.cache_data(ttl=1800)
def fetch_macro():
    tickers = {'10Y Yield %': '^TNX', 'VIX': '^VIX', 'Dollar Index': 'DX-Y.NYB'}
    out = {}
    for label, tk in tickers.items():
        try:
            h = yf.Ticker(tk).history(period='5d')
            if not h.empty:
                val = h['Close'].iloc[-1]
                if label == '10Y Yield %':
                    val = val / 10  # ^TNX quotes yield x10
                out[label] = round(val, 2)
            else:
                out[label] = np.nan
        except Exception:
            out[label] = np.nan
    return out


# ============================================================================
# 3. RAW STATEMENT VIEW (pillar-grouped, correctly scaled, chronological)
# ============================================================================
def build_display_table(annual, quarterly, n_annual=2):
    annual = annual.sort_index(axis=1, ascending=False)
    quarterly = quarterly.sort_index(axis=1, ascending=False)

    annual_cols = annual.iloc[:, :n_annual].copy()
    latest_annual_ts = annual.columns[0] if len(annual.columns) > 0 else None

    if latest_annual_ts is not None and len(quarterly.columns) > 0:
        valid_q = [c for c in quarterly.columns if c > latest_annual_ts]
        q_cols = quarterly[valid_q].copy()
    else:
        q_cols = pd.DataFrame()

    labeled = {}
    for c in annual_cols.columns:
        labeled[c] = f"FY {c.strftime('%Y-%m-%d')}"
    for c in q_cols.columns:
        labeled[c] = f"Q ({c.strftime('%Y-%m-%d')})"

    ordered_ts = sorted(labeled.keys(), reverse=False)
    combined = pd.concat([annual_cols, q_cols], axis=1)
    combined = combined[ordered_ts]
    combined.columns = [labeled[c] for c in ordered_ts]

    df = combined.reindex(ALL_ITEMS)
    df = df.dropna(how='all')
    return df


def scale_row(row_name, value):
    if pd.isna(value):
        return np.nan
    if row_name in PER_SHARE_ITEMS:
        return round(value, 2)
    if row_name in SHARE_COUNT_ITEMS:
        return round(value / 1e6, 1)
    if row_name in RATIO_ITEMS:
        return round(value * 100, 1)
    if row_name in PERCENT_ITEMS:
        return round(value, 1)
    return round(value / 1e9, 3)


def format_display_table(df):
    out = df.copy()
    for idx in out.index:
        out.loc[idx] = [scale_row(idx, v) for v in out.loc[idx]]
    return out


# ============================================================================
# 4. RATIO ENGINE
# ============================================================================
def safe_get(df, row, col_idx=0):
    try:
        return df.loc[row].iloc[col_idx]
    except (KeyError, IndexError):
        return np.nan


def pct(cur, prior):
    if pd.isna(cur) or pd.isna(prior) or prior == 0:
        return np.nan
    return round((cur - prior) / abs(prior) * 100, 1)


def ratio_pct(num, den):
    if pd.isna(num) or pd.isna(den) or den == 0:
        return np.nan
    return round(num / den * 100, 1)


def safe_div(num, den):
    if pd.isna(num) or pd.isna(den) or den == 0:
        return np.nan
    return round(num / den, 2)


def compute_ratios(annual, quarterly):
    a = annual.sort_index(axis=1, ascending=False)

    rev = safe_get(a, 'Total Revenue', 0)
    rev_prior = safe_get(a, 'Total Revenue', 1)
    gp = safe_get(a, 'Gross Profit', 0)
    ni = safe_get(a, 'Net Income', 0)
    ebit = safe_get(a, 'EBIT', 0)
    ebitda = safe_get(a, 'EBITDA', 0)
    eps = safe_get(a, 'Diluted EPS', 0)
    eps_prior = safe_get(a, 'Diluted EPS', 1)
    interest_exp = safe_get(a, 'Interest Expense', 0)

    total_debt = safe_get(a, 'Total Debt', 0)
    equity = safe_get(a, 'Stockholders Equity', 0)
    current_assets = safe_get(a, 'Current Assets', 0)
    current_liab = safe_get(a, 'Current Liabilities', 0)
    inventory = safe_get(a, 'Inventory', 0)
    cash = safe_get(a, 'Cash And Cash Equivalents', 0)
    total_assets = safe_get(a, 'Total Assets', 0)
    shares_now = safe_get(a, 'Ordinary Shares Number', 0)
    shares_prior = safe_get(a, 'Ordinary Shares Number', 1)

    roce_now = safe_get(a, 'ROCE %', 0)
    roce_prior = safe_get(a, 'ROCE %', 1)

    quick_assets = np.nan
    if not pd.isna(current_assets) and not pd.isna(inventory):
        quick_assets = current_assets - inventory

    return {
        'Revenue YoY %': pct(rev, rev_prior),
        'EPS YoY %': pct(eps, eps_prior),
        'Gross Margin %': ratio_pct(gp, rev),
        'Net Margin %': ratio_pct(ni, rev),
        'EBITDA Margin %': ratio_pct(ebitda, rev),
        'ROCE %': round(roce_now, 1) if not pd.isna(roce_now) else np.nan,
        'ROCE YoY Δ (pts)': round(roce_now - roce_prior, 1) if not pd.isna(roce_now) and not pd.isna(roce_prior) else np.nan,
        'Debt/Equity': safe_div(total_debt, equity),
        'Interest Coverage': safe_div(ebit, interest_exp),
        'Current Ratio': safe_div(current_assets, current_liab),
        'Quick Ratio': safe_div(quick_assets, current_liab),
        'Cash/Debt': safe_div(cash, total_debt),
        'Equity Ratio %': ratio_pct(equity, total_assets),
        'Share Count YoY %': pct(shares_now, shares_prior),
    }


def roce_trend_flag(quarterly, lookback=3):
    q = quarterly.sort_index(axis=1, ascending=False)
    if 'ROCE %' not in q.index:
        return False, []
    series = q.loc['ROCE %'].dropna()
    if len(series) < lookback + 1:
        return False, [(c.strftime('%Y-%m-%d'), round(v, 1)) for c, v in series.items()]
    recent = series.iloc[:lookback + 1].iloc[::-1]
    values = recent.tolist()
    declining = all(values[i] > values[i + 1] for i in range(len(values) - 1))
    display = [(c.strftime('%Y-%m-%d'), round(v, 1)) for c, v in recent.items()]
    return declining, display


# ============================================================================
# 5. SCORING
# ============================================================================
PILLAR_SCORING = {
    "Pillar 1: Quality": [
        ('Revenue YoY %', (10, 0), True),
        ('EPS YoY %', (10, 0), True),
        ('Net Margin %', (15, 5), True),
        ('EBITDA Margin %', (20, 10), True),
        ('ROCE %', (20, 10), True),
    ],
    "Pillar 2: Safety": [
        ('Debt/Equity', (0.5, 1.5), False),
        ('Interest Coverage', (8, 3), True),
        ('Current Ratio', (1.5, 1.0), True),
    ],
    "Pillar 3: Assets & Liquidity": [
        ('Quick Ratio', (1.0, 0.5), True),
        ('Cash/Debt', (0.5, 0.2), True),
    ],
    "Pillar 4: Capital Structure": [
        ('Equity Ratio %', (50, 30), True),
        ('Share Count YoY %', (-1, 1), False),
    ],
}


def score_metric(value, thresholds, higher_is_better):
    if pd.isna(value):
        return None
    strong, moderate = thresholds
    if higher_is_better:
        if value >= strong:
            return 2
        if value >= moderate:
            return 1
        return 0
    else:
        if value <= strong:
            return 2
        if value <= moderate:
            return 1
        return 0


def score_pillars(ratios):
    results = {}
    for pillar, metrics in PILLAR_SCORING.items():
        scores, detail = [], []
        for name, thresholds, higher_better in metrics:
            val = ratios.get(name, np.nan)
            s = score_metric(val, thresholds, higher_better)
            detail.append((name, val, s))
            if s is not None:
                scores.append(s)
        if scores:
            avg = sum(scores) / len(scores)
            pct_score = round(avg / 2 * 100)
            verdict = "Strong" if avg >= 1.5 else "Moderate" if avg >= 0.75 else "Weak"
        else:
            pct_score, verdict = None, "No Data"
        results[pillar] = {'score': pct_score, 'verdict': verdict, 'detail': detail}
    return results


def overall_verdict(pillar_results):
    safety = pillar_results.get("Pillar 2: Safety", {})
    if safety.get('verdict') == 'Weak':
        return "AVOID / HIGH RISK", "Safety pillar failed — this gates the verdict regardless of quality upside."
    scores = [v['score'] for v in pillar_results.values() if v['score'] is not None]
    if not scores:
        return "NO DATA", "Insufficient data to score."
    avg = sum(scores) / len(scores)
    if avg >= 70:
        return "BUY", "All pillars supportive."
    elif avg >= 50:
        return "HOLD", "Mixed signals — acceptable but not compelling."
    else:
        return "AVOID", "Weak fundamentals across pillars."


# ============================================================================
# 6. VALUATION / TIMING OVERLAY
# ============================================================================
def valuation_timing(hist, info, fund_verdict):
    if hist is None or hist.empty:
        return None
    price = hist['Close'].iloc[-1]
    high_2y = hist['Close'].max()
    low_2y = hist['Close'].min()
    pos_pct = (price - low_2y) / (high_2y - low_2y) * 100 if high_2y != low_2y else 50

    trailing_pe = info.get('trailingPE', np.nan)
    forward_pe = info.get('forwardPE', np.nan)

    if pos_pct <= 33:
        zone = "Value Zone (lower third of 2Y range)"
    elif pos_pct <= 66:
        zone = "Fair Zone (mid-range)"
    else:
        zone = "Extended Zone (upper third of 2Y range)"

    if fund_verdict == "BUY" and pos_pct <= 40:
        action = "ENTRY — fundamentals strong, price attractive"
    elif fund_verdict == "BUY" and pos_pct > 75:
        action = "WAIT — fundamentals strong, price extended; wait for pullback"
    elif fund_verdict in ("AVOID", "AVOID / HIGH RISK") and pos_pct > 60:
        action = "EXIT / TRIM — weak fundamentals, price still elevated"
    elif fund_verdict == "HOLD":
        action = "HOLD — monitor, no urgency either way"
    else:
        action = "MONITOR"

    return {'price': price, 'high_2y': high_2y, 'low_2y': low_2y, 'position_pct': round(pos_pct, 1),
            'zone': zone, 'trailing_pe': trailing_pe, 'forward_pe': forward_pe, 'action': action}


# ============================================================================
# 7. VALUATION MULTIPLES + ACCURATE HISTORICAL P/E BAND
# ============================================================================
def valuation_multiples(info):
    return {
        'P/E (Trailing)': info.get('trailingPE', np.nan),
        'P/E (Forward)': info.get('forwardPE', np.nan),
        'P/S (TTM)': info.get('priceToSalesTrailing12Months', np.nan),
        'P/B': info.get('priceToBook', np.nan),
        'EV/EBITDA': info.get('enterpriseToEbitda', np.nan),
        'PEG Ratio': info.get('pegRatio', np.nan),
    }


def pe_band_accurate(hist, quarterly):
    """Real point-in-time trailing P/E: at each quarter-end, sum the trailing
    4 quarters of actual reported Diluted EPS, then divide the actual price
    on that date by that actual TTM EPS. No constant-EPS assumption.
    Limited by how many quarters yfinance returns (often ~4-8), so the band
    may only cover the trailing year or two, not a full multi-year history."""
    if hist is None or hist.empty:
        return None
    q = quarterly.sort_index(axis=1, ascending=True)
    if 'Diluted EPS' not in q.index:
        return None
    eps_row = q.loc['Diluted EPS'].dropna()
    if len(eps_row) < 4:
        return None
    ttm_eps = eps_row.rolling(4).sum().dropna()
    if ttm_eps.empty:
        return None

    points = []
    hist_tz = hist.index.tz
    for dt, eps_val in ttm_eps.items():
        if pd.isna(eps_val) or eps_val == 0:
            continue
        dt_compare = dt.tz_localize(hist_tz) if hist_tz is not None and dt.tzinfo is None else dt
        future_prices = hist.loc[hist.index >= dt_compare]
        if future_prices.empty:
            continue
        price_at = future_prices['Close'].iloc[0]
        points.append((dt.strftime('%Y-%m-%d'), round(price_at / eps_val, 1)))

    if not points:
        return None

    values = [v for _, v in points]
    current_eps = ttm_eps.iloc[-1]
    current_price = hist['Close'].iloc[-1]
    current_pe = round(current_price / current_eps, 1) if current_eps != 0 else np.nan

    return {'min': round(min(values), 1), 'max': round(max(values), 1),
            'median': round(float(np.median(values)), 1), 'current': current_pe,
            'points': points, 'n_quarters': len(points)}


# ============================================================================
# 8. DUPONT (ROE/ROA breakdown) + FREE CASH FLOW QUALITY
# ============================================================================
def dupont_breakdown(annual):
    a = annual.sort_index(axis=1, ascending=False)
    ni = safe_get(a, 'Net Income', 0)
    equity = safe_get(a, 'Stockholders Equity', 0)
    assets = safe_get(a, 'Total Assets', 0)
    rev = safe_get(a, 'Total Revenue', 0)

    roe = safe_div(ni, equity)
    roa = safe_div(ni, assets)
    net_margin = safe_div(ni, rev)
    asset_turnover = safe_div(rev, assets)
    equity_multiplier = safe_div(assets, equity)

    return {
        'ROE %': round(roe * 100, 1) if not pd.isna(roe) else np.nan,
        'ROA %': round(roa * 100, 1) if not pd.isna(roa) else np.nan,
        'Net Margin %': round(net_margin * 100, 1) if not pd.isna(net_margin) else np.nan,
        'Asset Turnover (x)': asset_turnover,
        'Equity Multiplier (x)': equity_multiplier,
    }


def fcf_quality(annual):
    a = annual.sort_index(axis=1, ascending=False)
    fcf = safe_get(a, 'Free Cash Flow', 0)
    if pd.isna(fcf):
        ocf = safe_get(a, 'Operating Cash Flow', 0)
        capex = safe_get(a, 'Capital Expenditure', 0)
        if not pd.isna(ocf) and not pd.isna(capex):
            fcf = ocf + capex
    rev = safe_get(a, 'Total Revenue', 0)
    ni = safe_get(a, 'Net Income', 0)

    fcf_margin = safe_div(fcf, rev)
    fcf_conversion = safe_div(fcf, ni)

    return {
        'FCF ($B)': round(fcf / 1e9, 2) if not pd.isna(fcf) else np.nan,
        'FCF Margin %': round(fcf_margin * 100, 1) if not pd.isna(fcf_margin) else np.nan,
        'FCF / Net Income (x)': fcf_conversion,
    }


# ============================================================================
# 8.5 DCF INTRINSIC VALUE
# ============================================================================
def dcf_intrinsic_value(annual, info, growth_rate, discount_rate, terminal_growth, years=5):
    a = annual.sort_index(axis=1, ascending=False)
    fcf = safe_get(a, 'Free Cash Flow', 0)
    if pd.isna(fcf):
        ocf = safe_get(a, 'Operating Cash Flow', 0)
        capex = safe_get(a, 'Capital Expenditure', 0)
        if not pd.isna(ocf) and not pd.isna(capex):
            fcf = ocf + capex
    if pd.isna(fcf) or fcf <= 0:
        return None

    total_debt = safe_get(a, 'Total Debt', 0)
    cash = safe_get(a, 'Cash And Cash Equivalents', 0)
    shares = info.get('sharesOutstanding', np.nan)
    if pd.isna(shares) or shares == 0:
        shares = safe_get(a, 'Ordinary Shares Number', 0)
    if pd.isna(shares) or shares == 0:
        return None

    if discount_rate <= terminal_growth:
        return None

    pv_sum = 0.0
    projected = []
    for t in range(1, years + 1):
        fcf_t = fcf * (1 + growth_rate) ** t
        pv = fcf_t / (1 + discount_rate) ** t
        pv_sum += pv
        projected.append((t, round(fcf_t / 1e9, 3), round(pv / 1e9, 3)))

    fcf_final = fcf * (1 + growth_rate) ** years
    terminal_value = fcf_final * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** years

    enterprise_value = pv_sum + pv_terminal
    net_debt = (total_debt if not pd.isna(total_debt) else 0) - (cash if not pd.isna(cash) else 0)
    equity_value = enterprise_value - net_debt
    fair_value_per_share = equity_value / shares

    return {
        'fair_value': round(fair_value_per_share, 2),
        'enterprise_value_b': round(enterprise_value / 1e9, 2),
        'equity_value_b': round(equity_value / 1e9, 2),
        'terminal_value_b': round(terminal_value / 1e9, 2),
        'pv_terminal_pct': round(pv_terminal / enterprise_value * 100, 1) if enterprise_value else np.nan,
        'projected': projected,
    }


def dcf_sensitivity(annual, info, growth_rate, base_discount, base_terminal):
    rows = []
    for dr in [base_discount - 0.02, base_discount, base_discount + 0.02]:
        row = {'Discount Rate': f"{dr*100:.1f}%"}
        for tg in [base_terminal - 0.01, base_terminal, base_terminal + 0.01]:
            res = dcf_intrinsic_value(annual, info, growth_rate, dr, tg)
            row[f"Terminal g={tg*100:.1f}%"] = res['fair_value'] if res else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index('Discount Rate')


# ============================================================================
# 9. ANALYST CONSENSUS, OWNERSHIP, EARNINGS QUALITY, DIVIDENDS
# ============================================================================
def analyst_consensus(info):
    target_mean = info.get('targetMeanPrice', np.nan)
    current = info.get('currentPrice', np.nan)
    upside = np.nan
    if not pd.isna(target_mean) and not pd.isna(current) and current != 0:
        upside = round((target_mean - current) / current * 100, 1)
    return {
        'Recommendation': info.get('recommendationKey', 'n/a'),
        'Mean Rating (1=Strong Buy, 5=Sell)': info.get('recommendationMean', np.nan),
        '# Analysts': info.get('numberOfAnalystOpinions', np.nan),
        'Target Mean': target_mean,
        'Target High': info.get('targetHighPrice', np.nan),
        'Target Low': info.get('targetLowPrice', np.nan),
        'Implied Upside %': upside,
    }


def ownership_snapshot(info, insider_tx):
    ins = info.get('heldPercentInsiders', np.nan)
    inst = info.get('heldPercentInstitutions', np.nan)
    result = {
        'Insider Ownership %': round(ins * 100, 1) if ins is not None and not pd.isna(ins) else np.nan,
        'Institutional Ownership %': round(inst * 100, 1) if inst is not None and not pd.isna(inst) else np.nan,
    }
    activity = 'N/A'
    try:
        if insider_tx is not None and not insider_tx.empty:
            recent = insider_tx.head(10)
            text_col = None
            for c in ['Transaction', 'Text', 'transactionText']:
                if c in recent.columns:
                    text_col = c
                    break
            if text_col:
                buys = recent[text_col].astype(str).str.contains('Buy', case=False, na=False).sum()
                sells = recent[text_col].astype(str).str.contains('Sale|Sell', case=False, na=False).sum()
                activity = f"{buys} buys / {sells} sells (last {len(recent)} filings)"
    except Exception:
        activity = 'Could not parse insider filings'
    result['Recent Insider Activity'] = activity
    return result


def earnings_quality(earnings_hist, earnings_dates):
    result = {'beat_rate': 'N/A', 'next_earnings': 'N/A'}
    try:
        if earnings_hist is not None and not earnings_hist.empty:
            recent = earnings_hist.tail(4)
            if 'epsActual' in recent.columns and 'epsEstimate' in recent.columns:
                beats = int((recent['epsActual'] > recent['epsEstimate']).sum())
                result['beat_rate'] = f"{beats}/{len(recent)} beats (last {len(recent)} qtrs)"
    except Exception:
        pass
    try:
        if earnings_dates is not None and not earnings_dates.empty:
            now = pd.Timestamp.now(tz=earnings_dates.index.tz) if earnings_dates.index.tz else pd.Timestamp.now()
            future = earnings_dates[earnings_dates.index > now]
            if len(future) > 0:
                result['next_earnings'] = future.index.min().strftime('%Y-%m-%d')
    except Exception:
        pass
    return result


def dividend_buyback(info, ratios):
    div_yield = info.get('dividendYield', np.nan)
    payout = info.get('payoutRatio', np.nan)
    buyback_yield = -ratios.get('Share Count YoY %', np.nan) if not pd.isna(ratios.get('Share Count YoY %', np.nan)) else np.nan
    dy = div_yield if div_yield else 0.0
    return {
        'Dividend Yield %': round(dy, 2) if not pd.isna(dy) else 0.0,
        'Payout Ratio %': round(payout * 100, 1) if payout and not pd.isna(payout) else np.nan,
        'Buyback Yield %': round(buyback_yield, 1) if not pd.isna(buyback_yield) else np.nan,
    }


def short_interest_snapshot(info):
    spf = info.get('shortPercentOfFloat', np.nan)
    spo = info.get('sharesPercentSharesOut', np.nan)
    return {
        'Short % of Float': round(spf * 100, 2) if spf is not None and not pd.isna(spf) else np.nan,
        'Shares Short': info.get('sharesShort', np.nan),
        'Short Ratio (days to cover)': info.get('shortRatio', np.nan),
        'Short % of Shares Out': round(spo * 100, 2) if spo is not None and not pd.isna(spo) else np.nan,
    }


# ============================================================================
# 10. TECHNICAL CONFIRMATION LAYER
# ============================================================================
def technical_indicators(hist):
    if hist is None or hist.empty or len(hist) < 50:
        return None
    close = hist['Close']
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
    price = close.iloc[-1]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_now = rsi.iloc[-1]

    if not pd.isna(sma200) and price > sma50 > sma200:
        trend = 'Uptrend (price > 50MA > 200MA)'
    elif not pd.isna(sma200) and price < sma50 < sma200:
        trend = 'Downtrend (price < 50MA < 200MA)'
    else:
        trend = 'Mixed / Transitional'

    if pd.isna(rsi_now):
        rsi_note = 'N/A'
    elif rsi_now >= 70:
        rsi_note = 'Overbought'
    elif rsi_now <= 30:
        rsi_note = 'Oversold'
    else:
        rsi_note = 'Neutral'

    return {'Price': round(price, 2), 'SMA50': round(sma50, 2) if not pd.isna(sma50) else np.nan,
            'SMA200': round(sma200, 2) if not pd.isna(sma200) else np.nan,
            'Trend': trend, 'RSI(14)': round(rsi_now, 1) if not pd.isna(rsi_now) else np.nan,
            'RSI Note': rsi_note}


def technical_divergence_note(verdict, technical):
    if not technical:
        return "Not enough price history for a technical read."
    if verdict == "BUY" and 'Downtrend' in technical['Trend']:
        return "Fundamentals say buy, but price is still in a downtrend — no technical confirmation yet."
    if verdict in ("AVOID", "AVOID / HIGH RISK") and 'Uptrend' in technical['Trend']:
        return "Price is rising despite weak fundamentals — that's momentum/euphoria risk, not a quality rally."
    return "Technicals broadly align with the fundamental verdict."


# ============================================================================
# 11. FORENSIC RED FLAGS
# ============================================================================
def forensic_checks(annual):
    a = annual.sort_index(axis=1, ascending=False)
    ni = safe_get(a, 'Net Income', 0)
    ocf = safe_get(a, 'Operating Cash Flow', 0)
    assets = safe_get(a, 'Total Assets', 0)
    goodwill = safe_get(a, 'Goodwill', 0)
    rev = safe_get(a, 'Total Revenue', 0)
    rev_prior = safe_get(a, 'Total Revenue', 1)
    ar = safe_get(a, 'Accounts Receivable', 0)
    ar_prior = safe_get(a, 'Accounts Receivable', 1)

    flags = []
    accrual_ratio = np.nan
    if not pd.isna(ni) and not pd.isna(ocf) and not pd.isna(assets) and assets != 0:
        accrual_ratio = (ni - ocf) / assets * 100
        if accrual_ratio > 5:
            flags.append(f"High accruals ratio ({accrual_ratio:.1f}%) — net income running well ahead of "
                          f"cash flow, an earnings-quality red flag")

    goodwill_pct = np.nan
    if not pd.isna(goodwill) and not pd.isna(assets) and assets != 0:
        goodwill_pct = goodwill / assets * 100
        if goodwill_pct > 30:
            flags.append(f"Goodwill/intangibles are {goodwill_pct:.0f}% of total assets — impairment risk "
                          f"if growth stalls")

    ar_growth = pct(ar, ar_prior)
    rev_growth = pct(rev, rev_prior)
    if not pd.isna(ar_growth) and not pd.isna(rev_growth) and ar_growth > rev_growth + 15:
        flags.append(f"Receivables growing much faster than revenue ({ar_growth}% vs {rev_growth}%) — "
                      f"possible channel stuffing or collection issues")

    return {'Accruals Ratio %': round(accrual_ratio, 1) if not pd.isna(accrual_ratio) else np.nan,
            'Goodwill/Assets %': round(goodwill_pct, 1) if not pd.isna(goodwill_pct) else np.nan,
            'Receivables YoY %': ar_growth, 'Revenue YoY %': rev_growth, 'flags': flags}


# ============================================================================
# 12. PEER / SECTOR / PORTFOLIO SNAPSHOTS
# ============================================================================
def quick_peer_snapshot(ticker):
    try:
        annual, quarterly = fetch_statements(ticker)
        if annual.empty:
            return None
        ratios = compute_ratios(annual, quarterly)
        pillar_results = score_pillars(ratios)
        verdict, _ = overall_verdict(pillar_results)
        scores = [v['score'] for v in pillar_results.values() if v['score'] is not None]
        overall_score = round(np.mean(scores)) if scores else None
        return {'Ticker': ticker, 'Verdict': verdict,
                'Overall Score': overall_score if overall_score is not None else 'N/A',
                'ROCE %': ratios.get('ROCE %', np.nan), 'Net Margin %': ratios.get('Net Margin %', np.nan),
                'Debt/Equity': ratios.get('Debt/Equity', np.nan), 'Revenue YoY %': ratios.get('Revenue YoY %', np.nan)}
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_portfolio_prices(tickers, benchmark='SPY', period='1y'):
    all_tickers = list(dict.fromkeys(tickers + [benchmark]))
    data = {}
    for t in all_tickers:
        try:
            h = yf.Ticker(t).history(period=period)
            if not h.empty:
                data[t] = h['Close']
        except Exception:
            continue
    if not data:
        return None
    return pd.DataFrame(data).dropna(how='all')


def compute_beta(returns, stock_col, bench_col):
    if stock_col not in returns.columns or bench_col not in returns.columns:
        return np.nan
    aligned = returns[[stock_col, bench_col]].dropna()
    if len(aligned) < 20:
        return np.nan
    cov = aligned[stock_col].cov(aligned[bench_col])
    var = aligned[bench_col].var()
    return round(cov / var, 2) if var else np.nan


# ============================================================================
# 13. COMPOSITE GRADE + AUTO-GENERATED THESIS
# ============================================================================
def composite_grade(pillar_results, roce_declining, forensic_flags):
    scores = [v['score'] for v in pillar_results.values() if v['score'] is not None]
    base = np.mean(scores) if scores else 50
    penalty = 0
    if roce_declining:
        penalty += 20
    penalty += min(len(forensic_flags), 3) * 8
    final = max(0, base - penalty)
    if final >= 85:
        grade = 'A'
    elif final >= 70:
        grade = 'B'
    elif final >= 55:
        grade = 'C'
    elif final >= 40:
        grade = 'D'
    else:
        grade = 'F'
    return grade, round(final)


def generate_thesis(ticker, verdict, pillar_results, ratios, roce_declining, val, analyst, technical, forensic):
    lines = [f"**{ticker}: {verdict}.**"]
    q = pillar_results.get("Pillar 1: Quality", {})
    s = pillar_results.get("Pillar 2: Safety", {})
    lines.append(f"Quality is {q.get('verdict', 'N/A')} (ROCE {ratios.get('ROCE %', 'N/A')}%, "
                  f"net margin {ratios.get('Net Margin %', 'N/A')}%); "
                  f"Safety is {s.get('verdict', 'N/A')} (D/E {ratios.get('Debt/Equity', 'N/A')}).")
    if roce_declining:
        lines.append("Capital efficiency is deteriorating over the last 3 quarters — "
                      "treat any bullish price action with caution.")
    if forensic.get('flags'):
        lines.append("Forensic flags: " + "; ".join(forensic['flags']) + ".")
    if val:
        lines.append(f"Price sits in the {val['zone'].split(' (')[0]} of its 2-year range.")
    if analyst.get('Recommendation') not in (None, 'n/a'):
        lines.append(f"Street consensus: {analyst.get('Recommendation')} "
                      f"({analyst.get('# Analysts', '?')} analysts, target ${analyst.get('Target Mean', 'n/a')}, "
                      f"implied upside {analyst.get('Implied Upside %', 'n/a')}%).")
    if technical:
        lines.append(f"Technicals show a {technical['Trend'].lower()}, RSI {technical['RSI Note'].lower()}.")
    return " ".join(lines)


# ============================================================================
# 14. UI
# ============================================================================
st.title("🏛️ Institutional Stock Terminal")
st.caption("4-Pillar fundamental scorecard, ROCE trend, DCF, DuPont/FCF quality, ownership & analyst "
           "consensus, short interest & options, news, technical cross-check, forensic red flags, and "
           "portfolio-level view — rolled into one letter grade and thesis.")

macro = fetch_macro()
mcols = st.columns(3)
for i, (k, v) in enumerate(macro.items()):
    mcols[i].metric(k, f"{v}" if not pd.isna(v) else "N/A")

mode = st.radio("Mode", ["Single Stock Analysis", "My Portfolio"], horizontal=True)

# ----------------------------------------------------------------------------
# SINGLE STOCK MODE
# ----------------------------------------------------------------------------
if mode == "Single Stock Analysis":
    ticker = st.text_input("Enter Ticker:", "AAPL").upper()

    if st.button("Analyze"):
        with st.spinner(f"Pulling {ticker}..."):
            annual, quarterly = fetch_statements(ticker)
            hist, info = fetch_price_data(ticker)
            ext = fetch_extended_data(ticker)
            news = fetch_news(ticker)
            options_snap = fetch_options_snapshot(ticker)

        if annual.empty:
            st.error("No data returned — check the ticker.")
        else:
            ratios = compute_ratios(annual, quarterly)
            pillar_results = score_pillars(ratios)
            verdict, reason = overall_verdict(pillar_results)
            val = valuation_timing(hist, info, verdict)
            roce_declining, roce_series = roce_trend_flag(quarterly)
            forensic = forensic_checks(annual)
            analyst = analyst_consensus(info)
            technical = technical_indicators(hist)
            grade, grade_score = composite_grade(pillar_results, roce_declining, forensic['flags'])
            thesis = generate_thesis(ticker, verdict, pillar_results, ratios, roce_declining, val, analyst,
                                      technical, forensic)

            icon = {"BUY": "🟢", "HOLD": "🟡", "AVOID": "🔴", "AVOID / HIGH RISK": "🔴", "NO DATA": "⚪"}.get(verdict, "⚪")
            c1, c2 = st.columns([1, 4])
            with c1:
                st.metric("Composite Grade", f"{grade}", f"{grade_score}/100")
            with c2:
                st.header(f"{icon} {verdict}")
            st.write(thesis)

            if val:
                action = val['action']
                if roce_declining:
                    action = "EXIT — ROCE deteriorating for 3+ straight quarters"
                st.write(f"**Price:** ${val['price']:.2f}  |  **2Y Range Position:** {val['position_pct']}% "
                         f"({val['zone']})  |  **Action:** {action}")
            if roce_declining:
                trend_str = " → ".join(f"{d}: {v}%" for d, v in roce_series)
                st.warning(f"⚠️ ROCE trend declining: {trend_str}")
            if forensic['flags']:
                st.warning("🚩 " + " | ".join(forensic['flags']))

            tabs = st.tabs(["📊 Pillar Scorecard", "💰 Valuation", "🎯 Intrinsic Value (DCF)", "🔬 DuPont & FCF",
                             "👥 Ownership & Analysts", "🩳 Short Interest & Options", "📰 News",
                             "📉 Technicals", "🚩 Red Flags", "⚖️ Peer Comparison", "📄 Raw Financials"])

            with tabs[0]:
                cols = st.columns(4)
                for i, (pillar, res) in enumerate(pillar_results.items()):
                    with cols[i]:
                        st.subheader(pillar.split(": ")[1])
                        score_label = f"{res['score']}/100" if res['score'] is not None else "N/A"
                        st.metric("Score", score_label, res['verdict'])
                        for name, v, s in res['detail']:
                            mark = "✅" if s == 2 else "⚠️" if s == 1 else "❌" if s == 0 else "—"
                            v_str = f"{v}" if not pd.isna(v) else "N/A"
                            st.write(f"{mark} {name}: {v_str}")

            with tabs[1]:
                st.subheader("Valuation multiples")
                mult = valuation_multiples(info)
                vcols = st.columns(3)
                for i, (k, v) in enumerate(mult.items()):
                    vcols[i % 3].metric(k, f"{v:.2f}" if isinstance(v, (int, float)) and not pd.isna(v) else "N/A")

                st.subheader("Historical P/E band (actual trailing-4Q EPS at each date)")
                band = pe_band_accurate(hist, quarterly)
                if band:
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Low", band['min'])
                    b2.metric("Median", band['median'])
                    b3.metric("High", band['max'])
                    b4.metric("Current", band['current'])
                    st.caption(f"Based on {band['n_quarters']} actual quarterly EPS points — yfinance typically "
                               f"only returns a handful of trailing quarters, so this band may cover less than "
                               f"a full multi-year history.")
                else:
                    st.write("Not enough quarterly EPS history for a real P/E band.")

                if val:
                    st.line_chart(hist['Close'])

            with tabs[2]:
                st.subheader("Discounted Cash Flow — intrinsic value")
                st.caption("Adjust the assumptions — there's no single 'correct' discount rate or growth rate.")

                default_growth = ratios.get('Revenue YoY %', np.nan)
                default_growth = min(max(default_growth / 100, 0.02), 0.25) if not pd.isna(default_growth) else 0.08

                dc1, dc2, dc3, dc4 = st.columns(4)
                growth_input = dc1.slider("FCF growth rate (yrs 1-5)", 0.0, 0.30, float(round(default_growth, 2)),
                                           0.01, key="dcf_growth")
                discount_input = dc2.slider("Discount rate (WACC proxy)", 0.05, 0.15, 0.10, 0.005, key="dcf_discount")
                terminal_input = dc3.slider("Terminal growth rate", 0.0, 0.04, 0.025, 0.0025, key="dcf_terminal")
                years_input = dc4.selectbox("Projection years", [3, 5, 7, 10], index=1, key="dcf_years")

                dcf = dcf_intrinsic_value(annual, info, growth_input, discount_input, terminal_input, years_input)

                if dcf is None:
                    st.write("Couldn't compute a DCF — missing FCF/shares data, or discount rate isn't above "
                             "terminal growth.")
                else:
                    current_price = val['price'] if val else np.nan
                    margin_of_safety = np.nan
                    if not pd.isna(current_price) and current_price != 0:
                        margin_of_safety = round((dcf['fair_value'] - current_price) / current_price * 100, 1)

                    r1, r2, r3 = st.columns(3)
                    r1.metric("Fair Value / Share", f"${dcf['fair_value']}")
                    r2.metric("Current Price", f"${current_price:.2f}" if not pd.isna(current_price) else "N/A")
                    r3.metric("Margin of Safety", f"{margin_of_safety}%" if not pd.isna(margin_of_safety) else "N/A")

                    if not pd.isna(margin_of_safety):
                        if margin_of_safety >= 20:
                            st.success("Trading well below DCF fair value — margin-of-safety territory.")
                        elif margin_of_safety <= -20:
                            st.warning("Trading well above DCF fair value — priced for a lot of future growth.")
                        else:
                            st.info("Roughly in line with DCF fair value — no strong signal either way.")

                    st.caption(f"Enterprise Value ${dcf['enterprise_value_b']}B · Equity Value ${dcf['equity_value_b']}B "
                               f"· Terminal Value ${dcf['terminal_value_b']}B "
                               f"({dcf['pv_terminal_pct']}% of enterprise value).")

                    proj_df = pd.DataFrame(dcf['projected'], columns=['Year', 'Projected FCF ($B)', 'PV of FCF ($B)'])
                    st.dataframe(proj_df.set_index('Year'), use_container_width=True)

                    st.subheader("Sensitivity: fair value across discount rate × terminal growth")
                    sens = dcf_sensitivity(annual, info, growth_input, discount_input, terminal_input)
                    st.dataframe(sens, use_container_width=True)

            with tabs[3]:
                st.subheader("DuPont breakdown (ROE = Net Margin × Asset Turnover × Equity Multiplier)")
                dp = dupont_breakdown(annual)
                dcols = st.columns(5)
                for i, (k, v) in enumerate(dp.items()):
                    dcols[i].metric(k, f"{v}" if not pd.isna(v) else "N/A")

                st.subheader("Free cash flow quality")
                fq = fcf_quality(annual)
                fcols = st.columns(3)
                for i, (k, v) in enumerate(fq.items()):
                    fcols[i].metric(k, f"{v}" if not pd.isna(v) else "N/A")
                st.caption("FCF/Net Income well below 1.0x means earnings aren't converting to cash.")

            with tabs[4]:
                st.subheader("Analyst consensus")
                acols = st.columns(4)
                for i, (k, v) in enumerate(analyst.items()):
                    acols[i % 4].metric(k, f"{v}" if not (isinstance(v, float) and pd.isna(v)) else "N/A")

                st.subheader("Ownership")
                own = ownership_snapshot(info, ext.get('insider_tx'))
                ocols = st.columns(3)
                for i, (k, v) in enumerate(own.items()):
                    ocols[i % 3].metric(k, f"{v}" if not (isinstance(v, float) and pd.isna(v)) else "N/A")
                st.caption("Insider activity parsing is best-effort — yfinance's schema varies by ticker/version.")

                st.subheader("Earnings quality")
                eq = earnings_quality(ext.get('earnings_hist'), ext.get('earnings_dates'))
                ecols = st.columns(2)
                ecols[0].metric("Beat/Miss (last 4 qtrs)", eq['beat_rate'])
                ecols[1].metric("Next Earnings Date", eq['next_earnings'])

                st.subheader("Dividends & buybacks")
                db = dividend_buyback(info, ratios)
                dbcols = st.columns(3)
                for i, (k, v) in enumerate(db.items()):
                    dbcols[i].metric(k, f"{v}" if not (isinstance(v, float) and pd.isna(v)) else "N/A")

            with tabs[5]:
                st.subheader("Short interest")
                si = short_interest_snapshot(info)
                sicols = st.columns(4)
                for i, (k, v) in enumerate(si.items()):
                    sicols[i].metric(k, f"{v}" if not (isinstance(v, float) and pd.isna(v)) else "N/A")

                st.subheader("Options (nearest expiry)")
                if options_snap:
                    ocols2 = st.columns(3)
                    items2 = list(options_snap.items())
                    for i, (k, v) in enumerate(items2):
                        ocols2[i % 3].metric(k, f"{v}")
                    st.caption("Put/Call OI ratio above ~1.0 skews bearish positioning; high IV means the "
                               "options market is pricing a big expected move, in either direction.")
                else:
                    st.write("No options data available for this ticker.")

            with tabs[6]:
                st.subheader("Recent news")
                if news:
                    for item in news[:10]:
                        content = item.get('content', item)  # handle both yfinance schema versions
                        title = content.get('title', 'Untitled')
                        link = content.get('canonicalUrl', {}).get('url') if isinstance(content.get('canonicalUrl'), dict) else content.get('link', '')
                        publisher = content.get('provider', {}).get('displayName') if isinstance(content.get('provider'), dict) else content.get('publisher', '')
                        st.write(f"**[{title}]({link})** — {publisher}")
                else:
                    st.write("No recent news found.")

            with tabs[7]:
                if technical:
                    tcols = st.columns(5)
                    for i, (k, v) in enumerate(technical.items()):
                        tcols[i % 5].metric(k, f"{v}")
                    st.write(technical_divergence_note(verdict, technical))
                    st.line_chart(hist['Close'])
                else:
                    st.write("Not enough price history (need 50+ trading days).")

            with tabs[8]:
                st.subheader("Forensic red flags")
                frcols = st.columns(4)
                frcols[0].metric("Accruals Ratio %", forensic['Accruals Ratio %'])
                frcols[1].metric("Goodwill/Assets %", forensic['Goodwill/Assets %'])
                frcols[2].metric("Receivables YoY %", forensic['Receivables YoY %'])
                frcols[3].metric("Revenue YoY %", forensic['Revenue YoY %'])
                if forensic['flags']:
                    for f in forensic['flags']:
                        st.error(f)
                else:
                    st.success("No forensic red flags triggered on the current thresholds.")

            with tabs[9]:
                st.subheader("Peer / sector comparison")
                peer_input = st.text_input("Peer tickers (comma-separated):", "", key="peer_input")
                if st.button("Compare Peers"):
                    peer_tickers = [t.strip().upper() for t in peer_input.split(",") if t.strip()]
                    if not peer_tickers:
                        st.write("Enter at least one peer ticker.")
                    else:
                        rows = [quick_peer_snapshot(ticker)]
                        for pt in peer_tickers:
                            snap = quick_peer_snapshot(pt)
                            if snap:
                                rows.append(snap)
                        rows = [r for r in rows if r]
                        if rows:
                            st.dataframe(pd.DataFrame(rows).set_index('Ticker'), use_container_width=True)
                        else:
                            st.write("Couldn't pull data for any of the tickers entered.")

            with tabs[10]:
                display_df = build_display_table(annual, quarterly)
                formatted = format_display_table(display_df)
                st.dataframe(formatted, use_container_width=True)
                st.caption("Currency items in $B · EPS in $/share · share counts in millions · tax rate and ROCE in %.")

# ----------------------------------------------------------------------------
# PORTFOLIO MODE
# ----------------------------------------------------------------------------
elif mode == "My Portfolio":
    st.subheader("Portfolio holdings")
    st.caption("One holding per line: TICKER, SHARES, COST_BASIS_PER_SHARE  (e.g. META,50,320)")
    holdings_text = st.text_area("Holdings", "META,50,320\nGOOGL,20,140\nPLTR,100,25", height=120)

    if st.button("Analyze Portfolio"):
        holdings = []
        for line in holdings_text.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0]:
                try:
                    tk = parts[0].upper()
                    shares = float(parts[1])
                    cost = float(parts[2]) if len(parts) > 2 and parts[2] else np.nan
                    holdings.append((tk, shares, cost))
                except ValueError:
                    continue

        if not holdings:
            st.write("Enter at least one valid holding.")
        else:
            tickers_list = [h[0] for h in holdings]
            with st.spinner("Pulling portfolio data..."):
                price_df = fetch_portfolio_prices(tickers_list)
                rows = []
                for tk, shares, cost in holdings:
                    snap = quick_peer_snapshot(tk)
                    if not snap:
                        continue
                    price = np.nan
                    if price_df is not None and tk in price_df.columns:
                        s = price_df[tk].dropna()
                        price = s.iloc[-1] if not s.empty else np.nan
                    value = shares * price if not pd.isna(price) else np.nan
                    pl = (price - cost) * shares if not pd.isna(price) and not pd.isna(cost) else np.nan
                    pl_pct = round((price - cost) / cost * 100, 1) if not pd.isna(price) and not pd.isna(cost) and cost != 0 else np.nan
                    rows.append({**snap, 'Shares': shares,
                                 'Price': round(price, 2) if not pd.isna(price) else np.nan,
                                 'Position Value': round(value, 2) if not pd.isna(value) else np.nan,
                                 'Cost Basis': cost,
                                 'Unrealized P/L': round(pl, 2) if not pd.isna(pl) else np.nan,
                                 'P/L %': pl_pct})

            if rows:
                pdf = pd.DataFrame(rows).set_index('Ticker')
                st.dataframe(pdf, use_container_width=True)

                total_value = pdf['Position Value'].sum(skipna=True)
                total_pl = pdf['Unrealized P/L'].sum(skipna=True)
                avg_score = pd.to_numeric(pdf['Overall Score'], errors='coerce').dropna()

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Portfolio Value", f"${total_value:,.0f}")
                c2.metric("Total Unrealized P/L", f"${total_pl:,.0f}")
                c3.metric("Weighted Avg Score", f"{avg_score.mean():.0f}/100" if not avg_score.empty else "N/A")

                if price_df is not None:
                    st.subheader("Correlation matrix (1Y daily returns)")
                    returns = price_df.pct_change().dropna(how='all')
                    valid_tickers = [t for t in tickers_list if t in returns.columns]
                    if len(valid_tickers) >= 2:
                        corr = returns[valid_tickers].corr()
                        st.dataframe(corr, use_container_width=True)

                    st.subheader("Beta vs SPY")
                    beta_rows = [{'Ticker': tk, 'Beta vs SPY': compute_beta(returns, tk, 'SPY')} for tk in valid_tickers]
                    st.dataframe(pd.DataFrame(beta_rows).set_index('Ticker'), use_container_width=True)
                    st.caption("Beta > 1 moves more than the market; < 1 moves less. Useful for sizing "
                               "position risk relative to your overall exposure.")
            else:
                st.write("Couldn't pull data for any holdings entered.")
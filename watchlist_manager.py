import yfinance as yf
import math
import os
from colorama import init, Fore, Style
init(autoreset=True)

# ── Popular companies (just for quick search convenience) ────────
POPULAR = {
    # US – Tech
    "Apple Inc":              "AAPL",
    "Microsoft":              "MSFT",
    "Alphabet (Google)":      "GOOGL",
    "Amazon":                 "AMZN",
    "Meta Platforms":         "META",
    "Netflix":                "NFLX",
    "NVIDIA":                 "NVDA",
    "Tesla":                  "TSLA",
    "Intel":                  "INTC",
    "AMD":                    "AMD",
    "Qualcomm":               "QCOM",
    "Salesforce":             "CRM",
    "Adobe":                  "ADBE",
    "PayPal":                 "PYPL",
    "Uber":                   "UBER",
    "Oracle":                 "ORCL",
    "IBM":                    "IBM",
    "Cisco":                  "CSCO",
    # US – Finance
    "JPMorgan Chase":         "JPM",
    "Goldman Sachs":          "GS",
    "Bank of America":        "BAC",
    "Berkshire Hathaway":     "BRK-B",
    "Visa":                   "V",
    "Mastercard":             "MA",
    "American Express":       "AXP",
    # US – Healthcare
    "Johnson & Johnson":      "JNJ",
    "Pfizer":                 "PFE",
    "Eli Lilly":              "LLY",
    "AbbVie":                 "ABBV",
    "Moderna":                "MRNA",
    # US – Consumer
    "Coca-Cola":              "KO",
    "PepsiCo":                "PEP",
    "McDonald's":             "MCD",
    "Walmart":                "WMT",
    "Nike":                   "NKE",
    "Procter & Gamble":       "PG",
    "Costco":                 "COST",
    "Starbucks":              "SBUX",
    # US – Energy
    "ExxonMobil":             "XOM",
    "Chevron":                "CVX",
    # India – IT
    "TCS":                    "TCS.NS",
    "Infosys":                "INFY.NS",
    "Wipro":                  "WIPRO.NS",
    "HCL Technologies":       "HCLTECH.NS",
    "Tech Mahindra":          "TECHM.NS",
    "LTIMindtree":            "LTIM.NS",
    # India – Banking
    "HDFC Bank":              "HDFCBANK.NS",
    "ICICI Bank":             "ICICIBANK.NS",
    "State Bank of India":    "SBIN.NS",
    "Axis Bank":              "AXISBANK.NS",
    "Kotak Mahindra Bank":    "KOTAKBANK.NS",
    "Bajaj Finance":          "BAJFINANCE.NS",
    # India – Large Cap
    "Reliance Industries":    "RELIANCE.NS",
    "Adani Enterprises":      "ADANIENT.NS",
    "Tata Motors":            "TATAMOTORS.NS",
    "Maruti Suzuki":          "MARUTI.NS",
    "Asian Paints":           "ASIANPAINT.NS",
    "Hindustan Unilever":     "HINDUNILVR.NS",
    "ITC":                    "ITC.NS",
    "Sun Pharma":             "SUNPHARMA.NS",
    "Larsen & Toubro":        "LT.NS",
    "Tata Steel":             "TATASTEEL.NS",
    # Global
    "Samsung":                "005930.KS",
    "TSMC":                   "TSM",
    "Alibaba":                "BABA",
    "Toyota":                 "TM",
    "ASML":                   "ASML",
    "Novo Nordisk":           "NVO",
    "Shell":                  "SHEL",
    "SAP":                    "SAP",
}

SORTED_NAMES = sorted(POPULAR.keys())

# ── Helpers ──────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def header():
    print(Fore.CYAN + Style.BRIGHT)
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       SHARE FAIR VALUE CALCULATOR  —  ANY STOCK/TICKER      ║")
    print("║   Graham · DCF · P/E · P/B · DDM · PEG · Factor Scoring    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(Style.RESET_ALL)


def safe(val, default=None):
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return val
    except:
        return default


def color_price(fv, market):
    if fv is None or market is None:
        return "N/A"
    diff = (fv - market) / market * 100
    s = f"${fv:>9.2f}   ({diff:+.1f}%)"
    if diff > 10:  return Fore.GREEN  + s + Style.RESET_ALL
    if diff < -10: return Fore.RED    + s + Style.RESET_ALL
    return Fore.CYAN + s + Style.RESET_ALL


def color_verdict(v):
    if "STRONG BUY" in v: return Fore.GREEN  + Style.BRIGHT + v + Style.RESET_ALL
    if "BUY"        in v: return Fore.GREEN  + v + Style.RESET_ALL
    if "HOLD"       in v: return Fore.CYAN   + v + Style.RESET_ALL
    if "CAUTION"    in v: return Fore.YELLOW + v + Style.RESET_ALL
    return Fore.RED + v + Style.RESET_ALL


def score_m(val, good, ok, bad):
    if val is None: return 2.5
    if val >= good: return 5.0
    if val >= ok:   return 3.5
    if val >= bad:  return 2.0
    return 1.0


def graham_number(eps, bvps):
    try:
        if eps > 0 and bvps > 0:
            return math.sqrt(22.5 * eps * bvps)
    except: pass
    return None


def dcf_value(fcf_ps, g_rate, t_growth, discount):
    try:
        if fcf_ps <= 0 or discount <= t_growth: return None
        g, tg, r = g_rate / 100, t_growth / 100, discount / 100
        pv, cf = 0, fcf_ps
        for i in range(1, 6):
            cf *= (1 + g)
            pv += cf / (1 + r) ** i
        return pv + cf * (1 + tg) / (r - tg) / (1 + r) ** 5
    except: return None


# ── Company selection screen ─────────────────────────────────────

def select_company():
    while True:
        clear()
        header()

        print(Fore.WHITE + Style.BRIGHT + "  How would you like to select a company?" + Style.RESET_ALL)
        print()
        print(f"  {Fore.YELLOW}1.{Style.RESET_ALL} Search from popular companies list")
        print(f"  {Fore.YELLOW}2.{Style.RESET_ALL} Enter any ticker directly  (e.g. AAPL, RELIANCE.NS, 7203.T, SHEL.L)")
        print(f"  {Fore.YELLOW}q.{Style.RESET_ALL} Quit")
        print()
        ch = input("  >> ").strip().lower()

        if ch == 'q':
            return None, None

        elif ch == '1':
            result = search_from_list()
            if result:
                return result

        elif ch == '2':
            result = enter_custom_ticker()
            if result:
                return result

        else:
            print(Fore.RED + "  Invalid choice." + Style.RESET_ALL)
            input("  Press Enter to try again...")


def search_from_list():
    while True:
        clear()
        header()
        print(Fore.YELLOW + "  Type to search (company name or ticker), or press Enter to list all:" + Style.RESET_ALL)
        print(Fore.WHITE  + "  ('b' = go back)\n" + Style.RESET_ALL)
        query = input("  >> ").strip()

        if query.lower() == 'b':
            return None

        q = query.lower()
        matches = [n for n in SORTED_NAMES if q in n.lower() or q in POPULAR[n].lower()] if q else SORTED_NAMES

        if not matches:
            print(Fore.RED + "\n  No match found. Try a different term." + Style.RESET_ALL)
            input("  Press Enter to try again...")
            continue

        print(f"\n  {len(matches)} result(s):\n")
        for i, name in enumerate(matches, 1):
            ticker = POPULAR[name]
            if ".NS" in ticker:
                tag = Fore.GREEN   + "[NSE India]" + Style.RESET_ALL
            elif any(x in ticker for x in [".KS", ".HK", ".T", ".L"]):
                tag = Fore.MAGENTA + "[Intl]"      + Style.RESET_ALL
            else:
                tag = Fore.BLUE    + "[US]"         + Style.RESET_ALL
            print(f"  {Fore.WHITE}{i:>3}.{Style.RESET_ALL}  {name:<38} {Fore.CYAN}{ticker:<16}{Style.RESET_ALL} {tag}")

        print()
        print(Fore.YELLOW + "  Enter number to select  |  'b' = back  |  's' = search again" + Style.RESET_ALL)
        ch = input("  >> ").strip().lower()

        if ch == 'b':
            return None
        if ch == 's':
            continue
        try:
            idx = int(ch) - 1
            if 0 <= idx < len(matches):
                name = matches[idx]
                return name, POPULAR[name]
            print(Fore.RED + "  Number out of range." + Style.RESET_ALL)
            input("  Press Enter...")
        except ValueError:
            print(Fore.RED + "  Please enter a valid number." + Style.RESET_ALL)
            input("  Press Enter...")


def enter_custom_ticker():
    while True:
        clear()
        header()
        print(Fore.WHITE + Style.BRIGHT + "  Enter any stock ticker symbol:" + Style.RESET_ALL)
        print()
        print(Fore.WHITE + "  Examples:" + Style.RESET_ALL)
        print("   US stocks     :  AAPL  TSLA  MSFT  NVDA  JPM")
        print("   India (NSE)   :  RELIANCE.NS  TCS.NS  HDFCBANK.NS")
        print("   India (BSE)   :  RELIANCE.BO  TCS.BO")
        print("   UK stocks     :  SHEL.L  AZN.L  HSBA.L")
        print("   Japan         :  7203.T  6758.T  9984.T")
        print("   Germany       :  SAP.DE  BMW.DE  VOW3.DE")
        print("   Hong Kong     :  0700.HK  9988.HK")
        print("   Canada        :  SHOP.TO  RY.TO")
        print("   Australia     :  CBA.AX  BHP.AX")
        print("   Any exchange  :  just use Yahoo Finance ticker format")
        print()
        print(Fore.YELLOW + "  ('b' = back)" + Style.RESET_ALL)
        ticker = input("\n  Ticker >> ").strip().upper()

        if ticker.lower() == 'b' or ticker == 'B':
            return None

        if not ticker:
            print(Fore.RED + "  Please enter a ticker." + Style.RESET_ALL)
            input("  Press Enter...")
            continue

        print(Fore.CYAN + f"\n  Validating {ticker}..." + Style.RESET_ALL)
        try:
            info = yf.Ticker(ticker).info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if not price:
                raise ValueError("No price found.")
            name = info.get("longName") or info.get("shortName") or ticker
            print(Fore.GREEN + f"  Found: {name}  —  ${price:.2f}" + Style.RESET_ALL)
            input("  Press Enter to analyze...")
            return name, ticker
        except Exception as e:
            print(Fore.RED + f"\n  Could not find ticker '{ticker}'." + Style.RESET_ALL)
            print(Fore.RED + f"  Tip: Use Yahoo Finance format (e.g. RELIANCE.NS not RELIANCE)" + Style.RESET_ALL)
            print()
            print("  1. Try another ticker")
            print("  2. Go back")
            ch = input("  >> ").strip()
            if ch == '2':
                return None


# ── Main analysis ─────────────────────────────────────────────────

def analyze(company_name, ticker_symbol):
    clear()
    header()
    print(Fore.CYAN + f"  Fetching live data for {company_name} ({ticker_symbol})..." + Style.RESET_ALL)

    try:
        info = yf.Ticker(ticker_symbol).info
        price_check = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price_check:
            raise ValueError("No price data returned.")
    except Exception as e:
        print(Fore.RED + f"\n  Could not fetch data: {e}" + Style.RESET_ALL)
        input("\n  Press Enter to go back...")
        return

    name        = safe(info.get("longName") or info.get("shortName"), company_name)
    sector      = safe(info.get("sector"),             "N/A")
    industry    = safe(info.get("industry"),           "N/A")
    currency    = safe(info.get("currency"),           "USD")
    mkt_price   = safe(info.get("currentPrice") or info.get("regularMarketPrice"))
    eps         = safe(info.get("trailingEps"))
    fwd_eps     = safe(info.get("forwardEps"))
    bvps        = safe(info.get("bookValue"))
    pe          = safe(info.get("trailingPE"))
    fwd_pe      = safe(info.get("forwardPE"))
    pb          = safe(info.get("priceToBook"))
    div_rate    = safe(info.get("dividendRate"),       0)
    div_yield   = safe(info.get("dividendYield"),      0)
    beta        = safe(info.get("beta"),               1.0)
    roe         = safe(info.get("returnOnEquity"))
    npm         = safe(info.get("profitMargins"))
    rev_growth  = safe(info.get("revenueGrowth"))
    earn_growth = safe(info.get("earningsGrowth"))
    de_ratio    = safe(info.get("debtToEquity"))
    current_r   = safe(info.get("currentRatio"))
    ebitda_m    = safe(info.get("ebitdaMargins"))
    fcf         = safe(info.get("freeCashflow"))
    shares      = safe(info.get("sharesOutstanding"), 1)
    w52_high    = safe(info.get("fiftyTwoWeekHigh"))
    w52_low     = safe(info.get("fiftyTwoWeekLow"))
    mkt_cap     = safe(info.get("marketCap"))
    analyst_tp  = safe(info.get("targetMeanPrice"))
    rec         = safe(info.get("recommendationMean"))

    sym          = currency if currency else "$"
    fcf_ps       = (fcf / shares) if fcf and shares else None
    growth_est   = max(2.0, min((earn_growth * 100 if earn_growth else (rev_growth * 100 if rev_growth else 8.0)), 30.0))
    discount     = max(8.0, min(4.5 + (beta or 1.0) * 5.5, 15.0))
    industry_pe  = pe  if pe  and 5 < pe  < 60 else 18.0
    industry_pb  = pb  if pb  and 0.5 < pb < 10 else 2.5
    div_growth   = min(growth_est * 0.6, 6.0)

    clear()
    header()

    # Company header
    print(Fore.WHITE + Style.BRIGHT + f"  {name}  ({ticker_symbol})" + Style.RESET_ALL)
    print(f"  {sector}  |  {industry}  |  Currency: {currency}")
    line = ""
    if mkt_price: line += f"  Price: {Fore.YELLOW}{sym} {mkt_price:.2f}{Style.RESET_ALL}"
    if w52_low and w52_high: line += f"   52W: {sym}{w52_low:.2f} – {sym}{w52_high:.2f}"
    if mkt_cap: line += f"   Mkt cap: {sym}{mkt_cap/1e9:.1f}B"
    print(line)
    print()

    # Fundamentals
    print(Fore.WHITE + Style.BRIGHT + "  KEY FUNDAMENTALS" + Style.RESET_ALL)
    print("  " + "─" * 62)
    rows = [
        ("EPS (trailing)",    f"{sym}{eps:.2f}"          if eps         else "N/A",
         "Revenue growth",    f"{rev_growth*100:.1f}%"   if rev_growth  else "N/A"),
        ("EPS (forward)",     f"{sym}{fwd_eps:.2f}"      if fwd_eps     else "N/A",
         "Earnings growth",   f"{earn_growth*100:.1f}%"  if earn_growth else "N/A"),
        ("Book value/share",  f"{sym}{bvps:.2f}"         if bvps        else "N/A",
         "EBITDA margin",     f"{ebitda_m*100:.1f}%"     if ebitda_m    else "N/A"),
        ("P/E (trailing)",    f"{pe:.1f}x"               if pe          else "N/A",
         "Net profit margin", f"{npm*100:.1f}%"          if npm         else "N/A"),
        ("P/E (forward)",     f"{fwd_pe:.1f}x"           if fwd_pe      else "N/A",
         "ROE",               f"{roe*100:.1f}%"          if roe         else "N/A"),
        ("P/B ratio",         f"{pb:.2f}x"               if pb          else "N/A",
         "Debt / Equity",     f"{de_ratio:.2f}"          if de_ratio    else "N/A"),
        ("Dividend yield",    f"{div_yield*100:.2f}%"    if div_yield   else "Nil",
         "Current ratio",     f"{current_r:.2f}"         if current_r   else "N/A"),
        ("Beta",              f"{beta:.2f}"               if beta        else "N/A",
         "FCF per share",     f"{sym}{fcf_ps:.2f}"       if fcf_ps      else "N/A"),
    ]
    for r in rows:
        print(f"  {r[0]:<22} {Fore.CYAN}{r[1]:<14}{Style.RESET_ALL}  {r[2]:<22} {Fore.CYAN}{r[3]}{Style.RESET_ALL}")

    # Valuation models
    print()
    print(Fore.WHITE + Style.BRIGHT + "  VALUATION MODEL ESTIMATES" + Style.RESET_ALL)
    print("  " + "─" * 62)

    ddm_fv = None
    if div_rate and div_rate > 0:
        rr, g = discount / 100, div_growth / 100
        if rr > g:
            ddm_fv = div_rate * (1 + g) / (rr - g)

    models = [
        ("Graham Number", graham_number(eps, bvps) if eps and bvps else None,
                          "sqrt(22.5 x EPS x BVPS)"),
        ("DCF",           dcf_value(fcf_ps, growth_est, 3.0, discount) if fcf_ps else None,
                          f"g={growth_est:.1f}%, WACC={discount:.1f}%"),
        ("P/E method",    eps * industry_pe if eps else None,
                          f"EPS x {industry_pe:.1f}x P/E"),
        ("P/B method",    bvps * industry_pb if bvps else None,
                          f"BVPS x {industry_pb:.2f}x P/B"),
        ("DDM",           ddm_fv,
                          f"Div={sym}{div_rate}, g={div_growth:.1f}%"),
        ("PEG value",     (fwd_eps or eps) * growth_est if (fwd_eps or eps) else None,
                          "EPS x estimated growth rate"),
    ]

    valid_fvs = []
    for label, fv, note in models:
        if fv and fv > 0:
            valid_fvs.append(fv)
            print(f"  {label:<18} {color_price(fv, mkt_price):<55}  ({note})")
        else:
            print(f"  {label:<18} {'N/A':<20}  ({note}) — insufficient data")

    blended = None
    if valid_fvs:
        blended = sum(valid_fvs) / len(valid_fvs)
        print()
        print(f"  {'Blended fair value':<18} {color_price(blended, mkt_price)}")
        print(f"  {'Conservative low':<18} {color_price(min(valid_fvs), mkt_price)}")
        print(f"  {'Optimistic high':<18} {color_price(max(valid_fvs), mkt_price)}")
        if analyst_tp:
            print(f"  {'Analyst target':<18} {color_price(analyst_tp, mkt_price)}")

    # Factor scores
    print()
    print(Fore.WHITE + Style.BRIGHT + "  FACTOR SCORES  (1 = poor → 5 = excellent)" + Style.RESET_ALL)
    print("  " + "─" * 62)

    if blended and mkt_price:
        upside = (blended - mkt_price) / mkt_price * 100
        s_val  = 5 if upside > 30 else 4 if upside > 10 else 3 if upside > -10 else 2 if upside > -25 else 1
    else:
        s_val = 2.5

    factors = [
        ("Profitability (ROE)",     score_m(roe * 100 if roe else None, 20, 12, 6)),
        ("Net profit margin",       score_m(npm * 100 if npm else None, 20, 10, 5)),
        ("Revenue growth",          score_m(rev_growth * 100 if rev_growth else None, 15, 7, 2)),
        ("Earnings growth",         score_m(earn_growth * 100 if earn_growth else None, 15, 8, 2)),
        ("Debt safety",             score_m(-(de_ratio or 5), -0.3, -1.0, -2.0)),
        ("Liquidity (curr. ratio)", score_m(current_r, 2.0, 1.5, 1.0)),
        ("Low volatility (beta)",   score_m(2 - (beta or 1), 1.5, 1.0, 0.5)),
        ("Valuation vs fair value", float(s_val)),
    ]
    factors = [(n, max(1.0, min(5.0, s))) for n, s in factors]

    for fname, sc in factors:
        filled = int(sc / 5 * 28)
        bar    = "█" * filled + "░" * (28 - filled)
        col    = Fore.GREEN if sc >= 3.8 else Fore.YELLOW if sc >= 2.5 else Fore.RED
        print(f"  {fname:<30} {col}{bar}{Style.RESET_ALL}  {sc:.1f}/5")

    overall = sum(s for _, s in factors) / len(factors)

    # Verdict
    print()
    print("  " + "─" * 62)
    print(f"  Overall score : {Fore.WHITE + Style.BRIGHT}{overall:.2f} / 5.00{Style.RESET_ALL}")

    if   overall >= 4.2: verdict = "STRONG BUY  —  Significantly Undervalued"
    elif overall >= 3.5: verdict = "BUY  —  Moderately Undervalued"
    elif overall >= 2.8: verdict = "HOLD  —  Fairly Valued"
    elif overall >= 2.0: verdict = "CAUTION  —  Slight Overvaluation / Mixed Signals"
    else:                verdict = "AVOID  —  Overvalued or Weak Fundamentals"

    print(f"  Final verdict : {color_verdict(verdict)}")

    if rec:
        labels = {1:"Strong Buy", 2:"Buy", 3:"Hold", 4:"Sell", 5:"Strong Sell"}
        print(f"  Analyst view  : {Fore.YELLOW}{labels.get(round(rec), str(rec))}{Style.RESET_ALL}  (consensus {rec:.2f}/5)")

    print()
    print("  " + "=" * 62)


# ── Entry point ──────────────────────────────────────────────────

def main():
    while True:
        name, ticker = select_company()
        if name is None:
            clear()
            print(Fore.CYAN + "\n  Thank you. Goodbye!\n" + Style.RESET_ALL)
            break

        analyze(name, ticker)

        print(Fore.YELLOW + "\n  What next?" + Style.RESET_ALL)
        print("  1. Analyze another company")
        print("  2. Quit")
        ch = input("\n  >> ").strip()
        if ch != "1":
            clear()
            print(Fore.CYAN + "\n  Goodbye!\n" + Style.RESET_ALL)
            break


if __name__ == "__main__":
    main()

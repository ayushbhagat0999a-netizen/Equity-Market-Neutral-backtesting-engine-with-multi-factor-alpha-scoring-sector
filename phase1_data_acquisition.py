"""
Phase 1: Data Acquisition & Cleaning
=====================================
Downloads OHLCV (daily) + annual fundamentals from yfinance for a
multi-factor EMN backtester, then aligns them into a tidy MultiIndex
DataFrame (index: Date, Ticker).

What we need per ticker:
  - Daily OHLCV (auto_adjust=True → split/dividend adjusted)
  - Annual: Net Income, Gross Profit (income statement)
  - Annual: Book Value (Stockholders Equity), Total Assets (balance sheet)
  - Annual: Ordinary Shares Number (balance sheet) → used for MarketCap

Why annual instead of quarterly?
  yfinance only provides ~5 quarters of quarterly data but ~5 years
  of annual data.  For a backtest starting in 2020 we need the longer
  history.  Factor signals are updated once per year at rebalance.

Math conventions used:
  MarketCap_t = SharesOutstanding_latest_annual × Close_t
    (intra-year shares don't move much for these large caps)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# 1. UNIVERSE & DATE RANGE
# ─────────────────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'JPM',  'BAC',  'GS',
    'JNJ',  'PFE',  'UNH', 'MRK', 'ABBV',
    'XOM',  'CVX',
    'PG',   'KO',   'PEP',
    'DIS',  'NFLX',
    'NVDA', 'AMD',  'INTC',
    'V',    'MA',
    'HD',   'LOW',
    'WMT',  'TGT',
    'BA',   'CAT',
]

START = '2020-01-01'
END   = '2025-12-31'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, 'data')
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 1: DATA ACQUISITION & CLEANING")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# 2. DOWNLOAD OHLCV
# ─────────────────────────────────────────────────────────────────
print("\n[1/4] Downloading daily OHLCV ({} tickers)...".format(len(UNIVERSE)))
price_data = yf.download(
    UNIVERSE, start=START, end=END,
    auto_adjust=True, progress=False,
)
print("  Shape:", price_data.shape)
print("  Dates: {} -> {}".format(
    price_data.index[0].date(), price_data.index[-1].date()))

# ─────────────────────────────────────────────────────────────────
# 3. DOWNLOAD FUNDAMENTALS (annual statements)
# ─────────────────────────────────────────────────────────────────
print("\n[2/4] Downloading annual fundamentals...")

fund_rows = []

for i, ticker in enumerate(UNIVERSE, 1):
    print("  [{:2d}/{}] {}".format(i, len(UNIVERSE), ticker), end="")
    try:
        t = yf.Ticker(ticker)

        # --- Income statement (annual) ---
        fin = t.financials
        if fin is None or fin.empty:
            print("  no financials, skip")
            continue

        # --- Balance sheet (annual) ---
        bs = t.balance_sheet
        if bs is None or bs.empty:
            print("  no balance sheet, skip")
            continue

        # Locate the rows we need (field names vary by ticker)
        # Net Income
        ni_row = None
        for candidate in ["Net Income", "Net Income Common"]:
            if candidate in fin.index:
                ni_row = fin.loc[candidate]
                break
        if ni_row is None:
            print("  no Net Income, skip")
            continue

        # Gross Profit
        gp_row = fin.loc["Gross Profit"] if "Gross Profit" in fin.index else pd.Series(dtype=float)

        # Book Value = Stockholders Equity
        bv_row = None
        for candidate in ["Stockholders Equity", "Total Equity Gross Minority Interest"]:
            if candidate in bs.index:
                bv_row = bs.loc[candidate]
                break
        if bv_row is None:
            print("  no Stockholders Equity, skip")
            continue

        # Total Assets
        ta_row = bs.loc["Total Assets"] if "Total Assets" in bs.index else pd.Series(dtype=float)

        # Ordinary Shares Number (for market cap computation)
        shares_row = None
        for candidate in ["Ordinary Shares Number", "Common Stock Equity"]:
            if candidate in bs.index:
                shares_row = bs.loc[candidate]
                break

        # Iterate over fiscal-year columns
        for col in fin.columns:
            yr_end = col
            if not isinstance(yr_end, pd.Timestamp):
                continue
            fund_rows.append({
                'Ticker':       ticker,
                'Report_Date':  yr_end,
                'NetIncome':    ni_row.get(col, np.nan),
                'GrossProfit':  gp_row.get(col, np.nan) if isinstance(gp_row, pd.Series) else np.nan,
                'BookValue':    bv_row.get(col, np.nan) if col in bs.columns else np.nan,
                'TotalAssets':  ta_row.get(col, np.nan) if col in bs.columns else np.nan,
                'SharesOut':    shares_row.get(col, np.nan) if (shares_row is not None and col in bs.columns) else np.nan,
            })
        print("  ok")
    except Exception as e:
        print("  error:", e)

fund_df = pd.DataFrame(fund_rows)
print("\n  Raw rows:", len(fund_df))
print("  Unique tickers:", fund_df['Ticker'].nunique())
print("  Report dates:", sorted(fund_df['Report_Date'].unique()))

# ─────────────────────────────────────────────────────────────────
# 4. ALIGN FUNDAMENTALS TO DAILY FREQUENCY (with reporting lag)
# ─────────────────────────────────────────────────────────────────
# To avoid look-ahead bias we assume annual reports are available
# 3 months after the fiscal year end (Yahoo's "annual" data is
# typically the most recent filing).  We shift each observation
# forward by 90 days before forward-filling.

print("\n[3/4] Forward-filling fundamentals to daily...")

LAG_DAYS = 90  # reporting lag for annual statements
all_dates = price_data.index

records = []
for ticker in UNIVERSE:
    sub = fund_df[fund_df['Ticker'] == ticker].copy()
    if sub.empty:
        continue
    sub['Available_Date'] = sub['Report_Date'] + pd.Timedelta(days=LAG_DAYS)
    sub = sub.sort_values('Available_Date')

    metrics = ['NetIncome', 'GrossProfit', 'BookValue', 'TotalAssets', 'SharesOut']
    for metric in metrics:
        avail = sub.dropna(subset=[metric])
        if avail.empty:
            continue
        daily = avail.set_index('Available_Date')[metric]
        daily = daily[~daily.index.duplicated(keep='last')]
        daily = daily.reindex(all_dates, method='ffill').dropna()
        for dt, val in daily.items():
            records.append({'Date': dt, 'Ticker': ticker, 'Metric': metric, 'Value': val})

aligned = pd.DataFrame(records)

# Pivot to wide: columns = metrics, index = (Date, Ticker)
wide = aligned.pivot_table(
    index=['Date', 'Ticker'],
    columns='Metric',
    values='Value',
    aggfunc='first',
).reset_index()
# Remove the spurious column if pivot created
cols = [c for c in wide.columns if c not in ('Date', 'Ticker')]
# Flatten any MultiIndex columns
if isinstance(wide.columns, pd.MultiIndex):
    wide.columns = wide.columns.get_level_values(-1)

print("  Aligned rows in wide format:", len(wide))

# ─────────────────────────────────────────────────────────────────
# 5. BUILD MASTER DATAFRAME
# ─────────────────────────────────────────────────────────────────
print("\n[4/4] Merging price + fundamentals into master panel...")

# Stack tickers: long format from the wide price data
price_long = price_data.stack('Ticker', future_stack=True).reset_index()
price_long.columns = ['Date', 'Ticker', 'Close', 'High', 'Low', 'Open', 'Volume']

# Merge price with fundamentals
master = price_long.merge(wide, on=['Date', 'Ticker'], how='left')

# Compute MarketCap = SharesOut * Close
master['MarketCap'] = master['SharesOut'] * master['Close']

# Set MultiIndex
master = master.set_index(['Date', 'Ticker']).sort_index()

print("  Shape:", master.shape)
print("  Columns:", list(master.columns))

# ─────────────────────────────────────────────────────────────────
# 6. DEBUGGING CHECKS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DEBUGGING CHECKS")
print("=" * 60)

# Check 1: Missing data
print("\n>> Check 1: Missing values per column")
nulls = master.isnull().sum()
print(nulls[nulls > 0])

# Check 2: Sample rows (2024-10-01)
print("\n>> Check 2: Sample securities on 2024-10-01")
sample = master.xs('2024-10-01', level='Date')
print(sample[['Close', 'BookValue', 'NetIncome', 'MarketCap']].head(8))

# Check 3: Book-to-Price range (Value factor sanity)
print("\n>> Check 3: Book-to-Price ratio (Value) sample")
bp = master['BookValue'] / master['MarketCap']
print("  BP range: {:.4f} - {:.4f}".format(bp.min(), bp.max()))
print("  BP mean:  {:.4f}".format(bp.mean()))

# Check 4: MarketCap sanity
print("\n>> Check 4: MarketCap percentiles (2024-10-01)")
mc = master.xs('2024-10-01', level='Date')['MarketCap'].dropna()
print("  P50:  ${:.1f}B".format(mc.median() / 1e9))
print("  P10:  ${:.1f}B".format(mc.quantile(0.1) / 1e9))
print("  P90:  ${:.1f}B".format(mc.quantile(0.9) / 1e9))

# Check 5: No look-ahead bias (fundamentals appear after their report_date)
print("\n>> Check 5: First date with fundamental data per ticker (3 shown)")
first_valid = master.groupby('Ticker')[['NetIncome']].apply(
    lambda g: g.first_valid_index())
print(first_valid.head(3))

# ─────────────────────────────────────────────────────────────────
# 7. SAVE
# ─────────────────────────────────────────────────────────────────
master.to_csv(os.path.join(OUT_DIR, 'master_data.csv'))
master.to_parquet(os.path.join(OUT_DIR, 'master_data.parquet'))
print("\n  Saved to", OUT_DIR)
print("Phase 1 complete.  Ready for Phase 2 (Factor Engineering).")

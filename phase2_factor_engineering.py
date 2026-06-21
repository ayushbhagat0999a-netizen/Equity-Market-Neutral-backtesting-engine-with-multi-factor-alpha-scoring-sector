"""
Phase 2: Factor Engineering
============================
Constructs three factor families from the master DataFrame:

  1. Value  — Book-to-Price (B/P), Earnings Yield (E/P)
  2. Momentum — 12-month cumulative return (skip last month)
  3. Quality — ROE, Gross Profitability, Leverage (inverse)

Each raw factor is computed, then outliers are winsorized at the
1st / 99th percentile cross-sectionally.

Math conventions
----------------

Book-to-Price (Value)
    BP_t = BookValue_t / MarketCap_t
    A higher ratio means "cheaper" relative to accounting book value.

Earnings Yield (Value)
    EP_t = NetIncome_t / MarketCap_t
    A higher yield means more earnings per dollar of equity price.

12-month Momentum (ex-1m)
    MOM_t = (Close_t-21 / Close_t-252) - 1
    Skipping the most recent 21 trading days (~1 month) avoids the
    short-term reversal effect.  252 trading days ≈ 12 months.

ROE (Quality)
    ROE_t = NetIncome_t / BookValue_t
    Higher ROE signals a more profitable, capital-efficient business.

Gross Profitability (Quality)
    GP_t = GrossProfit_t / TotalAssets_t
    Higher gross profit per dollar of assets → better quality.

Leverage (Quality, inverted)
    LEV_t = TotalAssets_t / BookValue_t
    We use the inverse (1/LEV) in scoring so that lower leverage
    contributes positively to Quality.
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR  = DATA_DIR

print("=" * 60)
print("PHASE 2: FACTOR ENGINEERING")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# 1. LOAD MASTER DATA
# ─────────────────────────────────────────────────────────────────
print("\n[1/5] Loading master DataFrame...")
master = pd.read_csv(
    os.path.join(DATA_DIR, 'master_data.csv'),
    index_col=[0, 1],
    parse_dates=True,
)
master = master.sort_index()
print("  Shape:", master.shape)

# ─────────────────────────────────────────────────────────────────
# 2. COMPUTE RAW FACTOR VALUES
# ─────────────────────────────────────────────────────────────────
print("\n[2/5] Computing raw factors...")

# --- Value Factors ---
# Book-to-Price: Book equity / Market cap
#   Interpretation: BP > 1 → stock price < book value per share
master['BP'] = master['BookValue'] / master['MarketCap']

# Earnings Yield: Net income / Market cap
#   Interpretation: EP = 0.05 → $0.05 of earnings per $1 of equity
master['EP'] = master['NetIncome'] / master['MarketCap']

# --- Momentum Factor ---
# 12-month return skipping last month (21 trading days)
# We group by ticker then shift to avoid look-ahead.
def compute_momentum(group):
    close = group['Close']
    # Momentum = return from t-252 to t-21
    # Shift(21) gives us the close 1 month ago (which is today's return start)
    # Actually: we want (P_t-21 / P_t-252) - 1
    # P_t-21 = close.shift(21), P_t-252 = close.shift(252)
    mom = close.shift(21) / close.shift(252) - 1.0
    return mom

master['Momentum'] = master.groupby('Ticker', group_keys=False).apply(compute_momentum)

# --- Quality Factors ---
# ROE: Net Income / Book Value
#   Higher → company generates more profit per unit of equity
master['ROE'] = master['NetIncome'] / master['BookValue']

# Gross Profitability: Gross Profit / Total Assets
#   Higher → company has more pricing power / cost advantage
master['GP'] = master['GrossProfit'] / master['TotalAssets']

# Leverage: Total Assets / Book Value
#   Higher → more debt-financed (riskier)
master['Leverage'] = master['TotalAssets'] / master['BookValue']

print("  Factors added: BP, EP, Momentum, ROE, GP, Leverage")

# ─────────────────────────────────────────────────────────────────
# 3. WINSORIZE (clip extreme outliers)
# ─────────────────────────────────────────────────────────────────
# Winsorization replaces values below the 1st percentile with the
# 1st percentile value, and above the 99th with the 99th.
# This prevents a single extreme stock from dominating the rankings.

print("\n[3/5] Winsorizing at 1st / 99th percentile...")

FACTOR_COLS = ['BP', 'EP', 'Momentum', 'ROE', 'GP', 'Leverage']

winsorized = master[FACTOR_COLS].copy()

for col in FACTOR_COLS:
    # Compute percentiles of non-NaN values
    lo = winsorized[col].quantile(0.01)
    hi = winsorized[col].quantile(0.99)
    # Clip
    winsorized[col] = winsorized[col].clip(lo, hi)
    # Report
    n_before = master[col].notna().sum()
    n_clipped = (master[col] != winsorized[col]).sum()
    print("  {}: clipped {} values (range [{:.4f}, {:.4f}])".format(
        col, n_clipped, lo, hi))

# Store winsorized values back
for col in FACTOR_COLS:
    master[col + '_W'] = winsorized[col]

# ─────────────────────────────────────────────────────────────────
# 4. CROSS-SECTIONAL Z-SCORES
# ─────────────────────────────────────────────────────────────────
# A Z-score measures how many standard deviations a value is from
# the cross-sectional mean on a given date:
#     Z_i = (X_i - mean(X)) / std(X)
# This standardises factors so they can be combined later.
#
# For Leverage we use the NEGATIVE Z-score because lower leverage
# is better for Quality.

print("\n[4/5] Computing cross-sectional Z-scores...")

FACTOR_Z_MAP = {
    'BP_W':       'BP_Z',        # Value: higher → cheaper → good
    'EP_W':       'EP_Z',        # Value: higher → cheaper → good
    'Momentum_W': 'Momentum_Z',  # Momentum: higher → stronger trend → good
    'ROE_W':      'ROE_Z',       # Quality: higher ROE → good
    'GP_W':       'GP_Z',        # Quality: higher profitability → good
    'Leverage_W': 'Leverage_Z',  # Quality: lower leverage → good (negate later)
}

def cross_sectional_zscore(series):
    """Z-score within each date's cross-section."""
    mu = series.mean()
    sd = series.std(ddof=0)  # population std
    return (series - mu) / sd

for raw_col, z_col in FACTOR_Z_MAP.items():
    master[z_col] = master.groupby('Date', group_keys=False)[raw_col].apply(
        cross_sectional_zscore)
    print("  {} -> {}".format(raw_col, z_col))

# Negate Leverage Z-score so that lower leverage → positive score
master['Leverage_Z'] = -master['Leverage_Z']
print("  Leverage_Z negated (lower leverage = better quality)")

# ─────────────────────────────────────────────────────────────────
# 5. DEBUGGING CHECKS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DEBUGGING CHECKS")
print("=" * 60)

# Check 1: Summary stats for raw factors
print("\n>> Check 1: Raw factor descriptives (winsorized)")
desc = master[FACTOR_COLS].describe().transpose()
print(desc[['count', 'mean', 'std', 'min', 'max']].to_string())

# Check 2: Z-score distributions (should be ~N(0,1) each day)
print("\n>> Check 2: Z-score moment cross-section (2024-10-01)")
z_cols = [v for v in FACTOR_Z_MAP.values()]
z_sample = master.xs('2024-10-01', level='Date')[z_cols]
print(z_sample.describe().loc[['mean', 'std']].to_string())

# Check 3: Correlation between factors
print("\n>> Check 3: Factor Z-score correlation matrix")
corr = master[z_cols].corr()
print(corr.to_string())

# Check 4: Momentum looks reasonable
print("\n>> Check 4: Momentum range by ticker (2024-10-01)")
mom = master.xs('2024-10-01', level='Date')[['Close', 'Momentum_W']].dropna()
top3 = mom.nlargest(3, 'Momentum_W')
bot3 = mom.nsmallest(3, 'Momentum_W')
print("  Top 3 momentum:")
print(top3.to_string())
print("  Bottom 3 momentum:")
print(bot3.to_string())

# Check 5: BP and EP sanity
print("\n>> Check 5: BP (Value) top/bottom 3 stocks (2024-10-01)")
bp = master.xs('2024-10-01', level='Date')[['BP_W']].dropna()
print("  Cheapest (highest BP):")
print(bp.nlargest(3, 'BP_W').to_string())
print("  Most expensive (lowest BP):")
print(bp.nsmallest(3, 'BP_W').to_string())

# ─────────────────────────────────────────────────────────────────
# 6. SAVE
# ─────────────────────────────────────────────────────────────────
# Keep only the columns we need going forward (price + factors)
KEEP_COLS = [
    'Close', 'Open', 'High', 'Low', 'Volume', 'MarketCap',
] + FACTOR_COLS + [v for v in FACTOR_Z_MAP.values()]

output = master[KEEP_COLS].copy()
output.to_csv(os.path.join(OUT_DIR, 'factors.csv'))
output.to_parquet(os.path.join(OUT_DIR, 'factors.parquet'))
print("\n  Saved factors to", OUT_DIR)
print("  Columns:", list(output.columns))
print("Phase 2 complete.  Ready for Phase 3 (Standardization & Scoring).")

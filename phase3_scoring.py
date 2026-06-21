"""
Phase 3: Standardization & Scoring (Alpha Combination)
=======================================================
Combines individual factor Z-scores into composite alpha signals.

Strategy (AQR-style EMN):
  - Equal-weight sub-factors within each theme:
      Value   = 0.5 * BP_Z + 0.5 * EP_Z
      Quality = 1/3 * ROE_Z + 1/3 * GP_Z + 1/3 * (-Leverage_Z)
      Momentum = Momentum_Z  (already scored)
  - Equal-weight across themes:
      Alpha = (Value + Quality + Momentum) / 3

Then we standardise Alpha to Z-scores daily so the cross-sectional
mean is 0 and std is 1 — this makes position sizing consistent
through time.

Math reminder:
    Z_i = (X_i - mean(X)) / std(X)
    A Z-score of +1.5 means the stock is 1.5σ above the mean.
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR  = DATA_DIR

print("=" * 60)
print("PHASE 3: STANDARDIZATION & SCORING")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# 1. LOAD FACTORS
# ─────────────────────────────────────────────────────────────────
print("\n[1/4] Loading factor data...")
factors = pd.read_csv(
    os.path.join(DATA_DIR, 'factors.csv'),
    index_col=[0, 1],
    parse_dates=True,
)
print("  Shape:", factors.shape)

# ─────────────────────────────────────────────────────────────────
# 2. COMPOSITE THEME SCORES
# ─────────────────────────────────────────────────────────────────
print("\n[2/4] Computing composite theme scores...")

# Value composite: average of Book Yield and Earnings Yield
# Both capture "cheapness" but from different angles.
factors['Value_Score'] = 0.5 * factors['BP_Z'] + 0.5 * factors['EP_Z']

# Quality composite: average of ROE, Gross Profitability, and
# inverse Leverage.  Together they capture profitability, efficiency,
# and capital structure.
factors['Quality_Score'] = (
    1/3 * factors['ROE_Z'] +
    1/3 * factors['GP_Z'] +
    1/3 * factors['Leverage_Z']   # already negated in Phase 2
)

# Momentum stands alone
factors['Momentum_Score'] = factors['Momentum_Z']

print("  Value_Score   = 0.5*BP_Z + 0.5*EP_Z")
print("  Quality_Score = 1/3*ROE_Z + 1/3*GP_Z + 1/3*Leverage_Z")
print("  Momentum_Score = Momentum_Z")

# ─────────────────────────────────────────────────────────────────
# 3. FINAL ALPHA (equal-weight across themes)
# ─────────────────────────────────────────────────────────────────
print("\n[3/4] Computing final Alpha score...")

# Raw Alpha = equal-weighted theme combination
factors['Alpha_Raw'] = (
    1/3 * factors['Value_Score'] +
    1/3 * factors['Quality_Score'] +
    1/3 * factors['Momentum_Score']
)

# Re-standardise Alpha to Z-scores daily so the long/short
# legs are balanced around zero every day.
def zscore(series):
    return (series - series.mean()) / series.std(ddof=0)

factors['Alpha_Z'] = factors.groupby('Date', group_keys=False)['Alpha_Raw'].apply(zscore)

print("  Alpha_Raw = (Value + Quality + Momentum) / 3")
print("  Alpha_Z = daily cross-sectional Z-score of Alpha_Raw")

# ─────────────────────────────────────────────────────────────────
# 4. DEBUGGING CHECKS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DEBUGGING CHECKS")
print("=" * 60)

# Check 1: Theme score cross-section summary
print("\n>> Check 1: Theme scores distribution (2024-10-01)")
theme_cols = ['Value_Score', 'Quality_Score', 'Momentum_Score']
sample = factors.xs('2024-10-01', level='Date')[theme_cols + ['Alpha_Z']]
print(sample.describe().to_string())

# Check 2: Top / bottom stocks by Alpha
print("\n>> Check 2: Top 5 (long candidates) vs Bottom 5 (short candidates)")
ranked = sample[['Alpha_Z']].copy()
ranked['Rank'] = ranked['Alpha_Z'].rank(ascending=False)
print("  Long candidates (highest Alpha):")
print(ranked.nsmallest(5, 'Rank')[['Alpha_Z']].to_string())
print("\n  Short candidates (lowest Alpha):")
print(ranked.nlargest(5, 'Rank')[['Alpha_Z']].to_string())

# Check 3: Theme correlations
print("\n>> Check 3: Theme score correlation matrix")
corr = factors[theme_cols + ['Alpha_Z']].corr()
print(corr.to_string())

# Check 4: Alpha_Z distribution over time (should be ~N(0,1) each day)
print("\n>> Check 4: Alpha_Z time-series mean & std")
alpha_stats = factors.groupby('Date')['Alpha_Z'].agg(['mean', 'std'])
print("  Mean across dates:  {:.6f} (should be ~0)".format(alpha_stats['mean'].mean()))
print("  Std across dates:   {:.6f} (should be ~1)".format(alpha_stats['std'].mean()))
print("  Min daily mean:     {:.6f}".format(alpha_stats['mean'].min()))
print("  Max daily mean:     {:.6f}".format(alpha_stats['mean'].max()))

# Check 5: Factor exposure of Alpha (does Alpha actually load on each factor?)
print("\n>> Check 5: Alpha_Z correlation with each individual Z-score")
exposure = factors[['Alpha_Z', 'BP_Z', 'EP_Z', 'Momentum_Z', 'ROE_Z', 'GP_Z', 'Leverage_Z']].corr()['Alpha_Z']
print(exposure.to_string())

# ─────────────────────────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────────────────────────
KEEP_COLS = [
    'Close', 'Open', 'High', 'Low', 'Volume', 'MarketCap',
    'BP_Z', 'EP_Z', 'Momentum_Z', 'ROE_Z', 'GP_Z', 'Leverage_Z',
    'Value_Score', 'Quality_Score', 'Momentum_Score', 'Alpha_Z',
]
output = factors[KEEP_COLS].copy()
output.to_csv(os.path.join(OUT_DIR, 'scores.csv'))
output.to_parquet(os.path.join(OUT_DIR, 'scores.parquet'))
print("\n  Saved to", OUT_DIR)
print("Phase 3 complete.  Ready for Phase 4 (Portfolio Construction).")

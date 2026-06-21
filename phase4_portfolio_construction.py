"""
Phase 4: Portfolio Construction
================================
Converts Alpha_Z scores into a market-neutral long/short portfolio
with monthly rebalancing.

Construction rules:
  1. Rebalance on the first trading day of each calendar month.
  2. At each rebalance:
       Long leg  = stocks where Alpha_Z > 0
       Short leg = stocks where Alpha_Z < 0
  3. Weights are proportional to |Alpha_Z| within each leg, then
     scaled so that:
        sum(Long_weights)  = +1.0  (100% long exposure)
        sum(Short_weights) = -1.0  (100% short exposure)
        Net exposure = 0           (dollar neutral)
        Gross exposure = 2.0       (200%)

This is a classic market-neutral construction: long the stocks
the model likes most, short the stocks it dislikes most, in
proportion to conviction strength (|Z|).
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR  = DATA_DIR

print("=" * 60)
print("PHASE 4: PORTFOLIO CONSTRUCTION")
print("=" * 60)

# 1. LOAD SCORES
print("\n[1/5] Loading scores data...")
scores = pd.read_csv(
    os.path.join(DATA_DIR, 'scores.csv'),
    index_col=[0, 1],
    parse_dates=True,
)
scores = scores.sort_index()
print("  Shape:", scores.shape)
print("  Date range: {} -> {}".format(
    scores.index.get_level_values('Date').min().date(),
    scores.index.get_level_values('Date').max().date(),
))

# 2. IDENTIFY REBALANCE DATES
# Monthly rebalancing: first trading day of each calendar month.
print("\n[2/5] Identifying monthly rebalance dates...")

all_dates = scores.index.get_level_values('Date').unique().sort_values()

# Group by year/month and take the first date in each month
reb_idx = (
    all_dates.to_series()
    .groupby([all_dates.year, all_dates.month])
    .first()
)
# Flatten to a simple Index of Timestamps
rebalance_dates = pd.DatetimeIndex(reb_idx.values)
print("  Rebalance dates: {} total (showing first/last 3)".format(len(rebalance_dates)))
print("    First:", list(rebalance_dates[:3]))
print("    Last:",  list(rebalance_dates[-3:]))

# 3. BUILD TARGET WEIGHTS
print("\n[3/5] Computing target weights at each rebalance...")

weight_records = []

for rb_date in rebalance_dates:
    # Cross-section on this date
    cs = scores.xs(rb_date, level='Date').dropna(subset=['Alpha_Z']).copy()
    if cs.empty:
        continue

    z = cs['Alpha_Z']

    long_mask = z > 0
    short_mask = z < 0

    long_w = pd.Series(0.0, index=z.index)
    short_w = pd.Series(0.0, index=z.index)

    if long_mask.any():
        raw = z[long_mask]
        long_w[long_mask] = raw / raw.sum()

    if short_mask.any():
        raw = z[short_mask]
        short_w[short_mask] = raw / raw.abs().sum()

    w = long_w + short_w

    for ticker, weight in w.items():
        if weight != 0.0:
            weight_records.append({
                'Date': rb_date,
                'Ticker': ticker,
                'TargetWeight': weight,
                'Alpha_Z': z[ticker],
                'Leg': 'Long' if weight > 0 else 'Short',
            })

weights_df = pd.DataFrame(weight_records)

# Verify neutrality
print("\n  Verifying neutrality for each rebalance date:")
neutrality_check = (
    weights_df.groupby('Date')['TargetWeight']
    .agg(['sum', lambda x: x.abs().sum()])
    .rename(columns={'sum': 'Net', '<lambda_0>': 'Gross'})
)
print(neutrality_check.head(5).to_string())
print("  ...")
print(neutrality_check.tail(3).to_string())
print("  Net exposure should be ~0, Gross ~2.0 for every date.")

# 4. MERGE WEIGHTS BACK TO FULL PANEL
print("\n[4/5] Forward-filling weights to daily frequency...")

rb_weights = weights_df.set_index(['Date', 'Ticker'])['TargetWeight']
full_idx = scores.index
scores['TargetWeight'] = rb_weights.reindex(full_idx)
scores['TargetWeight'] = scores.groupby('Ticker')['TargetWeight'].ffill()

print("  Non-NaN weights: {} of {} rows".format(
    scores['TargetWeight'].notna().sum(), len(scores)))

# 5. DEBUGGING CHECKS
print("\n" + "=" * 60)
print("DEBUGGING CHECKS")
print("=" * 60)

# Check 1: Long/short composition over time
print("\n>> Check 1: Number of long vs short positions per rebalance")
pos_counts = weights_df.groupby(['Date', 'Leg']).size().unstack(fill_value=0)
print(pos_counts.head(10).to_string())

# Check 2: First rebalance with meaningful data
print("\n>> Check 2: Target weights (first rebalance with >5 positions)")
rich_rb = weights_df.groupby('Date').size()
rich_rb = rich_rb[rich_rb > 5].index[0]
fw = weights_df[weights_df['Date'] == rich_rb]
print("  Date: {}".format(rich_rb.date()))
print(fw[['Ticker', 'Leg', 'Alpha_Z', 'TargetWeight']].to_string())

# Check 3: Last rebalance
print("\n>> Check 3: Target weight distribution (last rebalance)")
lw = weights_df[weights_df['Date'] == rebalance_dates[-1]]
print(lw[['Ticker', 'Leg', 'Alpha_Z', 'TargetWeight']].head(10).to_string())

# Check 4: Position counts across time
print("\n>> Check 4: Long vs Short count by rebalance (sample)")
sample_dates = rebalance_dates[::6][:5]
for d in sample_dates:
    sub = weights_df[weights_df['Date'] == d]
    longs = (sub['TargetWeight'] > 0).sum()
    shorts = (sub['TargetWeight'] < 0).sum()
    print("  {}: {} long, {} short".format(d.date(), longs, shorts))

# Check 5: Weight forward-fill example
print("\n>> Check 5: AAPL weight forward-fill (first 10 non-NaN)")
aapl = scores.xs('AAPL', level='Ticker')[['Alpha_Z', 'TargetWeight']].dropna()
print(aapl.head(10).to_string())

# 6. SAVE
scores[['Close', 'MarketCap', 'Alpha_Z', 'TargetWeight']].to_csv(
    os.path.join(OUT_DIR, 'portfolios.csv'))
scores[['Close', 'MarketCap', 'Alpha_Z', 'TargetWeight']].to_parquet(
    os.path.join(OUT_DIR, 'portfolios.parquet'))

weights_df.to_csv(os.path.join(OUT_DIR, 'rebalance_weights.csv'), index=False)
print("\n  Saved to", OUT_DIR)
print("Phase 4 complete.  Ready for Phase 5 (Backtesting Engine).")

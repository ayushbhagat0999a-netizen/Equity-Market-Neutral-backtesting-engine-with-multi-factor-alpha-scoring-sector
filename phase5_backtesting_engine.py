"""
Phase 5: The Backtesting Engine
================================
Applies lagged target weights to forward stock returns, accounting
for transaction costs at each rebalance.

Pipeline:
  1. Compute daily stock returns (pct_change).
  2. Shift weights by 1 day: weight set at close of rebalance date
     is applied to returns starting the NEXT day (no look-ahead).
  3. Daily portfolio return = dot-product of live weights with
     stock returns across each date's cross-section.
  4. At each rebalance, subtract transaction cost:
       cost = one-way_turnover * tc_rate
       one-way_turnover = 0.5 * sum(|w_new - w_old|)
     Default tc_rate = 0.001 (10 bps), realistic for large caps.
  5. Cumulative equity curve = cumprod(1 + daily_return).

Math reminder:
    R_{p,t} = sum_i ( w_{i,t-1} * R_{i,t} )
    where w_{i,t-1} is the weight set at the most recent rebalance
    before day t.
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
OUT_DIR   = DATA_DIR
TC_RATE   = 0.001     # 10 bps transaction cost per unit of turnover
START_CAP = 1.0       # start with $1 (normalised)

print("=" * 60)
print("PHASE 5: BACKTESTING ENGINE")
print("=" * 60)
print("  TC_RATE = {} ({} bps)".format(TC_RATE, TC_RATE * 10000))

# ─────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────
print("\n[1/6] Loading data...")
scores = pd.read_csv(
    os.path.join(DATA_DIR, 'portfolios.csv'),
    index_col=[0, 1],
    parse_dates=True,
)
scores = scores.sort_index()
print("  Portfolios shape:", scores.shape)
print("  Columns:", list(scores.columns))

# Load rebalance weights (from Phase 4)
weights_df = pd.read_csv(os.path.join(DATA_DIR, 'rebalance_weights.csv'))
weights_df['Date'] = pd.to_datetime(weights_df['Date'])
print("  Rebalance records:", len(weights_df))

# ─────────────────────────────────────────────────────────────────
# 2. COMPUTE DAILY STOCK RETURNS
# ─────────────────────────────────────────────────────────────────
print("\n[2/6] Computing daily stock returns...")
scores['Stock_Return'] = scores.groupby('Ticker')['Close'].pct_change()
print("  Stock_Return range: [{:.4f}, {:.4f}]".format(
    scores['Stock_Return'].min(), scores['Stock_Return'].max()))

# ─────────────────────────────────────────────────────────────────
# 3. SHIFT WEIGHTS TO AVOID LOOK-AHEAD
# ─────────────────────────────────────────────────────────────────
# Weight set at close of day t is available for trading day t+1.
print("\n[3/6] Shifting weights by 1 day...")
scores['W_Live'] = scores.groupby('Ticker')['TargetWeight'].shift(1)
live_count = scores['W_Live'].notna().sum()
print("  Live weights available: {} of {} rows".format(live_count, len(scores)))

# ─────────────────────────────────────────────────────────────────
# 4. DAILY PORTFOLIO RETURN (BEFORE COSTS)
# ─────────────────────────────────────────────────────────────────
print("\n[4/6] Computing daily portfolio returns...")

def portfolio_return(day_group):
    """Dot product of live weights and stock returns for one day."""
    w = day_group['W_Live']
    r = day_group['Stock_Return']
    return (w * r).sum()

daily_ret = scores.groupby('Date').apply(portfolio_return)
daily_ret.name = 'Port_Ret_Raw'
print("  Dates with raw return: {}".format(daily_ret.notna().sum()))
print("  Mean daily return: {:.6f}".format(daily_ret.mean()))
print("  Std  daily return: {:.6f}".format(daily_ret.std()))

# ─────────────────────────────────────────────────────────────────
# 5. TRANSACTION COSTS
# ─────────────────────────────────────────────────────────────────
print("\n[5/6] Computing transaction costs at rebalance...")

# Build a mapping: Date -> dict of {Ticker: Weight} for each rebalance
rb_weights_by_date = {}
for rb_date, group in weights_df.groupby('Date'):
    rb_weights_by_date[rb_date] = group.set_index('Ticker')['TargetWeight'].to_dict()

# Sort rebalance dates
rb_dates = sorted(rb_weights_by_date.keys())
print("  Processing {} rebalance dates...".format(len(rb_dates)))

prev_w = {}   # ticker -> weight from previous rebalance
cost_records = []

for rb_date in rb_dates:
    current_w = rb_weights_by_date[rb_date]

    # One-way turnover
    all_tickers = set(prev_w.keys()) | set(current_w.keys())
    gross_turn = sum(abs(current_w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in all_tickers)
    one_way_turn = gross_turn / 2.0

    # Transaction cost
    cost = one_way_turn * TC_RATE

    cost_records.append({
        'Date': rb_date,
        'OneWayTurnover': one_way_turn,
        'Cost': cost,
    })

    prev_w = current_w

cost_df = pd.DataFrame(cost_records)
cost_df = cost_df.set_index('Date')

# Merge costs into daily returns
daily = daily_ret.to_frame()
daily['Cost'] = cost_df['Cost']
daily['Cost'] = daily['Cost'].fillna(0.0)
daily['Port_Ret'] = daily['Port_Ret_Raw'] - daily['Cost']

print("\n  Transaction cost summary:")
print("    Mean daily cost: {:.6f}".format(daily['Cost'].mean()))
print("    Total cost:      {:.4f}".format(daily['Cost'].sum()))
print("    Mean turnover:   {:.4f}".format(cost_df['OneWayTurnover'].mean()))

# ─────────────────────────────────────────────────────────────────
# 6. CUMULATIVE PERFORMANCE
# ─────────────────────────────────────────────────────────────────
print("\n[6/6] Computing equity curve...")

daily['Cumulative'] = (1 + daily['Port_Ret']).cumprod()
daily['Cumulative_Raw'] = (1 + daily['Port_Ret_Raw']).cumprod()

# Total return
total_ret = daily['Cumulative'].iloc[-1] - 1
total_ret_raw = daily['Cumulative_Raw'].iloc[-1] - 1
print("  Total return (after costs):  {:.2%}".format(total_ret))
print("  Total return (before costs): {:.2%}".format(total_ret_raw))
print("  Cost drag:                   {:.2%}".format(total_ret_raw - total_ret))

# ─────────────────────────────────────────────────────────────────
# DEBUGGING CHECKS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DEBUGGING CHECKS")
print("=" * 60)

# Check 1: First few days of the backtest
print("\n>> Check 1: First 10 trading days with returns")
active = daily[daily['Port_Ret'] != 0].head(10)
print(active.to_string())

# Check 2: Last 10 trading days
print("\n>> Check 2: Last 10 trading days")
print(daily.tail(10).to_string())

# Check 3: Summary stats
print("\n>> Check 3: Return statistics")
print(daily[['Port_Ret_Raw', 'Port_Ret']].describe())

# Check 4: Turnover over time
print("\n>> Check 4: Turnover at each rebalance (first 5, last 3)")
print(cost_df.head(5).to_string())
print("  ...")
print(cost_df.tail(3).to_string())

# Check 5: No look-ahead validation
print("\n>> Check 5: Look-ahead bias check (AAPL, rebalance month)")
aapl = scores.xs('AAPL', level='Ticker')[['Close', 'TargetWeight', 'W_Live', 'Stock_Return']]
# Show transition around a rebalance
trans = aapl.loc['2023-04-01':'2023-04-15']
print(trans.to_string())

# Check 6: Cumulative equity curve summary
print("\n>> Check 6: Equity curve milestones")
curve = daily['Cumulative']
print("  Peak:   {:.4f} on {}".format(curve.max(), curve.idxmax().date()))
print("  Trough: {:.4f} on {}".format(curve.min(), curve.idxmin().date()))
print("  Final:  {:.4f} on {}".format(curve.iloc[-1], curve.index[-1].date()))

# ─────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────
daily.to_csv(os.path.join(OUT_DIR, 'backtest_results.csv'))
cost_df.to_csv(os.path.join(OUT_DIR, 'transaction_costs.csv'))
print("\n  Saved to", OUT_DIR)
print("Phase 5 complete.  Ready for Phase 6 (Performance Metrics & Visualization).")

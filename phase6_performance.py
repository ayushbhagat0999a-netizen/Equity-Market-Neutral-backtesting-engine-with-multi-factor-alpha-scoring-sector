"""
Phase 6: Performance Metrics & Visualization
=============================================
Computes professional-grade risk/return metrics and generates plots.

Metrics computed:
  - Annualised return & volatility
  - Sharpe ratio (0% risk-free)
  - Sortino ratio (downside deviation only)
  - Maximum drawdown & Calmar ratio
  - Win rate, profit factor
  - Beta, correlation to SPY (market)
  - Rolling Sharpe (12-month window)

Visualisations:
  - Equity curve (cumulative return, log scale)
  - Underwater plot (drawdown series)
  - Rolling Sharpe ratio
  - Long / Short contribution breakdown
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
OUT_DIR    = DATA_DIR
FIGS_DIR   = os.path.join(DATA_DIR, 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)

RF_RATE    = 0.0       # risk-free rate assumption (0 for now)
N_DAYS_YR  = 252       # trading days per year

# ─────────────────────────────────────────────────────────────────
# 1. LOAD BACKTEST RESULTS & SPY
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 6: PERFORMANCE METRICS & VISUALISATION")
print("=" * 60)

print("\n[1/6] Loading backtest results...")
daily = pd.read_csv(
    os.path.join(DATA_DIR, 'backtest_results.csv'),
    index_col='Date', parse_dates=True,
)
print("  Rows:", len(daily))
print("  Date range: {} -> {}".format(
    daily.index.min().date(), daily.index.max().date()))

print("\n[2/6] Downloading SPY for market comparison...")
import yfinance as yf
spy = yf.download('SPY', start=daily.index.min(), end=daily.index.max(),
                  auto_adjust=True, progress=False)
# Flatten to Series (yf.download returns DataFrame with MultiIndex cols)
if isinstance(spy, pd.DataFrame):
    spy_close = spy['Close']
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
spy_ret = spy_close.pct_change().reindex(daily.index).fillna(0)
print("  SPY dates: {} -> {}".format(
    spy.index.min().date(), spy.index.max().date()))

# ─────────────────────────────────────────────────────────────────
# 2. COMPUTE PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────
print("\n[3/6] Computing performance metrics...")

port_ret = daily['Port_Ret']

# Annualised return (geometric)
cum = daily['Cumulative'].iloc[-1]
n_years = len(port_ret) / N_DAYS_YR
ann_ret = cum ** (1 / n_years) - 1

# Annualised volatility
ann_vol = port_ret.std() * np.sqrt(N_DAYS_YR)

# Sharpe ratio
sharpe = (ann_ret - RF_RATE) / ann_vol

# Sortino ratio (uses only negative returns for denominator)
downside = port_ret[port_ret < 0]
downside_vol = downside.std() * np.sqrt(N_DAYS_YR)
sortino = (ann_ret - RF_RATE) / downside_vol if downside_vol > 0 else np.nan

# Maximum drawdown
cumulative = daily['Cumulative']
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max
max_dd = drawdown.min()

# Calmar ratio
calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

# Win rate
win_rate = (port_ret > 0).sum() / port_ret.notna().sum()

# Profit factor
positive_sum = port_ret[port_ret > 0].sum()
negative_sum = abs(port_ret[port_ret < 0].sum())
profit_factor = positive_sum / negative_sum if negative_sum > 0 else np.nan

# Beta to SPY
beta = port_ret.cov(spy_ret) / spy_ret.var()

# Correlation to SPY
correlation = port_ret.corr(spy_ret)

# ─────────────────────────────────────────────────────────────────
# 3. PRINT METRIC SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PERFORMANCE SUMMARY")
print("=" * 60)

metrics = [
    ("Total Return", "{:.2%}".format(cum - 1)),
    ("Annualised Return", "{:.2%}".format(ann_ret)),
    ("Annualised Volatility", "{:.2%}".format(ann_vol)),
    ("Sharpe Ratio (rf=0)", "{:.3f}".format(sharpe)),
    ("Sortino Ratio", "{:.3f}".format(sortino)),
    ("Maximum Drawdown", "{:.2%}".format(max_dd)),
    ("Calmar Ratio", "{:.3f}".format(calmar)),
    ("Win Rate", "{:.2%}".format(win_rate)),
    ("Profit Factor", "{:.3f}".format(profit_factor)),
    ("Beta to SPY", "{:.3f}".format(beta)),
    ("Correlation to SPY", "{:.3f}".format(correlation)),
]

for name, val in metrics:
    print("  {:28s}  {}".format(name, val))

# ─────────────────────────────────────────────────────────────────
# 4. ROLLING METRICS
# ─────────────────────────────────────────────────────────────────
print("\n[4/6] Computing rolling metrics...")

ROLL = 63  # ~3 months of trading days

rolling_ret = port_ret.rolling(ROLL).mean() * N_DAYS_YR
rolling_vol = port_ret.rolling(ROLL).std() * np.sqrt(N_DAYS_YR)
rolling_sharpe = rolling_ret / rolling_vol
rolling_sharpe.name = 'Rolling_Sharpe'

# Store for later plotting
daily['Rolling_Sharpe'] = rolling_sharpe

# Long / short contribution (if available)
# We can reconstruct this from the raw portfolio return
long_ret = port_ret[port_ret > 0]
short_ret = port_ret[port_ret < 0]
print("  Rolling Sharpe (63d window) computed")

# ─────────────────────────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────────────────────────
print("\n[5/6] Generating plots...")

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # Style
    plt.style.use('ggplot')
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('EMN Multi-Factor Backtester — Performance Report',
                 fontsize=14, fontweight='bold')

    # --- Panel 1: Equity Curve (log scale) ---
    ax = axes[0, 0]
    ax.plot(cumulative.index, cumulative, label='Strategy', color='navy', lw=1.5)
    # Add SPY for comparison, scaled to start at 1.0
    spy_cum = (1 + spy_ret).cumprod()
    spy_cum = spy_cum / spy_cum.iloc[0]  # normalise
    ax.plot(spy_cum.index, spy_cum, label='SPY (scaled)', color='red', alpha=0.6, lw=1)
    ax.set_yscale('log')
    ax.set_title('Cumulative Return (log scale)')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(loc='upper left')
    ax.axhline(1.0, color='grey', lw=0.5, ls='--')

    # --- Panel 2: Underwater Plot (Drawdown) ---
    ax = axes[0, 1]
    ax.fill_between(drawdown.index, drawdown * 100, 0,
                     color='crimson', alpha=0.4)
    ax.plot(drawdown.index, drawdown * 100, color='crimson', lw=1)
    ax.set_title('Drawdown (%)')
    ax.set_ylabel('Drawdown (%)')
    ax.axhline(0, color='grey', lw=0.5)

    # --- Panel 3: Rolling Sharpe ---
    ax = axes[1, 0]
    ax.plot(rolling_sharpe.index, rolling_sharpe, color='darkgreen', lw=1)
    ax.axhline(0, color='grey', ls='--', lw=0.5)
    ax.axhline(sharpe, color='green', ls=':', lw=0.8, label='Full-period Sharpe')
    ax.set_title('Rolling Sharpe Ratio (63d window)')
    ax.set_ylabel('Sharpe')
    ax.legend()

    # --- Panel 4: Daily Return Distribution ---
    ax = axes[1, 1]
    ax.hist(port_ret * 100, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(0, color='red', ls='--', lw=1)
    ax.set_title('Daily Return Distribution (%)')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')

    # --- Panel 5: Monthly Returns Heatmap ---
    ax = axes[2, 0]
    monthly = port_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    monthly_pivot = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
    im = ax.imshow(monthly_pivot.values, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
    ax.set_yticks(range(len(monthly_pivot.index)))
    ax.set_yticklabels(monthly_pivot.index)
    ax.set_xticks(range(12))
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax.set_title('Monthly Returns (%)')
    # Annotate cells
    for i in range(len(monthly_pivot.index)):
        for j in range(12):
            val = monthly_pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, '{:.1%}'.format(val), ha='center', va='center',
                        fontsize=7, color='black' if abs(val) < 0.05 else 'white')

    # --- Panel 6: Metrics table ---
    ax = axes[2, 1]
    ax.axis('off')
    table_data = [
        ['Metric', 'Value'],
        ['Total Return', '{:.2%}'.format(cum - 1)],
        ['Ann. Return', '{:.2%}'.format(ann_ret)],
        ['Ann. Volatility', '{:.2%}'.format(ann_vol)],
        ['Sharpe', '{:.3f}'.format(sharpe)],
        ['Sortino', '{:.3f}'.format(sortino)],
        ['Max DD', '{:.2%}'.format(max_dd)],
        ['Calmar', '{:.3f}'.format(calmar)],
        ['Win Rate', '{:.2%}'.format(win_rate)],
        ['Beta', '{:.3f}'.format(beta)],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                     colWidths=[0.35, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('navy')
            cell.set_text_props(color='white', fontweight='bold')
    ax.set_title('Performance Summary')

    plt.tight_layout()
    fig_path = os.path.join(FIGS_DIR, 'performance_report.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("  Saved:", fig_path)
    plt.close()

except ImportError:
    print("  matplotlib not installed — skipping plots")
except Exception as e:
    print("  Plot error:", e)

# ─────────────────────────────────────────────────────────────────
# 6. SAVE METRICS
# ─────────────────────────────────────────────────────────────────
print("\n[6/6] Saving metrics...")

metrics_dict = {
    'Total Return': float(cum - 1),
    'Annualised Return': float(ann_ret),
    'Annualised Volatility': float(ann_vol),
    'Sharpe Ratio': float(sharpe),
    'Sortino Ratio': float(sortino),
    'Max Drawdown': float(max_dd),
    'Calmar Ratio': float(calmar),
    'Win Rate': float(win_rate),
    'Profit Factor': float(profit_factor),
    'Beta to SPY': float(beta),
    'Correlation to SPY': float(correlation),
}

metrics_series = pd.Series(metrics_dict)
metrics_series.to_csv(os.path.join(OUT_DIR, 'performance_metrics.csv'))
print("  Saved:", os.path.join(OUT_DIR, 'performance_metrics.csv'))

print("\n" + "=" * 60)
print("ALL 6 PHASES COMPLETE")
print("=" * 60)
print("""
  Project structure:
    emn-backtester/
    ├── phase1_data_acquisition.py
    ├── phase2_factor_engineering.py
    ├── phase3_scoring.py
    ├── phase4_portfolio_construction.py
    ├── phase5_backtesting_engine.py
    └── phase6_performance.py
    └── data\\
        ├── master_data.csv
        ├── factors.csv
        ├── scores.csv
        ├── portfolios.csv
        ├── backtest_results.csv
        ├── performance_metrics.csv
        └── figures\\
            └── performance_report.png
""")

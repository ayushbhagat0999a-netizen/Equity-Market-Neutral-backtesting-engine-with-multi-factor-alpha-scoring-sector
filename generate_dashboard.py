"""
Generate a standalone HTML dashboard for the EMN Backtester.
All charts are rendered as base64-embedded PNGs so the HTML
file is fully self-contained and shareable with recruiters.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import base64
import io
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIGS_DIR = os.path.join(DATA_DIR, 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Load data ──
daily = pd.read_csv(os.path.join(DATA_DIR, 'backtest_results.csv'),
                    index_col='Date', parse_dates=True)
metrics = pd.read_csv(os.path.join(DATA_DIR, 'performance_metrics.csv'),
                      index_col=0, header=None).squeeze('columns').to_dict()

# ── Helper: save fig to base64 ──
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return data

# ═══════════════════════════════════════════════════════════════
# CHART 1: Equity Curve + SPY Benchmark
# ═══════════════════════════════════════════════════════════════
print("Generating chart 1/5: Equity curve...")
import yfinance as yf
spy = yf.download('SPY', start=daily.index.min(), end=daily.index.max(),
                  auto_adjust=True, progress=False)
spy_close = spy['Close']
if isinstance(spy_close, pd.DataFrame):
    spy_close = spy_close.iloc[:, 0]
spy_ret = spy_close.pct_change().reindex(daily.index).fillna(0)
spy_cum = (1 + spy_ret).cumprod()
spy_cum = spy_cum / spy_cum.iloc[0]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(daily.index, daily['Cumulative'], label='EMN Strategy',
        color='#1a56db', lw=2)
ax.plot(spy_cum.index, spy_cum, label='SPY (normalised)',
        color='#e53e3e', lw=1.5, alpha=0.7)
ax.axhline(1.0, color='gray', lw=0.5, ls='--')
ax.set_yscale('log')
ax.set_title('Cumulative Return — EMN Multi-Factor Strategy vs SPY',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Portfolio Value ($1 initial)', fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
eq_curve = fig_to_b64(fig)

# ═══════════════════════════════════════════════════════════════
# CHART 2: Drawdown (Underwater)
# ═══════════════════════════════════════════════════════════════
print("Generating chart 2/5: Drawdown...")
cumulative = daily['Cumulative']
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max * 100

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.fill_between(drawdown.index, drawdown, 0, color='#e53e3e', alpha=0.3)
ax.plot(drawdown.index, drawdown, color='#c53030', lw=1)
ax.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Drawdown (%)', fontsize=11)
ax.axhline(0, color='gray', lw=0.5)
ax.grid(True, alpha=0.3)
dd_chart = fig_to_b64(fig)

# ═══════════════════════════════════════════════════════════════
# CHART 3: Rolling Sharpe (63d)
# ═══════════════════════════════════════════════════════════════
print("Generating chart 3/5: Rolling Sharpe...")
port_ret = daily['Port_Ret']
roll = 63
rolling_ret = port_ret.rolling(roll).mean() * 252
rolling_vol = port_ret.rolling(roll).std() * np.sqrt(252)
rolling_sharpe = rolling_ret / rolling_vol

full_sharpe = metrics.get('Sharpe Ratio', 0)

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(rolling_sharpe.index, rolling_sharpe, color='#2b6cb0', lw=1.5)
ax.axhline(0, color='gray', ls='--', lw=0.5)
ax.axhline(full_sharpe, color='#e53e3e', ls=':', lw=0.8,
           label='Full-period Sharpe={:.3f}'.format(full_sharpe))
ax.set_title('Rolling Sharpe Ratio (63-trading-day window)',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Sharpe Ratio', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
roll_sharpe = fig_to_b64(fig)

# ═══════════════════════════════════════════════════════════════
# CHART 4: Daily Return Distribution
# ═══════════════════════════════════════════════════════════════
print("Generating chart 4/5: Return distribution...")
fig, ax = plt.subplots(figsize=(12, 4))
ret_pct = port_ret * 100
ax.hist(ret_pct, bins=60, color='#3182ce', edgecolor='white', alpha=0.7)
ax.axvline(0, color='#e53e3e', ls='--', lw=1.5)
ax.axvline(ret_pct.mean(), color='#2f855a', ls='-', lw=1.5,
           label='Mean={:.3f}%'.format(ret_pct.mean()))
ax.set_title('Daily Return Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Daily Return (%)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ret_dist = fig_to_b64(fig)

# ═══════════════════════════════════════════════════════════════
# CHART 5: Monthly Returns Heatmap
# ═══════════════════════════════════════════════════════════════
print("Generating chart 5/5: Monthly heatmap...")
monthly = port_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
years = sorted(monthly.index.year.unique())
months = range(1, 13)
data_matrix = np.full((len(years), 12), np.nan)
for i, yr in enumerate(years):
    for m in months:
        mask = (monthly.index.year == yr) & (monthly.index.month == m)
        if mask.any():
            data_matrix[i, m-1] = monthly[mask].iloc[0]

fig, ax = plt.subplots(figsize=(10, 3.5))
cmap = plt.cm.RdYlGn
cmap.set_bad('#f0f0f0')
im = ax.imshow(data_matrix, cmap=cmap, aspect='auto',
               vmin=-0.12, vmax=0.12)
ax.set_yticks(range(len(years)))
ax.set_yticklabels(years, fontsize=10)
ax.set_xticks(range(12))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'],
                   fontsize=8, rotation=30)
ax.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold')

for i in range(len(years)):
    for j in range(12):
        val = data_matrix[i, j]
        if not np.isnan(val):
            color = 'black' if abs(val) < 0.05 else 'white'
            ax.text(j, i, '{:.1%}'.format(val), ha='center', va='center',
                    fontsize=7, color=color)
plt.colorbar(im, ax=ax, shrink=0.8)
monthly_heat = fig_to_b64(fig)

# ═══════════════════════════════════════════════════════════════
# BUILD HTML
# ═══════════════════════════════════════════════════════════════
print("Building HTML dashboard...")

# Format metrics nicely
def fmt_pct(v):
    try: return '{:.2%}'.format(float(v))
    except: return str(v)
def fmt_num(v, d=3):
    try: return '{:.{}f}'.format(float(v), d)
    except: return str(v)

metric_cards = [
    ('Total Return', fmt_pct(metrics.get('Total Return', 0)), 'The strategy grew $1 to ${:.2f}'.format(daily['Cumulative'].iloc[-1])),
    ('Ann. Return', fmt_pct(metrics.get('Annualised Return', 0)), 'Geometric annualised return'),
    ('Ann. Volatility', fmt_pct(metrics.get('Annualised Volatility', 0)), 'Annualised standard deviation of daily returns'),
    ('Sharpe Ratio', fmt_num(metrics.get('Sharpe Ratio', 0)), 'Risk-adjusted return (rf=0)'),
    ('Sortino Ratio', fmt_num(metrics.get('Sortino Ratio', 0)), 'Downside risk-adjusted return'),
    ('Max Drawdown', fmt_pct(metrics.get('Max Drawdown', 0)), 'Peak-to-trough decline'),
    ('Calmar Ratio', fmt_num(metrics.get('Calmar Ratio', 0)), 'Ann. return / |Max DD|'),
    ('Win Rate', fmt_pct(metrics.get('Win Rate', 0)), 'Fraction of positive days'),
    ('Beta to SPY', fmt_num(metrics.get('Beta to SPY', 0)), 'Market exposure (target: 0)'),
    ('Correlation to SPY', fmt_num(metrics.get('Correlation to SPY', 0)), 'Benchmark independence'),
]

# Methodology sections
phases = [
    ('Phase 1: Data Acquisition',
     ['Universe of 31 liquid US large-caps across sectors',
      'Daily OHLCV via yfinance (auto_adjust=True)',
      'Annual fundamentals: NetIncome, GrossProfit, BookValue, TotalAssets, SharesOut',
      '90-day reporting lag applied to avoid look-ahead bias',
      'Forward-filled to daily frequency → MultiIndex (Date, Ticker) panel']),
    ('Phase 2: Factor Engineering',
     ['Value: Book-to-Price (BP) and Earnings Yield (EP)',
      'Momentum: 12-month return skipping the last month (P_t-21 / P_t-252 - 1)',
      'Quality: ROE, Gross Profitability (GP/TA), and inverse Leverage',
      'Winsorized at 1st/99th percentile, then cross-sectional Z-scored daily']),
    ('Phase 3: Scoring & Alpha Combination',
     ['Value = 0.5\u00d7BP_Z + 0.5\u00d7EP_Z',
      'Quality = \u2153\u00d7ROE_Z + \u2153\u00d7GP_Z + \u2153\u00d7Leverage_Z',
      'Alpha = (Value + Quality + Momentum) / 3, re-normalised to Z daily']),
    ('Phase 4: Portfolio Construction',
     ['Monthly rebalancing on first trading day of each month',
      'Long leg: Alpha_Z > 0, Short leg: Alpha_Z < 0',
      'Weights proportional to |Alpha_Z| within each leg',
      'Long sum = +100%, Short sum = \u2212100% \u2192 dollar neutral, 200% gross']),
    ('Phase 5: Backtesting Engine',
     ['Weights shifted by 1 day to eliminate look-ahead bias',
      'Daily PnL = \u03a3(w_i \u00d7 R_i) across cross-section',
      'Transaction costs: 10 bps per unit of one-way turnover',
      'Equity curve = cumprod(1 + daily_return)']),
    ('Phase 6: Performance Metrics',
     ['Sharpe, Sortino, Max Drawdown, Calmar Ratio',
      'Rolling 63-day Sharpe, Beta to SPY, Win Rate',
      'Monthly return heatmap, return distribution']),
]

# Factor correlation table data
factor_corr_html = """
<table class="table table-sm table-bordered text-center" style="font-size:0.85rem">
<thead class="table-dark"><tr><th></th><th>BP_Z</th><th>EP_Z</th><th>Mom_Z</th><th>ROE_Z</th><th>GP_Z</th><th>Lev_Z</th></tr></thead>
<tbody>
<tr><td><strong>BP_Z</strong></td><td>1.00</td><td>0.40</td><td>-0.23</td><td>-0.28</td><td>-0.48</td><td>-0.05</td></tr>
<tr><td><strong>EP_Z</strong></td><td>0.40</td><td>1.00</td><td>-0.23</td><td>-0.08</td><td>0.21</td><td>-0.19</td></tr>
<tr><td><strong>Mom_Z</strong></td><td>-0.23</td><td>-0.23</td><td>1.00</td><td>0.00</td><td>0.16</td><td>0.01</td></tr>
<tr><td><strong>ROE_Z</strong></td><td>-0.28</td><td>-0.08</td><td>0.00</td><td>1.00</td><td>0.29</td><td>-0.71</td></tr>
<tr><td><strong>GP_Z</strong></td><td>-0.48</td><td>0.21</td><td>0.16</td><td>0.29</td><td>1.00</td><td>-0.32</td></tr>
<tr><td><strong>Lev_Z</strong></td><td>-0.05</td><td>-0.19</td><td>0.01</td><td>-0.71</td><td>-0.32</td><td>1.00</td></tr>
</tbody>
</table>
"""

# Sample stocks (top/bottom)
top_bottom_html = """
<div class="row">
<div class="col-md-6">
<h6 class="text-success">Top Long Candidates <small>(2024-10-01)</small></h6>
<table class="table table-sm table-striped"><thead><tr><th>Ticker</th><th>Alpha_Z</th></tr></thead><tbody>
<tr><td>NVDA</td><td class="text-success fw-bold">+3.28</td></tr>
<tr><td>META</td><td class="text-success fw-bold">+1.22</td></tr>
<tr><td>TGT</td><td class="text-success fw-bold">+1.18</td></tr>
<tr><td>NFLX</td><td class="text-success fw-bold">+1.00</td></tr>
<tr><td>WMT</td><td class="text-success fw-bold">+0.70</td></tr>
</tbody></table>
</div>
<div class="col-md-6">
<h6 class="text-danger">Top Short Candidates <small>(2024-10-01)</small></h6>
<table class="table table-sm table-striped"><thead><tr><th>Ticker</th><th>Alpha_Z</th></tr></thead><tbody>
<tr><td>BA</td><td class="text-danger fw-bold">-2.75</td></tr>
<tr><td>INTC</td><td class="text-danger fw-bold">-1.12</td></tr>
<tr><td>PFE</td><td class="text-danger fw-bold">-1.03</td></tr>
<tr><td>MRK</td><td class="text-danger fw-bold">-0.84</td></tr>
<tr><td>ABBV</td><td class="text-danger fw-bold">-0.71</td></tr>
</tbody></table>
</div>
</div>
"""

html = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EMN Multi-Factor Backtester — Performance Dashboard</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  body { background: #f8fafc; font-family: 'Segoe UI', -apple-system, sans-serif; }
  .header { background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%); color: white; padding: 2.5rem 0 2rem; }
  .header h1 { font-weight: 700; font-size: 2rem; }
  .header p { opacity: 0.9; font-size: 1rem; }
  .card { border: none; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 1.5rem; transition: box-shadow 0.2s; }
  .card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.12); }
  .card-header { background: white; border-bottom: 2px solid #edf2f7; font-weight: 600; font-size: 1.1rem; padding: 1rem 1.25rem; border-radius: 12px 12px 0 0 !important; }
  .metric-card { text-align: center; padding: 1.25rem 0.5rem; background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); height: 100%; }
  .metric-card .value { font-size: 1.75rem; font-weight: 700; }
  .metric-card .label { font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 0.03em; }
  .metric-card .desc { font-size: 0.72rem; color: #a0aec0; margin-top: 0.25rem; }
  .chart-container { background: white; border-radius: 12px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 1.5rem; }
  .chart-container img { width: 100%; height: auto; }
  .phase-item { padding: 0.6rem 0; border-bottom: 1px solid #edf2f7; }
  .phase-item:last-child { border-bottom: none; }
  .phase-item h6 { color: #2b6cb0; font-weight: 600; }
  .phase-item ul { margin: 0.25rem 0 0; padding-left: 1.25rem; }
  .phase-item li { font-size: 0.88rem; color: #4a5568; }
  .footer { background: #1a202c; color: #a0aec0; padding: 1.5rem 0; text-align: center; font-size: 0.85rem; margin-top: 2rem; }
  .badge-pf { font-size: 0.75rem; padding: 0.25rem 0.6rem; }
  .table-sm td, .table-sm th { padding: 0.25rem 0.5rem; }
  .green { color: #2f855a; }
  .red { color: #c53030; }
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="container">
    <h1>EMN Multi-Factor Backtester</h1>
    <p>Equity Market-Neutral strategy built from scratch with Python, pandas, NumPy, and yfinance — AQR/quantitative style</p>
    <div>
      <span class="badge bg-light text-dark me-2">31 stocks</span>
      <span class="badge bg-light text-dark me-2">3 factors</span>
      <span class="badge bg-light text-dark me-2">Monthly rebalance</span>
      <span class="badge bg-light text-dark me-2">200% gross exposure</span>
      <span class="badge bg-success me-2">Beta to SPY: ''' + fmt_num(metrics.get('Beta to SPY', 0)) + '''</span>
    </div>
  </div>
</div>

<div class="container py-4">

<!-- KEY METRICS -->
<div class="row g-3 mb-4">
'''

for label, value, desc in metric_cards:
    color = 'green' if value.startswith('+') or value.startswith('0.') or value.startswith('0,') else ('red' if value.startswith('-') else '')
    html += '''
  <div class="col-6 col-md-3 col-lg-2">
    <div class="metric-card">
      <div class="label">{}</div>
      <div class="value" style="color:{}">{}</div>
      <div class="desc">{}</div>
    </div>
  </div>
'''.format(label, '#2f855a' if '0.25' in value or '0.29' in value or '0.11' in value or '0.07' in value or ('0.0' in value and '26' not in value and '3.96' not in value and '15.72' not in value) else '#c53030' if value.startswith('-') else '#2b6cb0', value, desc)

html += '''
</div>

<!-- CHARTS ROW 1 -->
<div class="row">
  <div class="col-lg-8">
    <div class="chart-container">
      <h6 class="fw-bold mb-2">Cumulative Return (log scale)</h6>
      <img src="data:image/png;base64,''' + eq_curve + '''" alt="Equity Curve">
    </div>
  </div>
  <div class="col-lg-4">
    <div class="chart-container">
      <h6 class="fw-bold mb-2">Drawdown</h6>
      <img src="data:image/png;base64,''' + dd_chart + '''" alt="Drawdown">
    </div>
    <div class="chart-container">
      <h6 class="fw-bold mb-2">Return Distribution</h6>
      <img src="data:image/png;base64,''' + ret_dist + '''" alt="Return Distribution">
    </div>
  </div>
</div>

<!-- CHARTS ROW 2 -->
<div class="row">
  <div class="col-lg-6">
    <div class="chart-container">
      <h6 class="fw-bold mb-2">Rolling Sharpe (63d)</h6>
      <img src="data:image/png;base64,''' + roll_sharpe + '''" alt="Rolling Sharpe">
    </div>
  </div>
  <div class="col-lg-6">
    <div class="chart-container">
      <h6 class="fw-bold mb-2">Monthly Returns</h6>
      <img src="data:image/png;base64,''' + monthly_heat + '''" alt="Monthly Returns">
    </div>
  </div>
</div>

<!-- TOP/BOTTOM STOCKS + FACTOR CORRELATIONS -->
<div class="row">
  <div class="col-lg-5">
    <div class="card">
      <div class="card-header">Sample Alpha Scores</div>
      <div class="card-body">''' + top_bottom_html + '''</div>
    </div>
  </div>
  <div class="col-lg-7">
    <div class="card">
      <div class="card-header">Factor Z-Score Correlation Matrix</div>
      <div class="card-body">''' + factor_corr_html + '''
        <p class="text-muted small mt-2 mb-0">Value and Momentum negatively correlated (<span class="red">-0.23</span>) — classic value vs growth. Leverage and ROE strongly negatively correlated (<span class="red">-0.71</span>).</p>
      </div>
    </div>
  </div>
</div>

<!-- METHODOLOGY -->
<div class="card mt-3">
  <div class="card-header">Methodology — 6-Phase Pipeline</div>
  <div class="card-body">
    <div class="row">
'''

for i, (title, items) in enumerate(phases):
    html += '''
      <div class="col-md-6 col-lg-4">
        <div class="phase-item">
          <h6>''' + title + '''</h6>
          <ul>
'''
    for item in items:
        html += '            <li>' + item + '</li>\n'
    html += '          </ul>\n        </div>\n      </div>\n'

html += '''
    </div>
  </div>
</div>

<!-- PROJECT STRUCTURE -->
<div class="card mt-3">
  <div class="card-header">Project Structure</div>
  <div class="card-body">
    <div class="row">
      <div class="col-md-6">
        <h6 class="fw-bold">Python Scripts</h6>
        <ul class="list-unstyled small">
          <li><code>phase1_data_acquisition.py</code> — Download & clean OHLCV + fundamentals</li>
          <li><code>phase2_factor_engineering.py</code> — Compute Value, Momentum, Quality factors</li>
          <li><code>phase3_scoring.py</code> — Z-score standardisation & alpha combination</li>
          <li><code>phase4_portfolio_construction.py</code> — Market-neutral long/short weights</li>
          <li><code>phase5_backtesting_engine.py</code> — Lagged returns, transaction costs, equity curve</li>
          <li><code>phase6_performance.py</code> — Sharpe, drawdown, metrics & visualisation</li>
        </ul>
      </div>
      <div class="col-md-6">
        <h6 class="fw-bold">Data Files (data/)</h6>
        <ul class="list-unstyled small">
          <li><code>master_data.parquet</code> — Raw OHLCV + fundamentals panel</li>
          <li><code>factors.parquet</code> — Raw and winsorised factor values</li>
          <li><code>scores.parquet</code> — Z-scores, theme composites, Alpha</li>
          <li><code>portfolios.parquet</code> — Target weights with forward-fill</li>
          <li><code>backtest_results.csv</code> — Daily PnL, costs, equity curve</li>
          <li><code>performance_metrics.csv</code> — All computed risk/return metrics</li>
        </ul>
      </div>
    </div>
  </div>
</div>

</div>

<!-- FOOTER -->
<div class="footer">
  <div class="container">
    Built with Python 3.12, pandas 3.0, numpy, yfinance, matplotlib<br>
    Project: github.com/yourusername/emn-backtester &bull; ''' + str(len(daily)) + ''' trading days &bull; ''' + str(daily.index[-1].date()) + '''
  </div>
</div>

</body>
</html>
'''

# ── Write HTML ──
out_path = os.path.join(FIGS_DIR, 'dashboard.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print("Dashboard written to:", out_path)

print("Dashboard saved to:", out_path)
print("\nOpen 'data/figures/dashboard.html' in any browser to view.")
print("To share, just send the folder or zip it.")

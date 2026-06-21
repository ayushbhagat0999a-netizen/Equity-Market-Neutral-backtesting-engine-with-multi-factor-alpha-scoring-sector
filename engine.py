"""
Backtest engine for the Flask webapp.
Loads pre-computed factor Z-scores and runs the long/short construction + PnL
in-memory so the user can tweak weights without re-running Phases 1-3.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')

TICKER_SECTOR = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'META': 'Tech',
    'NVDA': 'Tech', 'AMD': 'Tech', 'INTC': 'Tech',
    'AMZN': 'ConsDisc', 'NFLX': 'ConsDisc', 'DIS': 'ConsDisc',
    'HD': 'ConsDisc', 'LOW': 'ConsDisc', 'TGT': 'ConsDisc',
    'JPM': 'Fin', 'BAC': 'Fin', 'GS': 'Fin', 'V': 'Fin', 'MA': 'Fin',
    'JNJ': 'Health', 'MRK': 'Health', 'PFE': 'Health', 'ABBV': 'Health', 'UNH': 'Health',
    'PG': 'ConsStap', 'KO': 'ConsStap', 'PEP': 'ConsStap', 'WMT': 'ConsStap',
    'XOM': 'Energy', 'CVX': 'Energy',
    'CAT': 'Indust', 'BA': 'Indust',
}


def load_factor_panel():
    """Read the saved cross-sectional Z-score panel."""
    path = os.path.join(DATA_DIR, 'scores.csv')
    panel = pd.read_csv(path, index_col=[0, 1], parse_dates=True)
    return panel.sort_index()


def compute_composite_alpha(factor_panel, w_val=0.33, w_mom=0.33, w_qual=0.34):
    """Weighted sum of three factor families, then cross-sectional z-scored daily."""
    if 'Value_Score' in factor_panel.columns:
        value_scores = factor_panel['Value_Score']
        mom_scores = factor_panel['Momentum_Score']
        qual_scores = factor_panel['Quality_Score']
    else:
        value_scores = 0.5 * factor_panel['BP_Z'] + 0.5 * factor_panel['EP_Z']
        mom_scores = factor_panel['Momentum_Z']
        qual_scores = (
            (1/3) * factor_panel['ROE_Z']
            + (1/3) * factor_panel['GP_Z']
            + (1/3) * factor_panel['Leverage_Z']
        )

    raw_alpha = w_val * value_scores + w_mom * mom_scores + w_qual * qual_scores

    def _zscore(s):
        return (s - s.mean()) / s.std(ddof=0)

    return raw_alpha.groupby('Date', group_keys=False).apply(_zscore)


def build_long_short_weights(alpha_scores, rebalance_freq='M', sector_neutral=False):
    """Dollar-neutral long/short weights, optionally sector-neutral.

    Without sector neutrality: long $1, short $1 across the full cross-section.
    With sector neutrality: each sector gets an equal allocation, and within
    each sector long/short positions are dollar-neutral.
    """
    all_trading_days = (
        alpha_scores.index
        .get_level_values('Date')
        .unique()
        .sort_values()
    )

    if rebalance_freq == 'M':
        anchor = all_trading_days.to_series().groupby(
            [all_trading_days.year, all_trading_days.month]
        ).first()
    elif rebalance_freq == 'Q':
        anchor = all_trading_days.to_series().groupby(
            [all_trading_days.year, all_trading_days.quarter]
        ).first()
    else:
        anchor = all_trading_days

    records = []
    for rb_date in anchor:
        cs = alpha_scores.xs(rb_date, level='Date').dropna()
        if cs.empty:
            continue

        w = pd.Series(0.0, index=cs.index)

        if sector_neutral:
            ticker_sectors = {}
            for t in cs.index:
                ticker_sectors[t] = TICKER_SECTOR.get(t, 'Other')
            sectors = {}
            for t, sec in ticker_sectors.items():
                sectors.setdefault(sec, []).append(t)
            n_sec = len(sectors)
            sec_frac = 1.0 / n_sec if n_sec > 0 else 1.0

            for sec, tickers in sectors.items():
                sec_cs = cs.loc[[t for t in tickers if t in cs.index]]
                if sec_cs.empty:
                    continue
                pos = sec_cs > 0
                neg = sec_cs < 0
                if pos.any():
                    w.loc[sec_cs[pos].index] = (sec_cs[pos] / sec_cs[pos].sum()) * sec_frac
                if neg.any():
                    w.loc[sec_cs[neg].index] = (sec_cs[neg] / sec_cs[neg].abs().sum()) * (-sec_frac)
        else:
            pos = cs > 0
            neg = cs < 0
            if pos.any():
                w[pos] = cs[pos] / cs[pos].sum()
            if neg.any():
                w[neg] = cs[neg] / cs[neg].abs().sum()

        for ticker, weight in w.items():
            if weight != 0.0:
                records.append(
                    {'Date': rb_date, 'Ticker': ticker, 'TargetWeight': weight}
                )

    if not records:
        return pd.Series(dtype=float)

    weights_t0 = (
        pd.DataFrame(records)
        .set_index(['Date', 'Ticker'])['TargetWeight']
    )

    ffill = weights_t0.reindex(alpha_scores.index)
    return ffill.groupby('Ticker', group_keys=False).apply(lambda g: g.ffill())


def compute_factor_attribution(pnl_ts, factor_panel, daily_returns, factor_exposure_df, has_theme):
    """Return contribution per factor via daily cross-sectional regression.

    Each day: stock_return ~ b_val*val_z + b_mom*mom_z + b_qual*qual_z (no intercept).
    Portfolio factor contribution = b_factor * portfolio factor exposure.
    Returns dict with dates and cumulative Value, Momentum, Quality, Specific arrays.
    """
    attr_dates = []
    cum_val, cum_mom, cum_qual, cum_spec = 0.0, 0.0, 0.0, 0.0
    val_arr, mom_arr, qual_arr, spec_arr = [], [], [], []

    for d in daily_returns.index:
        cross = pnl_ts.xs(d, level='Date')
        w = cross['W_Live'].fillna(0)
        if w.abs().sum() == 0:
            continue

        X, y = [], []
        for t in w.index:
            try:
                row = factor_panel.loc[(d, t)]
            except KeyError:
                continue
            if has_theme:
                vz = row.get('Value_Score', 0)
                mz = row.get('Momentum_Score', 0)
                qz = row.get('Quality_Score', 0)
            else:
                vz = 0.5 * row.get('BP_Z', 0) + 0.5 * row.get('EP_Z', 0)
                mz = row.get('Momentum_Z', 0)
                qz = (1/3)*row.get('ROE_Z', 0) + (1/3)*row.get('GP_Z', 0) + (1/3)*row.get('Leverage_Z', 0)
            ret = cross.loc[t, 'Stock_Ret']
            if pd.isna(ret):
                ret = 0.0
            X.append([vz, mz, qz])
            y.append(ret)

        if len(X) < 3:
            continue

        X = np.array(X)
        y = np.array(y)

        try:
            b = np.linalg.lstsq(X, y, rcond=None)[0]
        except Exception:
            b = np.zeros(3)

        fe_row = factor_exposure_df.loc[d] if d in factor_exposure_df.index else None
        if fe_row is not None:
            vc = b[0] * fe_row.get('Value', 0)
            mc = b[1] * fe_row.get('Momentum', 0)
            qc = b[2] * fe_row.get('Quality', 0)
        else:
            vc = mc = qc = 0.0

        port_ret = daily_returns.loc[d, 'Port_Ret']
        sc = port_ret - vc - mc - qc

        cum_val += vc
        cum_mom += mc
        cum_qual += qc
        cum_spec += sc

        attr_dates.append(d)
        val_arr.append(float(cum_val))
        mom_arr.append(float(cum_mom))
        qual_arr.append(float(cum_qual))
        spec_arr.append(float(cum_spec))

    return {
        'dates': [str(d.date()) for d in attr_dates],
        'value': val_arr,
        'momentum': mom_arr,
        'quality': qual_arr,
        'specific': spec_arr,
    }


def run_backtest(factor_panel, w_val=0.33, w_mom=0.33, w_qual=0.34,
                 tc_rate=0.001, rf_rate=0.0, rebalance_freq='M',
                 sector_neutral=False, max_turnover=1.0):
    """Full end-to-end: composite alpha long/short daily PnL risk metrics.

    Parameters
    ----------
    sector_neutral : bool
        When True, weight construction enforces dollar-neutrality per sector.
    max_turnover : float
        Maximum allowed one-way turnover per rebalance (0.0 = no trading, 1.0 = no limit).
    """
    pnl_ts = factor_panel[['Close']].copy()
    has_theme = 'Value_Score' in factor_panel.columns

    # 1. Composite alpha, cross-sectionally standardised
    alpha_scores = compute_composite_alpha(factor_panel, w_val, w_mom, w_qual)
    pnl_ts['Alpha_Z'] = alpha_scores

    # 2. Long / short weights, forward-filled to daily
    target_weights = build_long_short_weights(alpha_scores, rebalance_freq, sector_neutral)
    pnl_ts['TargetWeight'] = target_weights

    # 3. Daily stock returns & lagged weight
    pnl_ts['Stock_Ret'] = pnl_ts.groupby('Ticker')['Close'].pct_change()
    pnl_ts['W_Live'] = pnl_ts.groupby('Ticker')['TargetWeight'].shift(1)

    # 4. Daily portfolio return
    def _dot_product(g):
        return (g['W_Live'] * g['Stock_Ret']).sum()
    raw_daily_pnl = pnl_ts.groupby('Date').apply(_dot_product)
    raw_daily_pnl.name = 'Raw_Port_Ret'

    # 5. Transaction costs at rebalance dates, with optional turnover constraint
    rebalance_dates = sorted(
        target_weights.dropna().index.get_level_values('Date').unique()
    )
    cost_by_date = {}
    prev_snapshot = {}
    for d in rebalance_dates:
        cross = pnl_ts.xs(d, level='Date')['TargetWeight'].dropna()
        cur = cross.to_dict()
        all_tickers = set(prev_snapshot.keys()) | set(cur.keys())
        raw_turnover = (
            sum(abs(cur.get(t, 0) - prev_snapshot.get(t, 0))
                for t in all_tickers) / 2
        )

        if max_turnover < 1.0 and raw_turnover > max_turnover and prev_snapshot:
            lam = max_turnover / raw_turnover
            applied = {}
            for t in all_tickers:
                applied[t] = prev_snapshot.get(t, 0) + (cur.get(t, 0) - prev_snapshot.get(t, 0)) * lam
            cur = applied
            cost_by_date[d] = max_turnover * tc_rate
        else:
            cost_by_date[d] = raw_turnover * tc_rate

        prev_snapshot = cur

    daily = raw_daily_pnl.to_frame()
    daily['Cost'] = pd.Series(cost_by_date).reindex(daily.index).fillna(0)
    daily['Port_Ret'] = (daily['Raw_Port_Ret'] - daily['Cost']).fillna(0.0)

    # 6. Equity curve
    daily['Cumulative'] = (1 + daily['Port_Ret']).cumprod().fillna(1.0)

    # 7. Performance metrics
    n_trading_days = len(daily)
    n_years = n_trading_days / 252
    final_cumul = daily['Cumulative'].iloc[-1]

    ann_ret = final_cumul ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = daily['Port_Ret'].std() * np.sqrt(252)
    sharpe = (ann_ret - rf_rate) / ann_vol if ann_vol > 0 else 0

    running_peak = daily['Cumulative'].cummax()
    drawdown = (daily['Cumulative'] - running_peak) / running_peak
    max_dd = drawdown.min()

    neg_rets = daily['Port_Ret'][daily['Port_Ret'] < 0]
    downside_vol = neg_rets.std() * np.sqrt(252) if len(neg_rets) > 0 else 0
    sortino = (ann_ret - rf_rate) / downside_vol if downside_vol > 0 else 0

    win_rate = (daily['Port_Ret'] > 0).sum() / n_trading_days
    gross_profit = daily['Port_Ret'][daily['Port_Ret'] > 0].sum()
    gross_loss = abs(daily['Port_Ret'][daily['Port_Ret'] < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # 8. Market comparison (SPY via yfinance)
    try:
        import yfinance as yf
        spy = yf.download('SPY', start=daily.index[0],
                          end=daily.index[-1], auto_adjust=True, progress=False)
        sc = spy['Close']
        if isinstance(sc, pd.DataFrame):
            sc = sc.iloc[:, 0]
        spy_daily_ret = sc.pct_change().reindex(daily.index).fillna(0)
        beta = daily['Port_Ret'].cov(spy_daily_ret) / spy_daily_ret.var()

        ann_spy = (
            (1 + spy_daily_ret).prod() ** (1 / n_years) - 1
            if n_years > 0 else 0
        )
        jensen_alpha = ann_ret - (rf_rate + beta * (ann_spy - rf_rate))

        active = daily['Port_Ret'] - spy_daily_ret
        tracking_err = active.std() * np.sqrt(252)
        info_ratio = (ann_ret - ann_spy) / tracking_err if tracking_err > 0 else 0
    except Exception:
        beta = 0.0
        ann_spy = 0.0
        jensen_alpha = 0.0
        info_ratio = 0.0

    metrics = {
        'Total Return': final_cumul - 1,
        'Annualised Return': ann_ret,
        'Annualised Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd,
        'Calmar Ratio': calmar,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Beta to SPY': beta,
        'Alpha (Jensen)': jensen_alpha,
        'Information Ratio': info_ratio,
        'Benchmark Return': ann_spy,
    }

    # 9. Factor loadings over time (portfolio-weighted avg Z-score)
    factor_sources = {
        'Value': 'Value_Score',
        'Momentum': 'Momentum_Score',
        'Quality': 'Quality_Score',
    }
    factor_records = []
    for date, group in pnl_ts.groupby('Date'):
        w = group['W_Live'].fillna(0)
        if w.abs().sum() == 0:
            continue
        row = {'Date': date}
        for label, col in factor_sources.items():
            if col in factor_panel.columns:
                vals = factor_panel.loc[
                    (date, group.index.get_level_values('Ticker')), col
                ]
                vals.index = group.index
                row[label] = (w * vals.fillna(0)).sum()
        factor_records.append(row)

    factor_loadings = (
        pd.DataFrame(factor_records).set_index('Date')
        if factor_records else pd.DataFrame()
    )

    # 10. Gross / Net / Long / Short exposure over time
    exposure_bars = []
    for date, group in pnl_ts.groupby('Date'):
        w = group['W_Live'].fillna(0)
        long_ex = w[w > 0].sum()
        short_ex = w[w < 0].sum()
        exposure_bars.append({
            'Date': date,
            'Net': long_ex + short_ex,
            'Gross': w.abs().sum(),
            'Long': long_ex,
            'Short': short_ex,
        })
    exposure_ts = (
        pd.DataFrame(exposure_bars).set_index('Date')
        if exposure_bars else pd.DataFrame()
    )

    # 11. Factor attribution (return decomposition)
    factor_attribution = compute_factor_attribution(
        pnl_ts, factor_panel, daily, factor_loadings, has_theme
    )

    return {
        'daily': daily,
        'metrics': metrics,
        'drawdown': drawdown,
        'last_weights': _snapshot_latest_weights(pnl_ts),
        'factor_exposure': factor_loadings,
        'exposure_tracker': exposure_ts,
        'factor_attribution': factor_attribution,
    }


def _snapshot_latest_weights(pnl_ts):
    """Pull the most recent non-zero position snapshot for the positions table."""
    last_date = pnl_ts.index.get_level_values('Date').max()
    cross = pnl_ts.xs(last_date, level='Date')
    w = cross['TargetWeight'].dropna()
    z = cross['Alpha_Z'].reindex(w.index)
    ranked = pd.DataFrame({
        'Ticker': w.index,
        'Weight': w.values,
        'Alpha_Z': z.values,
    }).sort_values('Weight', ascending=False)
    ranked['Leg'] = ranked['Weight'].apply(lambda x: 'Long' if x > 0 else 'Short')
    return ranked

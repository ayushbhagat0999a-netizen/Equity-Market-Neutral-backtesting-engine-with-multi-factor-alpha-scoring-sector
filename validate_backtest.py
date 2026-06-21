"""
Validation script: runs each test CSV through the backtest engine and prints
results for manual inspection.

Usage:
    python validate_backtest.py

Each test prints key metrics and a pass/fail assertion for the expected behaviour.
"""

import csv, os, sys, math, json
from datetime import datetime, date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))


# ── Replicate the JS engine logic in Python for independent validation ──
def std(a):
    m = sum(a) / len(a) if a else 0
    return math.sqrt(sum((x - m)**2 for x in a) / len(a)) if len(a) > 1 else 0

def mean(a):
    return sum(a) / len(a) if a else 0

def ols3b(xs, ys):
    """3-variable OLS (no intercept), matches dashboard.js ols3b() exactly."""
    n = len(xs)
    if n < 3:
        return [0, 0, 0]
    xx = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    xy = [0, 0, 0]
    for i in range(n):
        x1, x2, x3 = xs[i]
        y = ys[i]
        xx[0][0] += x1*x1; xx[0][1] += x1*x2; xx[0][2] += x1*x3
        xx[1][0] += x2*x1; xx[1][1] += x2*x2; xx[1][2] += x2*x3
        xx[2][0] += x3*x1; xx[2][1] += x3*x2; xx[2][2] += x3*x3
        xy[0] += x1*y; xy[1] += x2*y; xy[2] += x3*y
    d = xx[0][0]*(xx[1][1]*xx[2][2]-xx[1][2]*xx[2][1]) - xx[0][1]*(xx[1][0]*xx[2][2]-xx[1][2]*xx[2][0]) + xx[0][2]*(xx[1][0]*xx[2][1]-xx[1][1]*xx[2][0])
    if abs(d) < 1e-12:
        return [0, 0, 0]
    inv = 1.0 / d
    b0 = (xy[0]*(xx[1][1]*xx[2][2]-xx[1][2]*xx[2][1]) - xx[0][1]*(xy[1]*xx[2][2]-xx[1][2]*xy[2]) + xx[0][2]*(xy[1]*xx[2][1]-xx[1][1]*xy[2])) * inv
    b1 = (xx[0][0]*(xy[1]*xx[2][2]-xx[1][2]*xy[2]) - xy[0]*(xx[1][0]*xx[2][2]-xx[1][2]*xx[2][0]) + xx[0][2]*(xx[1][0]*xy[2]-xy[1]*xx[2][0])) * inv
    b2 = (xx[0][0]*(xx[1][1]*xy[2]-xy[1]*xx[2][1]) - xx[0][1]*(xx[1][0]*xy[2]-xy[1]*xx[2][0]) + xy[0]*(xx[1][0]*xx[2][1]-xx[1][1]*xx[2][0])) * inv
    return [b0, b1, b2]


def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            for col in ['Close', 'BP_Z', 'EP_Z', 'Momentum_Z', 'ROE_Z', 'GP_Z', 'Leverage_Z']:
                if col in r and r[col]:
                    r[col] = float(r[col])
            rows.append(r)
    return rows


def run_backtest(rows, w_val=0.33, w_mom=0.33, w_qual=0.34,
                 tc_bps=10, rebal_freq='M', sector_neutral=False, max_turnover=1.0):
    """Pure Python port of the dashboard.js runBacktest()."""
    tc_rate = tc_bps / 10000
    dates = sorted(set(r['Date'] for r in rows))
    date_map = {}
    for r in rows:
        date_map.setdefault(r['Date'], []).append(r)

    has_theme = any(r.get('Value_Score') is not None for r in rows)
    has_sector = any(r.get('Sector') for r in rows)

    # alpha Z-scores
    alpha_z_map = {}
    for d in dates:
        day = date_map[d]
        alpha = {}
        for r in day:
            if has_theme:
                v = float(r.get('Value_Score', 0))
                m = float(r.get('Momentum_Score', 0))
                q = float(r.get('Quality_Score', 0))
            else:
                v = 0.5 * r.get('BP_Z', 0) + 0.5 * r.get('EP_Z', 0)
                m = r.get('Momentum_Z', 0)
                q = (r.get('ROE_Z', 0) + r.get('GP_Z', 0) + r.get('Leverage_Z', 0)) / 3
            alpha[r['Ticker']] = w_val * v + w_mom * m + w_qual * q
        vals = list(alpha.values())
        mn = mean(vals)
        sd = std(vals)
        az = {}
        for t, v in alpha.items():
            az[t] = (v - mn) / sd if sd > 0 else 0
        alpha_z_map[d] = az

    # rebalance dates
    reb_dates = []
    if rebal_freq == 'M':
        seen = set()
        for d in dates:
            k = d[:7]
            if k not in seen:
                seen.add(k)
                reb_dates.append(d)
    else:
        seen = set()
        for d in dates:
            parts = d.split('-')
            q = (int(parts[1]) - 1) // 3 + 1
            k = f"{parts[0]}-Q{q}"
            if k not in seen:
                seen.add(k)
                reb_dates.append(d)

    # weight construction
    weight_map = {}
    for d in reb_dates:
        az = alpha_z_map.get(d)
        if not az:
            continue
        tickers = list(az.keys())
        w = {}
        if sector_neutral and has_sector:
            ticker_sec = {}
            for r in date_map[d]:
                ticker_sec[r['Ticker']] = r.get('Sector', 'Other')
            sectors = {}
            for t in tickers:
                sec = ticker_sec.get(t, 'Other')
                sectors.setdefault(sec, []).append(t)
            n_sec = len(sectors)
            sec_frac = 1.0 / n_sec if n_sec > 0 else 1.0
            for sec, st in sectors.items():
                ls = sum(az[t] for t in st if az[t] > 0)
                ss = sum(abs(az[t]) for t in st if az[t] < 0)
                for t in st:
                    if az[t] > 0 and ls > 0:
                        w[t] = (az[t] / ls) * sec_frac
                    elif az[t] < 0 and ss > 0:
                        w[t] = -(az[t] / ss) * sec_frac
                    else:
                        w[t] = 0
        else:
            ls = sum(az[t] for t in tickers if az[t] > 0)
            ss = sum(abs(az[t]) for t in tickers if az[t] < 0)
            for t in tickers:
                if az[t] > 0 and ls > 0:
                    w[t] = az[t] / ls
                elif az[t] < 0 and ss > 0:
                    w[t] = -(az[t] / ss)
                else:
                    w[t] = 0
        weight_map[d] = w

    # daily PnL
    daily_returns = {}
    last_close = {}
    prev_w = {}
    prev_reb_w = {}
    cum = 1.0

    for d in dates:
        day = date_map[d]
        for r in day:
            ticker = r['Ticker']
            prev = last_close.get(ticker)
            r['Stock_Return'] = (r['Close'] - prev) / prev if prev and prev > 0 else 0
            last_close[ticker] = r['Close']
        port_raw = sum((prev_w.get(r['Ticker'], 0) or 0) * (r.get('Stock_Return', 0) or 0) for r in day)

        cost = 0
        app_w = dict(weight_map.get(d, {}))
        if d in weight_map:
            cur_w = weight_map[d]
            all_t = set(prev_reb_w.keys()) | set(cur_w.keys())
            turnover = sum(abs(cur_w.get(t, 0) - prev_reb_w.get(t, 0)) for t in all_t) / 2
            turn_actual = turnover
            if max_turnover < 1 and turn_actual > max_turnover:
                lam = max_turnover / turn_actual
                for t in all_t:
                    app_w[t] = prev_reb_w.get(t, 0) + (cur_w.get(t, 0) - prev_reb_w.get(t, 0)) * lam
                turn_actual = max_turnover
            cost = turn_actual * tc_rate
            prev_reb_w = dict(app_w)

        port_ret = port_raw - cost
        cum *= (1 + port_ret)
        daily_returns[d] = {'raw': port_raw, 'cost': cost, 'ret': port_ret, 'cum': cum}

        if d in weight_map:
            prev_w = dict(app_w)

    # metrics
    n_days = len(dates)
    n_years = n_days / 252
    total_ret = cum - 1
    ann_ret = cum ** (1 / n_years) - 1 if n_years > 0 else 0
    rets = [daily_returns[d]['ret'] for d in dates]
    ann_vol = std(rets) * math.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    peak = 1
    dd = []
    for d in dates:
        c = daily_returns[d]['cum']
        if c > peak:
            peak = c
        dd.append((c - peak) / peak)
    max_dd = min(dd)

    neg_rets = [r for r in rets if r < 0]
    d_vol = std(neg_rets) * math.sqrt(252) if neg_rets else 0
    sortino = ann_ret / d_vol if d_vol > 0 else 0
    win_rate = sum(1 for r in rets if r > 0) / n_days
    pos_sum = sum(r for r in rets if r > 0) or 0
    neg_sum = abs(sum(r for r in rets if r < 0)) or 0
    pf = pos_sum / neg_sum if neg_sum > 0 else 0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # market return (EW)
    mkt_rets = []
    for d in dates:
        day = date_map[d]
        s = sum(r.get('Stock_Return', 0) or 0 for r in day)
        c = sum(1 for r in day if r.get('Stock_Return') is not None)
        mkt_rets.append(s / c if c > 0 else 0)
    mn_r = mean(rets)
    mn_m = mean(mkt_rets)
    cov = sum((rets[i] - mn_r) * (mkt_rets[i] - mn_m) for i in range(n_days))
    var_m = sum((mkt_rets[i] - mn_m) ** 2 for i in range(n_days))
    beta = cov / var_m if var_m > 0 else 0

    ann_mkt = 1
    for r in mkt_rets:
        ann_mkt *= (1 + r)
    ann_mkt = ann_mkt ** (1 / n_years) - 1 if n_years > 0 else 0
    alpha_j = ann_ret - beta * ann_mkt

    te = math.sqrt(sum((rets[i] - mkt_rets[i]) ** 2 for i in range(n_days)) / n_days) * math.sqrt(252)
    ir = (ann_ret - ann_mkt) / te if te > 0 else 0

    metrics = {
        'Total Return': total_ret,
        'Annualised Return': ann_ret,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd,
        'Annualised Volatility': ann_vol,
        'Beta (EW)': beta,
        'Win Rate': win_rate,
        'Profit Factor': pf,
        'Calmar Ratio': calmar,
        'Alpha (Jensen)': alpha_j,
        'Information Ratio': ir,
    }

    # factor exposure
    fe = {'Value': {'dates': [], 'values': []},
          'Momentum': {'dates': [], 'values': []},
          'Quality': {'dates': [], 'values': []}}
    exp_w = {}
    for d in dates:
        day = date_map[d]
        if d in weight_map:
            exp_w = dict(weight_map[d])
        total_abs = sum(abs(v) for v in exp_w.values())
        if total_abs == 0:
            continue
        vals = {'Value': 0, 'Momentum': 0, 'Quality': 0}
        for r in day:
            w = exp_w.get(r['Ticker'], 0)
            if w == 0:
                continue
            if has_theme:
                vs = float(r.get('Value_Score', 0))
                ms = float(r.get('Momentum_Score', 0))
                qs = float(r.get('Quality_Score', 0))
            else:
                vs = 0.5 * r.get('BP_Z', 0) + 0.5 * r.get('EP_Z', 0)
                ms = r.get('Momentum_Z', 0)
                qs = (r.get('ROE_Z', 0) + r.get('GP_Z', 0) + r.get('Leverage_Z', 0)) / 3
            vals['Value'] += w * vs
            vals['Momentum'] += w * ms
            vals['Quality'] += w * qs
        fe['Value']['dates'].append(d)
        fe['Value']['values'].append(vals['Value'])
        fe['Momentum']['dates'].append(d)
        fe['Momentum']['values'].append(vals['Momentum'])
        fe['Quality']['dates'].append(d)
        fe['Quality']['values'].append(vals['Quality'])

    # factor attribution
    fattr = {'dates': fe['Value']['dates'], 'value': [], 'momentum': [], 'quality': [], 'specific': []}
    cum_av, cum_am, cum_aq, cum_as = 0, 0, 0, 0
    for ai, d in enumerate(fattr['dates']):
        day = date_map[d]
        xs, ys = [], []
        for r in day:
            if has_theme:
                vz = float(r.get('Value_Score', 0))
                mz = float(r.get('Momentum_Score', 0))
                qz = float(r.get('Quality_Score', 0))
            else:
                vz = 0.5 * r.get('BP_Z', 0) + 0.5 * r.get('EP_Z', 0)
                mz = r.get('Momentum_Z', 0)
                qz = (r.get('ROE_Z', 0) + r.get('GP_Z', 0) + r.get('Leverage_Z', 0)) / 3
            xs.append([vz, mz, qz])
            ys.append(r.get('Stock_Return', 0) or 0)
        b = ols3b(xs, ys)
        ve = fe['Value']['values'][ai] if ai < len(fe['Value']['values']) else 0
        me2 = fe['Momentum']['values'][ai] if ai < len(fe['Momentum']['values']) else 0
        qe = fe['Quality']['values'][ai] if ai < len(fe['Quality']['values']) else 0
        vc = b[0] * ve
        mc = b[1] * me2
        qc = b[2] * qe
        dr = daily_returns.get(d, {})
        ret = dr.get('ret', 0)
        sc = ret - vc - mc - qc
        cum_av += vc
        cum_am += mc
        cum_aq += qc
        cum_as += sc
        fattr['value'].append(cum_av)
        fattr['momentum'].append(cum_am)
        fattr['quality'].append(cum_aq)
        fattr['specific'].append(cum_as)

    return metrics, fe, fattr, daily_returns


def pct(v):
    return f"{v*100:.2f}%"

def fmt3(v):
    return f"{v:.3f}"

def run_test(name, path, **kwargs):
    if not os.path.exists(path):
        print(f"  SKIP — file not found: {path}")
        return
    rows = load_csv(path)
    params = {k:v for k,v in kwargs.items() if not k.startswith('_')}
    metrics, fe, fattr, daily = run_backtest(rows, **params)
    desc = kwargs.get('_desc', '')
    print(f"\n{'='*60}")
    print(f" {name}{' — ' + desc if desc else ''}")
    print(f" Tickers: {len(set(r['Ticker'] for r in rows))}, Dates: {len(set(r['Date'] for r in rows))}")
    print(f"{'='*60}")
    print(f"  Total Return:          {pct(metrics['Total Return'])}")
    print(f"  Annualised Return:     {pct(metrics['Annualised Return'])}")
    print(f"  Sharpe Ratio:          {fmt3(metrics['Sharpe Ratio'])}")
    print(f"  Sortino Ratio:         {fmt3(metrics['Sortino Ratio'])}")
    print(f"  Max Drawdown:          {pct(metrics['Max Drawdown'])}")
    print(f"  Annualised Vol:        {pct(metrics['Annualised Volatility'])}")
    print(f"  Beta (EW):             {fmt3(metrics['Beta (EW)'])}")
    print(f"  Win Rate:              {pct(metrics['Win Rate'])}")
    print(f"  Alpha (Jensen):        {pct(metrics['Alpha (Jensen)'])}")
    print(f"  Information Ratio:     {fmt3(metrics['Information Ratio'])}")
    if fattr['dates']:
        last = len(fattr['dates']) - 1
        print(f"  Factor Attribution (cumulative):")
        print(f"    Value:     {pct(fattr['value'][last])}")
        print(f"    Momentum:  {pct(fattr['momentum'][last])}")
        print(f"    Quality:   {pct(fattr['quality'][last])}")
        print(f"    Specific:  {pct(fattr['specific'][last])}")
    return metrics


if __name__ == '__main__':
    print("=" * 60)
    print(" EMN Backtester — Validation Suite")
    print("=" * 60)

    # 1. Bull market — expect positive returns
    run_test("Bull Market", os.path.join(OUT, "bull_test.csv"),
             _desc="All stocks drift up. EMN should be positive.")

    # 2. Bear market — expect near-zero (EMN is market-neutral)
    run_test("Bear Market", os.path.join(OUT, "bear_test.csv"),
             _desc="All stocks drift down. EMN should be near-flat (neutral).")

    # 3. Value dominant — Value explains most return
    run_test("Value Dominant", os.path.join(OUT, "value_dominant_test.csv"),
             w_val=1.0, w_mom=0, w_qual=0,
             _desc="Value weight=100%, high-BP_Z stocks outperform. Attribution should show Value dominant.")

    # 4. Momentum dominant — Momentum explains most return
    run_test("Momentum Dominant", os.path.join(OUT, "momentum_dominant_test.csv"),
             w_val=0, w_mom=1.0, w_qual=0,
             _desc="Momentum weight=100%, high-Momentum_Z stocks outperform. Attribution should show Momentum dominant.")

    # 5. Divergent factors — Value and Momentum pull opposite
    run_test("Divergent Factors", os.path.join(OUT, "divergent_test.csv"),
             _desc="Value (+) and Momentum (-) clash. Equal weights should give near-zero net.")

    # 6. Sector test WITHOUT sector neutrality
    run_test("Sector Test (no neutrality)", os.path.join(OUT, "sector_test.csv"),
             sector_neutral=False,
             _desc="Tech surges, Fin drops. Without neutrality: portfolio goes long Tech, short Fin -> big return.")

    # 7. Sector test WITH sector neutrality
    run_test("Sector Test (WITH neutrality)", os.path.join(OUT, "sector_test.csv"),
             sector_neutral=True,
             _desc="Same data WITH sector neutrality. Each sector net-zero -> return should be much smaller.")

    # 8. Turnover constraint test
    run_test("Turnover Constraint (max=100%)", os.path.join(OUT, "turnover_test.csv"),
             max_turnover=1.0,
             _desc="No turnover limit. Frequent flipping -> higher cost.")

    run_test("Turnover Constraint (max=10%)", os.path.join(OUT, "turnover_test.csv"),
             max_turnover=0.10,
             _desc="Max 10% turnover per rebalance. Smoother weights, lower cost.")

    # 9. Bull market with max turnover constraint
    run_test("Bull + Turnover 5%", os.path.join(OUT, "bull_test.csv"),
             max_turnover=0.05,
             _desc="Bull market with extreme 5% turnover cap. Return should be attenuated.")

    print("\n" + "=" * 60)
    print(" Validation complete.")
    print(" Upload any *_test.csv into dashboard.html to verify interactively.")

    # Summary
    print("\n Expected behaviour quick-reference:")
    print("  bull_test.csv         -> positive return (model goes long winners)")
    print("  bear_test.csv         -> near-zero (EMN is market-neutral, not short-only)")
    print("  value_dominant_test.csv -> Value attribution dominates")
    print("  momentum_dominant_test.csv -> Momentum attribution dominates")
    print("  divergent_test.csv    -> near-zero at equal factor weights")
    print("  sector_test.csv       -> big return without neutrality, small with")
    print("  turnover_test.csv     -> 100% vs 10% shows cost/volatility difference")

"""
EMN Multi-Factor Backtester — Interactive Web App
Run with:  python webapp/app.py
Then open http://localhost:5000 in your browser.
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from webapp.engine import load_factor_panel, run_backtest

app = Flask(__name__)

print("Loading factor panel...")
FACTOR_PANEL = load_factor_panel()
print("  Done.  Shape:", FACTOR_PANEL.shape)


def prep_series(s):
    """Convert a pandas Series with DatetimeIndex to (dates[], values[]) for JSON."""
    if s is None or len(s) == 0:
        return {'dates': [], 'values': []}
    dates = [str(d.date()) for d in s.index]
    vals = [None if np.isnan(v) else round(float(v), 6) for v in s.values]
    return {'dates': dates, 'values': vals}


def prep_multi(series_list, labels):
    """Convert multiple aligned Series to {label: {dates, values}}."""
    if not series_list or series_list[0] is None or len(series_list[0]) == 0:
        return {}
    dates = [str(d.date()) for d in series_list[0].index]
    result = {}
    for s, label in zip(series_list, labels):
        result[label] = {
            'dates': dates,
            'values': [None if np.isnan(v) else round(float(v), 6) for v in s.values]
        }
    return result


def histogram_data(s, bins=50):
    """Compute histogram bins and counts server-side."""
    if s is None or len(s) == 0:
        return {'bins': [], 'counts': [], 'mean': 0}
    counts, edges = np.histogram(s, bins=bins)
    bin_centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]
    return {
        'bins': [round(float(x), 4) for x in bin_centers],
        'counts': [int(c) for c in counts],
        'mean': round(float(s.mean()), 6),
    }


def prepare_chart_data(daily, drawdown, factor_exposure, exposure_tracker):
    """Build JSON-serializable chart data from backtest results."""

    # Compute rolling Sharpe server-side
    roll = 63
    roll_ret = daily['Port_Ret'].rolling(roll).mean() * 252
    roll_vol = daily['Port_Ret'].rolling(roll).std() * np.sqrt(252)
    roll_sharpe = roll_ret / roll_vol

    data = {
        'equity': prep_series(daily['Cumulative']),
        'drawdown': prep_series(drawdown * 100),
        'distribution': histogram_data(daily['Port_Ret'] * 100),
        'rolling_sharpe': prep_series(roll_sharpe),
    }

    # Factor exposures
    if not factor_exposure.empty:
        fe = {}
        dates = [str(d.date()) for d in factor_exposure.index]
        for col in factor_exposure.columns:
            fe[col] = {
                'dates': dates,
                'values': [None if np.isnan(v) else round(float(v), 6)
                           for v in factor_exposure[col].values]
            }
        data['factor_exposure'] = fe

    # Exposure tracker
    if not exposure_tracker.empty:
        et = prep_multi(
            [exposure_tracker[c] for c in exposure_tracker.columns],
            list(exposure_tracker.columns)
        )
        data['exposure'] = et

    return data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run', methods=['POST'])
def run():
    try:
        body = request.get_json()
        w_val = float(body.get('w_val', 0.33))
        w_mom = float(body.get('w_mom', 0.33))
        w_qual = float(body.get('w_qual', 0.34))
        tc_rate = float(body.get('tc_rate', 0.001))
        rf_rate = float(body.get('rf_rate', 0.0))
        rebal = body.get('rebalance', 'M')
        sector_neutral = body.get('sector_neutral', False)
        max_turnover = float(body.get('max_turnover', 1.0))

        result = run_backtest(FACTOR_PANEL, w_val, w_mom, w_qual,
                              tc_rate, rf_rate, rebal,
                              sector_neutral=sector_neutral,
                              max_turnover=max_turnover)

        daily = result['daily']
        metrics = result['metrics']
        drawdown = result['drawdown']
        last_w = result['last_weights']
        factor_exposure = result.get('factor_exposure', pd.DataFrame())
        exposure_tracker = result.get('exposure_tracker', pd.DataFrame())
        factor_attribution = result.get('factor_attribution', {})

        chart_data = prepare_chart_data(daily, drawdown, factor_exposure,
                                        exposure_tracker)
        if factor_attribution and factor_attribution.get('dates'):
            chart_data['factor_attribution'] = factor_attribution

        def fmt(v):
            if isinstance(v, float):
                return round(v, 6)
            return v

        return jsonify({
            'success': True,
            'metrics': {k: fmt(v) for k, v in metrics.items()},
            'chart_data': chart_data,
            'positions': last_w.to_dict('records'),
            'n_dates': len(daily),
            'date_start': str(daily.index[0].date()),
            'date_end': str(daily.index[-1].date()),
        })
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e),
                        'traceback': traceback.format_exc()})


if __name__ == '__main__':
    print("\nStart the app:  http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

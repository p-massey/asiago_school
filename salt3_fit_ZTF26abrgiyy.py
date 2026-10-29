"""
SALT3 fit to the ZTF forced difference photometry of ZTF26abrgiyy, to find the
time of B-band maximum (t0).

Fluxes are forced-photometry uJy, so the AB zero point is 23.9. Duplicate rows
(the same epoch listed as both a detection and a limit) are dropped. The fit
is done with the redshift fixed at z = 0.020, twice: first on all epochs, then on the rest-frame phase window
[-15, +45] d around the first-pass t0, as usual for SALT fits. The SALT3 model
covariance is included (modelcov=True); with data errors alone chi2/dof is ~100.
"""
import os

import numpy as np
import pandas as pd
import sncosmo
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'ZTF26abrgiyy_difference_photometry.csv')
OUT_CSV = os.path.join(HERE, 'ZTF26abrgiyy_salt3_fit.csv')
OUT_FIG = os.path.join(HERE, 'ZTF26abrgiyy_salt3_fit.pdf')

BANDS = {'g': 'ztfg', 'r': 'ztfr', 'i': 'ztfi'}
ZP_UJY = 23.9
REDSHIFT = 0.020
PHASE_WINDOW = (-15.0, 45.0)


def load_photometry():
    d = pd.read_csv(CSV).dropna(subset=['forced_ujy', 'forced_ujy_error'])
    d = d.drop_duplicates(subset=['MJD', 'filter']).sort_values('MJD')
    return Table({'time': d['MJD'].values, 'band': d['filter'].map(BANDS).values,
                  'flux': d['forced_ujy'].values, 'fluxerr': d['forced_ujy_error'].values,
                  'zp': np.full(len(d), ZP_UJY), 'zpsys': np.full(len(d), 'ab')})


def fit(data, model, t0_guess):
    model.set(z=REDSHIFT, t0=t0_guess)
    bounds = {'t0': (t0_guess - 15, t0_guess + 15), 'x1': (-3, 3), 'c': (-0.3, 0.5)}
    return sncosmo.fit_lc(data, model, ['t0', 'x0', 'x1', 'c'], bounds=bounds, modelcov=True)


def main():
    data = load_photometry()
    model = sncosmo.Model(source='salt3')

    # first pass on all epochs, starting from the brightest g-band point
    g = data[data['band'] == 'ztfg']
    t0_guess = g['time'][np.argmax(g['flux'])]
    res, fitted = fit(data, model, t0_guess)

    # second pass in the standard rest-frame phase window
    phase = (data['time'] - fitted['t0']) / (1 + REDSHIFT)
    data_win = data[(phase >= PHASE_WINDOW[0]) & (phase <= PHASE_WINDOW[1])]
    res, fitted = fit(data_win, model, fitted['t0'])

    chi2_dof = res.chisq / res.ndof
    print(f'{len(data_win)} epochs in the phase window, chi2/dof = {res.chisq:.1f}/{res.ndof} = {chi2_dof:.2f}')
    rows = []
    for name in res.vparam_names:
        val, err = fitted[name], res.errors[name]
        print(f'  {name:3s} = {val:.5g} +/- {err:.2g}')
        rows.append({'param': name, 'value': val, 'error': err})
    # errors rescaled by sqrt(chi2/dof) when the fit is worse than the photometric errors allow
    scale = max(1.0, np.sqrt(chi2_dof))
    print(f't0 (B max) = MJD {fitted["t0"]:.2f} +/- {res.errors["t0"]:.2f} '
          f'(+/- {scale * res.errors["t0"]:.2f} rescaled by sqrt(chi2/dof))')
    print(f'peak m_B (rest) = {fitted.source_peakmag("bessellb", "ab"):.2f}')
    rows += [{'param': 'chi2', 'value': res.chisq, 'error': np.nan},
             {'param': 'ndof', 'value': res.ndof, 'error': np.nan},
             {'param': 't0_err_rescaled', 'value': scale * res.errors['t0'], 'error': np.nan}]
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    fig = sncosmo.plot_lc(data_win, model=fitted, errors=res.errors, zp=ZP_UJY, figtext=None)
    fig.savefig(OUT_FIG, bbox_inches='tight')
    print(f'Saved {OUT_CSV} and {OUT_FIG}')


if __name__ == '__main__':
    main()

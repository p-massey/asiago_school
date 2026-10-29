"""
SALT3 fit to the ATLAS forced photometry of SN 2026zwp, to find the time of
B-band maximum (t0). Same procedure as salt3_fit_ZTF26abrgiyy.py.

The input is ATLAS forced photometry in uJy (AB zero point 23.9), one row per
exposure, in the c and o bands. The redshift is fixed at z = 0.021.
The fit is done twice: first on all epochs, then on the rest-frame phase
window [-15, +45] d around the first-pass t0. The SALT3 model covariance is
included (modelcov=True).
"""
import os

import numpy as np
import pandas as pd
import sncosmo
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
PHOT = os.path.join(HERE, '2026zwp.txt')
OUT_CSV = os.path.join(HERE, '2026zwp_salt3_fit.csv')
OUT_FIG = os.path.join(HERE, '2026zwp_salt3_fit.pdf')

BANDS = {'c': 'atlasc', 'o': 'atlaso'}
ZP_UJY = 23.9
REDSHIFT = 0.021
PHASE_WINDOW = (-15.0, 45.0)


def load_photometry(max_chin=None):
    d = pd.read_csv(PHOT, sep=r'\s+')
    d.columns = [c.lstrip('#') for c in d.columns]
    d = d[(d['err'] == 0) & d['F'].isin(BANDS)].sort_values('MJD')
    if max_chin is not None:
        d = d[d['chi/N'] <= max_chin]
    return Table({'time': d['MJD'].values, 'band': d['F'].map(BANDS).values,
                  'flux': d['uJy'].values.astype(float), 'fluxerr': d['duJy'].values.astype(float),
                  'zp': np.full(len(d), ZP_UJY), 'zpsys': np.full(len(d), 'ab')})


def fit(data, model, t0_guess, z=REDSHIFT):
    model.set(z=z, t0=t0_guess)
    bounds = {'t0': (t0_guess - 15, t0_guess + 15), 'x1': (-3, 3), 'c': (-0.3, 0.5)}
    return sncosmo.fit_lc(data, model, ['t0', 'x0', 'x1', 'c'], bounds=bounds, modelcov=True)


def fit_two_pass(data, z=REDSHIFT):
    model = sncosmo.Model(source='salt3')
    t0_guess = data['time'][np.argmax(data['flux'])]
    res, fitted = fit(data, model, t0_guess, z)
    phase = (data['time'] - fitted['t0']) / (1 + z)
    data_win = data[(phase >= PHASE_WINDOW[0]) & (phase <= PHASE_WINDOW[1])]
    res, fitted = fit(data_win, model, fitted['t0'], z)
    return res, fitted, data_win


def main():
    res, fitted, data_win = fit_two_pass(load_photometry())

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

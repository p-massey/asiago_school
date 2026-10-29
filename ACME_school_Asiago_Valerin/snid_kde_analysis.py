"""
KDE phase estimates for the Asiago reduced spectra SNID outputs, with the
CfA error snake added to the KDE uncertainty.

KDE: rlap^2-weighted Gaussian KDE (Silverman bandwidth) over the top N good
fits, from snid_analysis/src/utils/snid.py. N = 8 is the value chosen by the
pull-calibration sweep in snid_analysis/phase_extraction/KDE/cfa_kde_phases.ipynb,
which produced the CfA calibration sample used for the snake.

Error snake: as in empirical_fitting_analysis/error_snake_csp_ztf_test.ipynb.
In equal-count bins of found phase, the snake is half the 16th-84th percentile
range of t_KDE - t_LC on the CfA sample; it is added in quadrature to each side
of the KDE uncertainty. No bias correction is applied.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/pxm588@student.bham.ac.uk/PhD/snid_analysis/src/utils')
from snid import (parse_snid_file, kde_phase_estimate, calculate_median_age,
                  calculate_simple_mean)

# SNID run directories; ia_only drops non-Ia template matches before the KDE, for runs
# that were not restricted to Ia templates in SNID itself (usetype)
SNID_RUNS = [
    dict(name='reduced_spectra_snid', dir=os.path.expanduser('~/Desktop/snid/asiago/reduced_spectra_snid'), ia_only=False),
    dict(name='obs_Ia', dir=os.path.expanduser('~/Desktop/snid/asiago/obs_Ia'), ia_only=True),
]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reduced_spectra_snid')
OUT_CSV = os.path.join(OUT_DIR, 'snid_kde_phases.csv')
SNAKE_CSV = os.path.join(OUT_DIR, 'kde_error_snake_cfa.csv')

TOP_N = 8
MIN_FITS = 8

# CfA calibration sample: KDE (N = 8) on the zfilter = 0.0025 SNID run
CFA_KDE_CSV = '/Users/pxm588@student.bham.ac.uk/PhD/snid_analysis/phase_extraction/KDE/kde_phase_results.csv'
SUBTYPES = ['N', 'HV']
SNR_MIN = 10.0
PHASE_WINDOW = (-12.0, 33.0)
N_BINS = 8


def read_snid_agem(path):
    """Return SNID's own agem and its error from the output header."""
    with open(path) as f:
        for line in f:
            if line.startswith('agem'):
                parts = line.split()
                return float(parts[1]), float(parts[2])
    return None, None


def parse_snid_file_ia(filename):
    """As parse_snid_file, keeping only matches to SN Ia templates."""
    ages, rlaps, redshifts = [], [], []
    in_rlap_section = False
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith('### rlap-ordered template listings ###'):
                in_rlap_section = True
                next(f)  # Skip the header line
                continue
            if in_rlap_section:
                if line.startswith('#--- rlap cutoff'):
                    break
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) > 9 and parts[8] == '0' and parts[9] == 'good' and parts[2].startswith('Ia'):
                    ages.append(float(parts[7]))
                    rlaps.append(float(parts[4]))
                    redshifts.append(float(parts[5]))
    return ages, rlaps, redshifts


def read_snid_header(path):
    """Run options and SNID's median redshift from the output header."""
    keys = {'Use only types': 'snid_usetype', 'Forced initial redshift': 'snid_forcez',
            'Redshift filter': 'snid_zfilter'}
    info = {}
    with open(path) as f:
        for line in f:
            if line.startswith('# ') and ':' in line:
                key, val = (x.strip() for x in line[2:].split(':', 1))
                if key in keys:
                    info[keys[key]] = val
            elif line.startswith('zmed'):
                info['SNID_zmed'] = float(line.split()[1])
                break
    return info


def build_snake(calib, n_bins=N_BINS):
    """Equal-count bins in KDE phase; spread = half the 68 per cent range of the residual."""
    rows = []
    for iv, g in calib.groupby(pd.qcut(calib['kde_phase'], n_bins, duplicates='drop'), observed=True):
        p16, p84 = np.percentile(g['kde_residual'], [16, 84])
        rows.append(dict(bin_lo=iv.left, bin_hi=iv.right, n=len(g), n_sne=g['sn_name'].nunique(),
                         median_resid=g['kde_residual'].median(), spread=0.5 * (p84 - p16)))
    return pd.DataFrame(rows)


def lookup(table, phase):
    """Snake value for the bin containing `phase` (edge bins extend outwards)."""
    edges = np.append(table['bin_lo'].values, table['bin_hi'].iloc[-1])
    return table['spread'].values[np.clip(np.searchsorted(edges, phase, side='right') - 1, 0, len(table) - 1)]


def load_calibration():
    cfa = pd.read_csv(CFA_KDE_CSV)
    cfa = cfa[cfa['subtype'].isin(SUBTYPES) & (cfa['snr'] >= SNR_MIN)
              & cfa['phase_true'].between(*PHASE_WINDOW) & np.isfinite(cfa['kde_phase'])]
    return cfa.reset_index(drop=True)


def main():
    cfa = load_calibration()
    snake = build_snake(cfa)
    snake.to_csv(SNAKE_CSV, index=False, float_format='%.3f')
    print(f'CfA calibration: {len(cfa)} spectra of {cfa["sn_name"].nunique()} SNe')
    print(snake.round(2).to_string(index=False), '\n')

    rows = []
    paths = [(run, p) for run in SNID_RUNS for p in sorted(glob.glob(os.path.join(run['dir'], '*_snid.output')))]
    for run, path in paths:
        spectrum = os.path.basename(path).replace('_snid.output', '')
        ages, rlaps, redshifts = (parse_snid_file_ia if run['ia_only'] else parse_snid_file)(path)
        agem, agem_err = read_snid_agem(path)

        kde, sig_m, sig_p = (None, None, None)
        if len(ages) >= MIN_FITS:
            kde, sig_m, sig_p = kde_phase_estimate(ages, rlaps, bw_method='silverman',
                                                   weight_power=2, top_n=TOP_N)
        spread = lookup(snake, kde) if kde is not None else None
        med_std, med20 = calculate_median_age(ages, rlaps, top_n=20)
        mean_std, mean20 = calculate_simple_mean(ages, rlaps, top_n=20)

        rows.append({
            'spectrum': spectrum,
            'snid_run': run['name'],
            **read_snid_header(path),
            'ia_only_filter': run['ia_only'],
            'n_good_fits': len(ages),
            'max_rlap': max(rlaps) if rlaps else None,
            'KDE_T8': kde,
            'KDE_T8_SigM': sig_m,
            'KDE_T8_SigP': sig_p,
            'Snake': spread,
            'KDE_T8_SigM_Total': np.hypot(sig_m, spread) if kde is not None else None,
            'KDE_T8_SigP_Total': np.hypot(sig_p, spread) if kde is not None else None,
            'Median_T20': med20,
            'Median_T20_Std': med_std,
            'Mean_T20': mean20,
            'Mean_T20_Std': mean_std,
            'SNID_agem': agem,
            'SNID_agem_err': agem_err,
        })

    df = pd.DataFrame(rows)
    df['SNID_zmed'] = df['SNID_zmed'].map('{:.4f}'.format)
    df.to_csv(OUT_CSV, index=False, float_format='%.2f')
    print(df.to_string(index=False, float_format='%.2f'))
    print(f'\nSaved to {OUT_CSV}\nSnake table saved to {SNAKE_CSV}')


if __name__ == '__main__':
    main()

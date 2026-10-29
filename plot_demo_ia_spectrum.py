"""
Demo figure: a high-S/N normal SN Ia spectrum at maximum light, with the main
absorption features labelled.

SN 1998aq, CfA FAST spectrum of 1998 April 28 (t_LC = +0.06 d, S/N ~ 336,
dm15 = 1.11), dereddened, from snid_analysis/data/cfa_spectra_dereddened.
MNRAS two-column width.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = '/Users/pxm588@student.bham.ac.uk/PhD/snid_analysis/data/cfa_spectra_dereddened/sn1998aq-19980428.26-fast.flm'
OUT = os.path.join(HERE, 'demo_ia_spectrum_sn1998aq.pdf')
Z = 0.0039

# (label, rest wavelength of the line, rest-frame window in which to find the absorption minimum)
FEATURES = [
    (r'Ca\,\textsc{ii} H\&K', 3945, (3740, 3850)),
    (r'Si\,\textsc{ii} $\lambda4130$', 4130, (3940, 4040)),
    (r'Mg\,\textsc{ii} $\lambda4481$', 4481, (4150, 4400)),
    (r'Fe\,\textsc{ii}', 5018, (4750, 5000)),
    (r'S\,\textsc{ii} $\lambda5454$', 5454, (5200, 5350)),
    (r'S\,\textsc{ii} $\lambda5640$', 5640, (5380, 5520)),
    (r'Si\,\textsc{ii} $\lambda5972$', 5972, (5650, 5850)),
    (r'Si\,\textsc{ii} $\lambda6355$', 6355, (6000, 6250)),
]

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'text.usetex': True,
    'font.size': 9,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def main():
    wave, flux, err = np.loadtxt(SPEC, unpack=True)
    wave = wave / (1 + Z)
    smooth = median_filter(flux, size=15)
    norm = smooth.max()
    flux, err, smooth = flux / norm, err / norm, smooth / norm

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.fill_between(wave, flux - err, flux + err, color='0.75', lw=0, zorder=1)
    ax.plot(wave, flux, color='black', lw=0.6, zorder=2)

    for label, lam0, (lo, hi) in FEATURES:
        win = (wave >= lo) & (wave <= hi)
        i = np.argmin(smooth[win])
        lam_min, f_min = wave[win][i], smooth[win][i]
        v = 2.998e5 * ((lam_min / lam0) ** 2 - 1) / ((lam_min / lam0) ** 2 + 1)
        print(f'{label:35s} minimum at {lam_min:7.1f} A  v = {v / 1e3:6.1f} x 10^3 km/s')
        ax.annotate(label, xy=(lam_min, f_min - 0.01), xytext=(lam_min, f_min - 0.05),
                    ha='center', va='top', fontsize=7.5, rotation=90,
                    arrowprops=dict(arrowstyle='-', color='0.3', lw=0.6))

    ax.set_xlim(wave.min(), wave.max())
    ax.set_ylim(-0.3, 1.08)
    ax.set_xlabel(r'Rest wavelength (\AA)')
    ax.set_ylabel(r'Normalized $F_\lambda$')
    fig.savefig(OUT)
    fig.savefig(OUT.replace('.pdf', '.png'))
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()

# Copyright (C) 2026
# The Institute of Statistical Mathematics
# 10-3 Midori-cho, Tachikawa, Tokyo 190-8562, Japan.
#
# This file is part of priism-selfcal.
#
# priism-selfcal is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# priism-selfcal is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with priism-selfcal.  If not, see <https://www.gnu.org/licenses/>.
"""
End-to-end example: fully automated imaging + self-calibration of a real
ALMA Measurement Set, using paramsearch.run_staged_parameter_search() to
choose all four regularization weights (lambda1, lambda_tsv, mu_sq,
mu_abs) rather than setting them by hand.

Usage:
    python examples/hd142527_paramsearch.py /path/to/data.ms /path/to/outdir

Edit MSNAME/OUTDIR below directly if you'd rather not pass them as
arguments.

Outputs written to OUTDIR: image.fits (the actual imaging-command
output -- a real CASA/FITS image from the winning (l1, ltsv) solve, via
AlmaSparseModelingImager.exportimage()) and selfcal.gcal (the
self-calibration result as a standard CASA G-Jones caltable, via
casa_gaintable.write_gaintable(), applicable to the MS with CASA's own
applycal); plus image_raw.npy/image.npy/image.png/gain_scatter.png/
summary.json as diagnostics.

Optional: for the exact EHT color palette used in the saved image
(afmhot_10us, the perceptually-uniformized afmhot variant used in EHT
M87/Sgr A* images), install ehtplot --
    pip install git+https://github.com/liamedeiros/ehtplot
(not on PyPI under that name; installs as "pyehtplot"). Without it,
this script falls back to matplotlib's stock 'afmhot'.

Expect this to take a while: on real HD142527 data (512x512 image,
~50000 visibilities), a full run (the default n_refine_rounds=1: one
lambda search, one mu search, one refined lambda search, each 50
combined Optuna startup+search trials, plus each lambda stage's own
higher-maxiter convergence solve) took ~2.5 hours end to end
(2026-08-22). mu-search trials dominate the cost, since each runs a full
totalimaging.run() (imaging <-> self-calibration alternation); lambda-
search trials are much cheaper (no self-calibration). See README.md's
"paramsearch" section for what each stage does and why it's shaped this
way, and tests/test_paramsearch.py for a fast synthetic-data smoke test
to try your setup against before committing to a real, multi-hour run.

Validated result (2026-08-22): this script reproduced HD142527's
expected crescent-ring morphology with gains clustered near 1+0j
(|gain| in [0.92, 1.28], no outliers) at
(lambda1, lambda_tsv, mu_sq, mu_abs) = (1e5, 1e14, 1e5, 10).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as mcm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# EHT's own color palette (github.com/liamedeiros/ehtplot, installed as
# "pyehtplot") registers 'afmhot_10us' -- the perceptually-uniformized
# afmhot variant used in EHT M87/Sgr A* images -- as a matplotlib
# colormap on import. Its color/ctab.py still calls the removed
# matplotlib.cm.get_cmap (dropped in matplotlib >=3.11), so shim it back
# before importing; falls back to matplotlib's stock 'afmhot' if
# ehtplot isn't installed at all.
try:
    if not hasattr(mcm, 'get_cmap'):
        mcm.get_cmap = plt.get_cmap  # matplotlib >=3.11 compat shim for ehtplot
    import ehtplot.color  # noqa: F401  (registers 'afmhot_10us' with matplotlib)
    IMAGE_CMAP = 'afmhot_10us'
except ImportError:
    print("WARNING: ehtplot not installed (pip install git+https://github.com/"
          "liamedeiros/ehtplot) -- falling back to matplotlib's stock 'afmhot', "
          "not the exact EHT-uniformized 'afmhot_10us'.", flush=True)
    IMAGE_CMAP = 'afmhot'

# priism's "sakura" alignment/gridding helper library (libsakurapy) is a
# separate, sometimes hard-to-obtain C extension that this package's own
# NUFFT-based imaging path (solver='pymfista_nufft', used below) doesn't
# actually need -- only priism.core.datacontainer/alma.gridder import it
# at module load time. If it isn't installed, stub it out; if it *is*
# installed, this stub is simply never used (the real import wins).
try:
    import priism.external.sakura  # noqa: F401
except ImportError:
    import types
    sakura_stub = types.ModuleType('priism.external.sakura')
    sakura_stub.empty_aligned = lambda shape, dtype=np.float64: np.empty(shape, dtype=dtype)
    sakura_stub.empty_like_aligned = lambda a: np.empty_like(a)
    sys.modules['priism.external.sakura'] = sakura_stub

from priism_selfcal.msreader import read_ms_for_selfcal
from priism_selfcal.casa_gaintable import write_gaintable
from priism_selfcal import paramsearch

MSNAME = sys.argv[1] if len(sys.argv) > 1 else '/path/to/concat.ms.cal.HD142527.avg60'
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else '/path/to/output/directory'

IMSIZE = 512
CELL_ARCSEC = 0.01  # must match the `cell` string passed to read_ms_for_selfcal below


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    t_start = time.time()

    print('=== reading MS ===', flush=True)
    data = read_ms_for_selfcal(
        MSNAME, spw='0', imsize=IMSIZE, cell=f'{CELL_ARCSEC}arcsec', field='0',
        datacolumn='data', solver='pymfista_nufft',
    )
    ws = data.imager.working_set
    vis_org = ws.rdata + 1j * ws.idata
    sigma = 1.0 / np.sqrt(ws.weight)
    print(f'M = {vis_org.size}, gain_num = {data.gains.gain_num}, st_num = {data.gains.st_num}',
          flush=True)

    # 'large_variance' allows more gain wobble than 'small_variance'
    # (Ikeda et al. 2025's two tested targets); pick whichever matches
    # how noisy you expect the observing conditions to have been.
    target = paramsearch.GAIN_TARGETS['large_variance']
    print(f'gain target: sigma_ph={target.sigma_ph} deg, sigma_amp={target.sigma_amp}', flush=True)

    # Search ranges below are deliberately wide (found by trial and
    # error on this exact dataset, 2026-08-21/22): too-narrow ranges
    # silently return a "best effort" result that never actually
    # satisfies the lambda search's C1/C2 soft constraints, or clamps
    # mu to whatever boundary is closest to the true optimum. If you're
    # imaging a different dataset/pixel scale, start even wider and
    # narrow down once you see where the search actually lands.
    result = paramsearch.run_staged_parameter_search(
        data.imager, ws.antenna1, ws.antenna2, ws.time, vis_org, sigma, target,
        l1_exp_range=(-4, 6), ltsv_exp_range=(-2, 15),
        mu_sq_exp_range=(0, 12), mu_abs_exp_range=(0, 10),
        narrow_width=2, n_refine_rounds=1,
        selfcal_num=3,
        bayesopt_n_startup_trials=20, bayesopt_n_search_trials=30,
        imaging_maxiter=300, selfcal_maxiter=2000,
        imageprefix=os.path.join(OUTDIR, '_scratch_image'),
    )

    t_end = time.time()
    print(f'=== staged search complete in {t_end - t_start:.1f} sec ===', flush=True)

    # --- save outputs ---
    np.save(os.path.join(OUTDIR, 'image_raw.npy'), result.image)

    # Standing display convention for this pipeline's images (confirmed
    # against HD142527's known orientation, 2026-08-21): rotate 90 degrees
    # clockwise, then flip left-right. Re-check against a known source
    # orientation before trusting this for a different phasecenter/imsize
    # setup.
    displayed_image = np.fliplr(np.rot90(result.image, k=-1))
    np.save(os.path.join(OUTDIR, 'image.npy'), displayed_image)

    # Angular (not pixel) axes, RA increasing to the left as in Ikeda et
    # al. 2025's own figures: extent's left edge is +half_fov (positive
    # RA), right edge is -half_fov, so imshow's normal left-to-right
    # pixel order renders decreasing RA.
    ny, nx = displayed_image.shape
    half_fov_x = nx / 2.0 * CELL_ARCSEC
    half_fov_y = ny / 2.0 * CELL_ARCSEC

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    im = ax.imshow(
        displayed_image, origin='lower', cmap=IMAGE_CMAP,
        extent=(half_fov_x, -half_fov_x, -half_fov_y, half_fov_y),
    )
    ax.set_xlabel('Relative RA [arcsec]')
    ax.set_ylabel('Relative Dec [arcsec]')
    ax.set_title('HD142527 (staged param search)')
    # make_axes_locatable ties the colorbar axis's height to the image
    # axes' own height exactly, unlike fig.colorbar(im, ax=ax)'s default
    # (which can end up taller/shorter than the image once tight_layout
    # adjusts the figure).
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, label='Jy/pixel')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'image.png'), dpi=150)
    plt.close(fig)

    result.gains.plot_gains(fname=os.path.join(OUTDIR, 'gain_scatter.png'))

    # The actual imaging command: data.imager.working_set/imparam already
    # hold the winning (l1, ltsv) solve and the gain-corrected visibility
    # from the staged search's last stage (see
    # paramsearch.search_imaging_regularizers's docstring), so this is a
    # real, importable CASA/FITS image -- not just the .npy/.png diagnostic
    # dumps above -- of the same result plotted in image.png.
    fits_path = os.path.join(OUTDIR, 'image.fits')
    data.imager.exportimage(fits_path, overwrite=True)

    # Write the final round's self-calibration gains out as a CASA G-Jones
    # caltable, so the result can actually be applied to the MS (e.g. via
    # CASA's applycal) rather than staying locked inside this script's own
    # Gains object. antenna_offset=data.antenna_offset is 0 here (single-MS
    # use); see read_multi_ms_for_selfcal/write_gaintable's own docs for
    # the multi-MS case.
    gaintable_path = os.path.join(OUTDIR, 'selfcal.gcal')
    write_gaintable(
        result.gains, MSNAME, gaintable_path, spw=0, field=0,
        overwrite=True, antenna_offset=data.antenna_offset,
    )

    sigma_ph, sigma_amp = paramsearch.gain_dispersion(result.gains.gain)
    amp = np.abs(result.gains.gain)
    summary = dict(
        elapsed_sec=t_end - t_start,
        fits_image=fits_path,
        gaintable=gaintable_path,
        l1=result.l1, ltsv=result.ltsv, mu_sq=result.mu_sq, mu_abs=result.mu_abs,
        gain_target=dict(sigma_ph=target.sigma_ph, sigma_amp=target.sigma_amp),
        gain_dispersion_achieved=dict(sigma_ph=sigma_ph, sigma_amp=sigma_amp),
        gain_amp_min=float(amp.min()),
        gain_amp_max=float(amp.max()),
        image_sum=float(result.image.sum()),
        image_max=float(result.image.max()),
        stage_results={
            stage: dict(
                **({'l1': r.l1, 'ltsv': r.ltsv} if hasattr(r, 'l1') else {}),
                **({'mu_sq': r.mu_sq, 'mu_abs': r.mu_abs, 'cost': r.cost} if hasattr(r, 'mu_sq') else {}),
            )
            for stage, r in result.stages.items()
        },
    )
    with open(os.path.join(OUTDIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print('=== DONE ===', flush=True)


if __name__ == '__main__':
    main()

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
Staged Bayesian-optimization search over the 4 imaging/self-cal
hyperparameters (lambda1, lambda_tsv, mu_sq, mu_abs), per Ikeda et al.
2025 (PASJ 77(2):260-276), section 3.4.

The paper explicitly rejects a naive joint 4-D grid search as too slow,
and instead alternates two independent 2-parameter searches:

  - (lambda1, lambda_tsv), scored by uvcriteria's C1/C2 u-v-distance
    criterion (eq. 16), with gains held fixed -- this is priism's own
    optimizeparameters(criterion='ellipsoid'), reused here unchanged.
  - (mu_sq, mu_abs), scored by how close the resulting gains' phase/
    amplitude standard deviation (sigma_ph, sigma_amp) come to
    caller-supplied target values (eq. 17), with lambda1/lambda_tsv held
    fixed -- new in this module, since priism itself has no self-cal
    concept.

Sequencing (verified 2026-08-21 against the actual historical driver
scripts in old/python/*_step0.py..step3.py via their hardcoded-constant
provenance chain, not the paper's own text -- the paper's "we repeated
steps 2 and 3 for 10 iterations" turned out to be a mislabeling that
reuses wording from section 3.3's *inner* TotalImaging loop description,
not a literal 10x outer repeat; the surviving code only ever ran a
single pass):

  stage 0: mu search from scratch (gain initialized to 1+0j each trial),
           lambda1=lambda_tsv=1.0 fixed.
  stage 1: lambda search (priism's ellipsoid criterion, OnlyImaging-style
           -- no self-cal loop), gain fixed to stage 0's winner.
  stage 2: mu search again, lambda fixed to stage 1's winner, search
           range narrowed to stage 0's winner +/- narrow_width decades.
  stage 3: lambda search again, mu fixed to stage 2's winner, search
           range narrowed to stage 1's winner +/- narrow_width decades.

The result is stage 3's own winning (lambda1, lambda_tsv) together with
stage 2's own winning (mu_sq, mu_abs) and gains -- no extra "production"
TotalImaging run is performed afterward, since all needed state (image,
gains) is already available in memory from the stages themselves.
"""
from __future__ import annotations

import math
from collections import namedtuple

import numpy as np
import optuna

from .gains import Gains
from . import totalimaging
from .selfcal import update_visibility

GainTarget = namedtuple('GainTarget', ['sigma_ph', 'sigma_amp'])

# The two threshold combinations tested in Ikeda et al. 2025, section 3.4.
GAIN_TARGETS = {
    'small_variance': GainTarget(sigma_ph=5.0, sigma_amp=0.05),
    'large_variance': GainTarget(sigma_ph=15.0, sigma_amp=0.10),
}

GainSearchResult = namedtuple(
    'GainSearchResult', ['mu_sq', 'mu_abs', 'cost', 'gains', 'image']
)
ImagingSearchResult = namedtuple(
    'ImagingSearchResult', ['l1', 'ltsv', 'image']
)
StagedSearchResult = namedtuple(
    'StagedSearchResult',
    ['l1', 'ltsv', 'mu_sq', 'mu_abs', 'gains', 'image', 'stages']
)


def gain_dispersion(gain: np.ndarray):
    """(sigma_ph_deg, sigma_amp) as defined in Ikeda et al. 2025, section
    3.4: sigma_ph is the standard deviation of the gain phase (degrees),
    sigma_amp is the standard deviation of the gain amplitude.
    """
    sigma_ph = float(np.degrees(np.angle(gain).std()))
    sigma_amp = float(np.abs(gain).std())
    return sigma_ph, sigma_amp


def gain_dispersion_cost(gain: np.ndarray, target: GainTarget) -> float:
    """Eq. 17: (sigma_ph/sigma_ph* - 1)^2 + (sigma_amp/sigma_amp* - 1)^2.

    Deliberately a plain squared relative deviation (no hinge/soft-
    constraint band) -- confirmed against the paper's actual equation 17,
    which has no such margin, unlike the eq. 16 lambda criterion.
    """
    sigma_ph, sigma_amp = gain_dispersion(gain)
    return (sigma_ph / target.sigma_ph - 1.0) ** 2 + (sigma_amp / target.sigma_amp - 1.0) ** 2


def _log_grid(exp_range):
    low, high = exp_range
    return (10.0 ** np.arange(low, high + 1)).tolist()


def search_gain_regularizers(
        imager, antenna1, antenna2, time, vis_org, sigma,
        l1: float, ltsv: float, target: GainTarget,
        mu_sq_exp_range=(-6, 4), mu_abs_exp_range=(-3, 7),
        selfcal_num: int = 3, bayesopt_maxiter: int = 30,
        imaging_maxiter: int = 300, imaging_eps: float = 1.0e-4,
        selfcal_maxiter: int = 2000, selfcal_eps: float = 1.0e-6,
        total_eps: float = 1.0e-2,
        nonnegative: bool = True, scalehyperparam: bool = False, nthreads: int = 1,
) -> GainSearchResult:
    """
    Search (mu_sq, mu_abs) via Optuna: each trial runs totalimaging.run()
    from fresh (gain=1+0j) initial gains with l1/ltsv held fixed, and is
    scored by gain_dispersion_cost() against `target` (eq. 17).

    imager -- a priism SparseModelingImager with .working_set.u/v already
              set (rdata/idata get overwritten every trial by
              totalimaging.run(), so their initial values don't matter)
    antenna1, antenna2, time -- gain metadata for the same M visibilities
              as vis_org/sigma (see msreader.read_ms_for_selfcal); a new
              Gains instance is built from these fresh for every trial,
              so trials never leak gain state into each other
    l1, ltsv -- held fixed for every trial in this search
    mu_sq_exp_range, mu_abs_exp_range -- (low, high) decade exponents;
              the search grid is 10**low .. 10**high inclusive, integer
              steps (matches Ikeda et al. 2025's own log-integer encoding)
    selfcal_num, imaging_maxiter, imaging_eps, selfcal_maxiter,
    selfcal_eps, total_eps, nonnegative, scalehyperparam, nthreads --
              passed through to totalimaging.run() every trial
    """
    mu_sq_list = _log_grid(mu_sq_exp_range)
    mu_abs_list = _log_grid(mu_abs_exp_range)

    best = {'cost': np.inf}

    def objective(trial):
        mu_sq = mu_sq_list[trial.suggest_int('mu_sq_index', 0, len(mu_sq_list) - 1)]
        mu_abs = mu_abs_list[trial.suggest_int('mu_abs_index', 0, len(mu_abs_list) - 1)]

        gains = Gains(antenna1, antenna2, time)
        result = totalimaging.run(
            imager=imager, vis_org=vis_org, sigma=sigma, gains=gains,
            l1=l1, ltsv=ltsv, mu_sq=mu_sq, mu_abs=mu_abs,
            selfcal_num=selfcal_num, imaging_maxiter=imaging_maxiter, imaging_eps=imaging_eps,
            selfcal_maxiter=selfcal_maxiter, selfcal_eps=selfcal_eps, total_eps=total_eps,
            nonnegative=nonnegative, scalehyperparam=scalehyperparam, nthreads=nthreads,
        )
        cost = gain_dispersion_cost(gains.gain, target)

        if cost < best['cost']:
            best.update(cost=cost, mu_sq=mu_sq, mu_abs=mu_abs, gains=gains, image=result.image)

        print(f'mu_sq={mu_sq:.3g} mu_abs={mu_abs:.3g}: cost={cost:.4g}')
        return cost

    study = optuna.create_study()
    study.optimize(objective, n_trials=bayesopt_maxiter)

    return GainSearchResult(
        mu_sq=best['mu_sq'], mu_abs=best['mu_abs'], cost=best['cost'],
        gains=best['gains'], image=best['image']
    )


def search_imaging_regularizers(
        imager, l1_exp_range=(-15, 15), ltsv_exp_range=(-15, 15),
        bayesopt_maxiter: int = 30, imaging_maxiter: int = 500, imaging_eps: float = 1.0e-4,
        ellipse_th: float = 0.995, cos_th: float = 0.99,
        nonnegative: bool = True, scalehyperparam: bool = False,
        imageprefix: str = 'image_paramsearch',
) -> ImagingSearchResult:
    """
    Search (l1, ltsv) via priism's own optimizeparameters(criterion=
    'ellipsoid', optimizer='bayesian') (eq. 16) -- no self-calibration is
    performed here; imager.working_set must already hold whatever
    (possibly gain-corrected) visibility this search should evaluate
    against (see update_visibility).

    l1_exp_range, ltsv_exp_range -- (low, high) decade exponents, same
              convention as search_gain_regularizers
    ellipse_th, cos_th -- default to the exact values used in Ikeda et
              al. 2025 section 4 (0.995, 0.99), which differ slightly
              from priism's own general-purpose default of (0.99, 0.99)
    imageprefix -- passed to optimizeparameters(); writes FITS files as
              a side effect (imagepolicy='best' below removes all but
              the final one). Point this at a scratch directory if
              calling from a test or a batch job.

    Returns the winning (l1, ltsv) together with the resulting image,
    obtained via one extra explicit solve() at that point (rather than
    reading back the FITS file optimizeparameters() already wrote) so
    that imager.imagearray/working_set reliably reflect the winner even
    though Optuna's last-evaluated trial isn't necessarily the best one.
    """
    l1_list = _log_grid(l1_exp_range)
    ltsv_list = _log_grid(ltsv_exp_range)

    best = imager.optimizeparameters(
        l1_list=l1_list, ltsv_list=ltsv_list,
        criterion='ellipsoid', optimizer='bayesian',
        bayesopt_maxiter=bayesopt_maxiter,
        ellipse_th=ellipse_th, cos_th=cos_th,
        maxiter=imaging_maxiter, eps=imaging_eps,
        nonnegative=nonnegative, scalehyperparam=scalehyperparam,
        imageprefix=imageprefix, imagepolicy='best', summarize=False,
    )
    best_l1, best_ltsv = best['L1'], best['Ltsv']

    imager.solve(best_l1, best_ltsv, maxiter=imaging_maxiter, eps=imaging_eps,
                 nonnegative=nonnegative, scalehyperparam=scalehyperparam)
    image = np.squeeze(imager.imagearray.data)

    return ImagingSearchResult(l1=best_l1, ltsv=best_ltsv, image=image)


def _nearest_exponent(value: float) -> int:
    return int(round(math.log10(value)))


def run_staged_parameter_search(
        imager, antenna1, antenna2, time, vis_org, sigma,
        target: GainTarget,
        l1_exp_range=(-15, 15), ltsv_exp_range=(-15, 15),
        mu_sq_exp_range=(-6, 4), mu_abs_exp_range=(-3, 7),
        narrow_width: int = 3,
        selfcal_num_stage0: int = 3, selfcal_num_stage2: int = 3,
        bayesopt_maxiter_stage0: int = 30, bayesopt_maxiter_stage1: int = 30,
        bayesopt_maxiter_stage2: int = 30, bayesopt_maxiter_stage3: int = 30,
        imaging_maxiter: int = 300, imaging_eps: float = 1.0e-4,
        selfcal_maxiter: int = 2000, selfcal_eps: float = 1.0e-6, total_eps: float = 1.0e-2,
        ellipse_th: float = 0.995, cos_th: float = 0.99,
        nonnegative: bool = True, scalehyperparam: bool = False, nthreads: int = 1,
        imageprefix: str = 'image_paramsearch',
) -> StagedSearchResult:
    """
    0 -> 1 -> 2 -> 1 staged search for (lambda1, lambda_tsv, mu_sq,
    mu_abs) -- see this module's docstring for the stage definitions and
    why the sequence stops after stage 3 rather than repeating further.

    imager -- a priism SparseModelingImager with .working_set.u/v/weight
              already set from the raw (uncorrected) visibility; its
              rdata/idata get overwritten in place as the stages proceed
    antenna1, antenna2, time, vis_org, sigma -- see
              search_gain_regularizers/msreader.read_ms_for_selfcal
    target -- gain dispersion target (see GAIN_TARGETS for the two
              combinations from Ikeda et al. 2025)
    narrow_width -- decades +/- around the previous stage's winner used
              for stage 2/3's search ranges (matches the historical
              driver scripts' own choice of 3)
    selfcal_num_stage0, selfcal_num_stage2 -- kept as separate arguments
              (stage 0 solves from scratch, stage 2 refines) but default
              to the same value, matching the same TotalImaging round
              budget throughout unless deliberately overridden
    bayesopt_maxiter_stage{0,1,2,3} -- kept separate per stage for the
              same reason; all default to 30 (Ikeda et al. 2025's own
              trial count)

    Returns the final (l1, ltsv) from stage 3 and (mu_sq, mu_abs, gains)
    from stage 2 -- these are mutually consistent in the sense that
    stage 3's own winning trial is imaged with stage 2's gain correction
    already applied, matching how the historical procedure terminated
    (no separate non-optimized "production" run needed; see this
    module's docstring).
    """
    stage0 = search_gain_regularizers(
        imager, antenna1, antenna2, time, vis_org, sigma,
        l1=1.0, ltsv=1.0, target=target,
        mu_sq_exp_range=mu_sq_exp_range, mu_abs_exp_range=mu_abs_exp_range,
        selfcal_num=selfcal_num_stage0, bayesopt_maxiter=bayesopt_maxiter_stage0,
        imaging_maxiter=imaging_maxiter, imaging_eps=imaging_eps,
        selfcal_maxiter=selfcal_maxiter, selfcal_eps=selfcal_eps, total_eps=total_eps,
        nonnegative=nonnegative, scalehyperparam=scalehyperparam, nthreads=nthreads,
    )

    vis_cal = update_visibility(stage0.gains, vis_org)
    imager.working_set.rdata = vis_cal.real.copy()
    imager.working_set.idata = vis_cal.imag.copy()
    stage1 = search_imaging_regularizers(
        imager, l1_exp_range=l1_exp_range, ltsv_exp_range=ltsv_exp_range,
        bayesopt_maxiter=bayesopt_maxiter_stage1,
        imaging_maxiter=imaging_maxiter, imaging_eps=imaging_eps,
        ellipse_th=ellipse_th, cos_th=cos_th,
        nonnegative=nonnegative, scalehyperparam=scalehyperparam,
        imageprefix=imageprefix + '_stage1',
    )

    log_mu_sq0 = _nearest_exponent(stage0.mu_sq)
    log_mu_abs0 = _nearest_exponent(stage0.mu_abs)
    stage2 = search_gain_regularizers(
        imager, antenna1, antenna2, time, vis_org, sigma,
        l1=stage1.l1, ltsv=stage1.ltsv, target=target,
        mu_sq_exp_range=(log_mu_sq0 - narrow_width, log_mu_sq0 + narrow_width),
        mu_abs_exp_range=(log_mu_abs0 - narrow_width, log_mu_abs0 + narrow_width),
        selfcal_num=selfcal_num_stage2, bayesopt_maxiter=bayesopt_maxiter_stage2,
        imaging_maxiter=imaging_maxiter, imaging_eps=imaging_eps,
        selfcal_maxiter=selfcal_maxiter, selfcal_eps=selfcal_eps, total_eps=total_eps,
        nonnegative=nonnegative, scalehyperparam=scalehyperparam, nthreads=nthreads,
    )

    vis_cal = update_visibility(stage2.gains, vis_org)
    imager.working_set.rdata = vis_cal.real.copy()
    imager.working_set.idata = vis_cal.imag.copy()
    log_l1_1 = _nearest_exponent(stage1.l1)
    log_ltsv_1 = _nearest_exponent(stage1.ltsv)
    stage3 = search_imaging_regularizers(
        imager,
        l1_exp_range=(log_l1_1 - narrow_width, log_l1_1 + narrow_width),
        ltsv_exp_range=(log_ltsv_1 - narrow_width, log_ltsv_1 + narrow_width),
        bayesopt_maxiter=bayesopt_maxiter_stage3,
        imaging_maxiter=imaging_maxiter, imaging_eps=imaging_eps,
        ellipse_th=ellipse_th, cos_th=cos_th,
        nonnegative=nonnegative, scalehyperparam=scalehyperparam,
        imageprefix=imageprefix + '_stage3',
    )

    return StagedSearchResult(
        l1=stage3.l1, ltsv=stage3.ltsv,
        mu_sq=stage2.mu_sq, mu_abs=stage2.mu_abs,
        gains=stage2.gains, image=stage3.image,
        stages=dict(stage0=stage0, stage1=stage1, stage2=stage2, stage3=stage3),
    )

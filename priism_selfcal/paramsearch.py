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

Sequencing follows the paper's own explicit section 3.4 procedure:

  "(1) Find the best combination of {lambda1, lambda2} for the
       visibility without self-calibration.
   (2) Fix {lambda1, lambda2} for those found in step 1 and search the
       best combination of {mu1, mu2}.
   (3) Fix {mu1, mu2} for those found in step 2 and search the best
       combination of {lambda1, lambda2}."

i.e.:

  stage 0: lambda search with gain fixed at 1+0j -- no self-calibration
           at all yet, priism's ellipsoid criterion on the raw
           (uncorrected) visibility.
  stage 1: mu search from scratch (gain re-initialized to 1+0j every
           trial), lambda fixed to stage 0's winner.
  stage 2: lambda search again, gain fixed to stage 1's winner, search
           range narrowed to stage 0's winner +/- narrow_width decades.

`n_refine_rounds` (default 1) controls whether stages 1-2 repeat: the
paper allows repeating them ("we repeated steps 2 and 3 ... because the
image does not change greatly after a few iterations"), but the actual
historical driver scripts (old/python/*_step0.py..step3.py) only ever
ran a single round in practice, traced via their hardcoded-constant
provenance chain (2026-08-21) -- n_refine_rounds=1 matches that; raise
it (e.g. to 2) to repeat the mu/lambda refinement an extra time.

An earlier version of this module mistakenly started from a mu search
(following old/python's own "step0.py" naming, which already assumes a
fixed lambda=1 to bootstrap its mu search -- i.e. it *is* stage 1 above,
not stage 0) rather than the paper's actual stage 0. Corrected 2026-08-21.

The result is the final round's own winning (lambda1, lambda_tsv, mu_sq,
mu_abs, gains, image) -- no extra "production" TotalImaging run is
performed afterward, since all needed state is already available in
memory from the stages themselves.
"""
from __future__ import annotations

import glob
import math
import os
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


def gain_outlier_penalty(
        gain: np.ndarray,
        amp_bounds: tuple[float, float] = (0.5, 1.5),
        phase_bound_deg: float = 90.0,
) -> float:
    """Penalize any *individual* gain whose amplitude/phase falls outside
    the given bounds -- not part of Ikeda et al. 2025's eq. 17, which only
    scores the aggregate (population) standard deviation and can be
    satisfied by a solution that still contains a few extreme outliers.

    Confirmed 2026-08-21 on real HD142527 data: a mu_sq/mu_abs combination
    (1e3/1e4) gave sigma_amp=0.092, close to the 0.10 target, while
    individual |gain| ranged up to 2.43 -- a value gain_dispersion_cost
    alone has no way to penalize, since a handful of outliers barely move
    the population standard deviation. Real ALMA gains should stay near
    1+0j (see this package's own test/README conventions, which already
    use amp in [0.5, 1.5] as the "physically plausible" bound); this
    penalty makes the mu search actively avoid solutions that violate
    that expectation, on top of (not instead of) hitting the target
    dispersion.

    amp_bounds -- (low, high) acceptable |gain| range; anything outside
              this is squared-hinge penalized
    phase_bound_deg -- acceptable |angle(gain)| range in degrees, symmetric
              around 0; deliberately looser than amp_bounds since large
              overall phase offsets are less immediately implausible than
              large amplitude excursions (a global phase reference is
              somewhat arbitrary), but a truly wild individual outlier
              should still be penalized
    """
    amp = np.abs(gain)
    low, high = amp_bounds
    amp_over = np.maximum(amp - high, 0.0)
    amp_under = np.maximum(low - amp, 0.0)

    phase_deg = np.degrees(np.angle(gain))
    phase_over = np.maximum(np.abs(phase_deg) - phase_bound_deg, 0.0)

    return float(np.sum(amp_over ** 2) + np.sum(amp_under ** 2) + np.sum(phase_over ** 2))


def _log_grid(exp_range):
    low, high = exp_range
    return (10.0 ** np.arange(low, high + 1)).tolist()


def search_gain_regularizers(
        imager, antenna1, antenna2, time, vis_org, sigma,
        l1: float, ltsv: float, target: GainTarget,
        mu_sq_exp_range=(-6, 4), mu_abs_exp_range=(-3, 7),
        selfcal_num: int = 3,
        bayesopt_n_startup_trials: int = 20, bayesopt_n_search_trials: int = 30,
        imaging_maxiter: int = 300, imaging_eps: float = 1.0e-4,
        selfcal_maxiter: int = 2000, selfcal_eps: float = 1.0e-6,
        total_eps: float = 1.0e-2,
        outlier_amp_bounds: tuple[float, float] = (0.5, 1.5),
        outlier_phase_bound_deg: float = 90.0,
        outlier_penalty_scale: float = 100.0,
        nonnegative: bool = True, scalehyperparam: bool = False, nthreads: int = 1,
) -> GainSearchResult:
    """
    Search (mu_sq, mu_abs) via Optuna: each trial runs totalimaging.run()
    from fresh (gain=1+0j) initial gains with l1/ltsv held fixed, and is
    scored by gain_dispersion_cost() against `target` (eq. 17) plus
    outlier_penalty_scale * gain_outlier_penalty() (2026-08-21 addition,
    not in the paper -- see gain_outlier_penalty's docstring for why the
    aggregate-only eq. 17 criterion isn't enough on its own).

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
    bayesopt_n_startup_trials -- number of purely-random trials Optuna's
              TPESampler runs before it starts using its surrogate model
              to guide sampling. Total trials = n_startup + n_search; if
              n_search were 0 this would just be random search (verified
              2026-08-21: a 10-trial budget with Optuna's own default
              n_startup_trials=10 meant *every* trial of a real-data run
              was random, with no Bayesian guidance ever applied).
    bayesopt_n_search_trials -- number of TPE-guided trials run after the
              startup phase
    outlier_amp_bounds, outlier_phase_bound_deg -- passed to
              gain_outlier_penalty()
    outlier_penalty_scale -- weight on gain_outlier_penalty() relative to
              gain_dispersion_cost(); default 100 deliberately makes even
              a single bad outlier (e.g. one |gain| at 2x the amp_bounds
              margin) comparable to or larger than a typical eq.-17
              mismatch, since the point of this term is to make outliers
              essentially unacceptable rather than a minor tiebreaker
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
        dispersion_cost = gain_dispersion_cost(gains.gain, target)
        outlier_cost = outlier_penalty_scale * gain_outlier_penalty(
            gains.gain, amp_bounds=outlier_amp_bounds, phase_bound_deg=outlier_phase_bound_deg
        )
        cost = dispersion_cost + outlier_cost

        if cost < best['cost']:
            best.update(cost=cost, mu_sq=mu_sq, mu_abs=mu_abs, gains=gains, image=result.image)

        print(f'mu_sq={mu_sq:.3g} mu_abs={mu_abs:.3g}: '
              f'dispersion={dispersion_cost:.4g} outlier={outlier_cost:.4g} cost={cost:.4g}')
        return cost

    sampler = optuna.samplers.TPESampler(n_startup_trials=bayesopt_n_startup_trials)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=bayesopt_n_startup_trials + bayesopt_n_search_trials)

    return GainSearchResult(
        mu_sq=best['mu_sq'], mu_abs=best['mu_abs'], cost=best['cost'],
        gains=best['gains'], image=best['image']
    )


def search_imaging_regularizers(
        imager, l1_exp_range=(-15, 15), ltsv_exp_range=(-15, 15),
        bayesopt_n_startup_trials: int = 20, bayesopt_n_search_trials: int = 30,
        imaging_maxiter: int = 500, imaging_eps: float = 1.0e-4,
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
    bayesopt_n_startup_trials, bayesopt_n_search_trials -- see
              search_gain_regularizers's docstring; passed through to
              optimizeparameters()'s own bayesopt_n_startup_trials
              (requires priism's selfcal-gain-metadata branch or later,
              which added this parameter -- 2026-08-21)
    ellipse_th, cos_th -- default to the exact values used in Ikeda et
              al. 2025 section 4 (0.995, 0.99), which differ slightly
              from priism's own general-purpose default of (0.99, 0.99)
    imageprefix -- passed to optimizeparameters(); writes one FITS file
              per trial as a side effect. All of them (plus the
              "<imageprefix>.<suffix>" best-image copy optimizeparameters()
              itself makes) are removed once this function has extracted
              the winning image array, regardless of optimizer/criterion.
              Point this at a scratch directory if calling from a test or
              a batch job.

    Returns the winning (l1, ltsv) together with the resulting image,
    obtained via one extra explicit solve() at that point (rather than
    reading back the FITS file optimizeparameters() already wrote) so
    that imager.imagearray/working_set reliably reflect the winner even
    though Optuna's last-evaluated trial isn't necessarily the best one.
    """
    l1_list = _log_grid(l1_exp_range)
    ltsv_list = _log_grid(ltsv_exp_range)

    # imagepolicy='best' is avoided here: optimizeparameters()'s own
    # cleanup deletes every entry in its internal trial-image list
    # without deduplicating it first, so it raises FileNotFoundError
    # whenever optimizer='bayesian' (Optuna) samples the same grid point
    # twice -- a pre-existing priism bug, unrelated to self-cal, hit
    # while running this package's staged search on real data
    # (2026-08-21). imagepolicy='full' skips that internal cleanup
    # entirely; we do our own (glob-based, so duplicate paths are naturally
    # deduplicated) cleanup below instead.
    best = imager.optimizeparameters(
        l1_list=l1_list, ltsv_list=ltsv_list,
        criterion='ellipsoid', optimizer='bayesian',
        bayesopt_maxiter=bayesopt_n_startup_trials + bayesopt_n_search_trials,
        bayesopt_n_startup_trials=bayesopt_n_startup_trials,
        ellipse_th=ellipse_th, cos_th=cos_th,
        maxiter=imaging_maxiter, eps=imaging_eps,
        nonnegative=nonnegative, scalehyperparam=scalehyperparam,
        imageprefix=imageprefix, imagepolicy='full', summarize=False,
    )
    best_l1, best_ltsv = best['L1'], best['Ltsv']

    imager.solve(best_l1, best_ltsv, maxiter=imaging_maxiter, eps=imaging_eps,
                 nonnegative=nonnegative, scalehyperparam=scalehyperparam)
    image = np.squeeze(imager.imagearray.data)

    for leftover in glob.glob(imageprefix + '*'):
        try:
            os.remove(leftover)
        except OSError:
            pass

    return ImagingSearchResult(l1=best_l1, ltsv=best_ltsv, image=image)


def _nearest_exponent(value: float) -> int:
    return int(round(math.log10(value)))


def run_staged_parameter_search(
        imager, antenna1, antenna2, time, vis_org, sigma,
        target: GainTarget,
        l1_exp_range=(-15, 15), ltsv_exp_range=(-15, 15),
        mu_sq_exp_range=(-6, 4), mu_abs_exp_range=(-3, 7),
        narrow_width: int = 3, n_refine_rounds: int = 1,
        selfcal_num: int = 3,
        bayesopt_n_startup_trials: int = 20, bayesopt_n_search_trials: int = 30,
        imaging_maxiter: int = 300, imaging_eps: float = 1.0e-4,
        selfcal_maxiter: int = 2000, selfcal_eps: float = 1.0e-6, total_eps: float = 1.0e-2,
        ellipse_th: float = 0.995, cos_th: float = 0.99,
        outlier_amp_bounds: tuple[float, float] = (0.5, 1.5),
        outlier_phase_bound_deg: float = 90.0,
        outlier_penalty_scale: float = 100.0,
        nonnegative: bool = True, scalehyperparam: bool = False, nthreads: int = 1,
        imageprefix: str = 'image_paramsearch',
) -> StagedSearchResult:
    """
    Stage 0 -> (stage 1 -> stage 2) x n_refine_rounds staged search for
    (lambda1, lambda_tsv, mu_sq, mu_abs) -- see this module's docstring
    for the stage definitions, and why stage 0 is a lambda search (not a
    mu search).

    imager -- a priism SparseModelingImager with .working_set.u/v/weight
              already set from the raw (uncorrected) visibility; its
              rdata/idata get overwritten in place as the stages proceed
              (stage 0 runs directly against the untouched raw values,
              since gain=1+0j means no correction at all)
    antenna1, antenna2, time, vis_org, sigma -- see
              search_gain_regularizers/msreader.read_ms_for_selfcal
    target -- gain dispersion target (see GAIN_TARGETS for the two
              combinations from Ikeda et al. 2025)
    narrow_width -- decades +/- around the previous round's winner used
              to narrow both the lambda and mu search ranges from the
              second round onward (matches the historical driver
              scripts' own choice of 3)
    n_refine_rounds -- how many times to run (mu search, lambda search);
              default 1 matches the historical driver scripts' actual
              practice (see module docstring); the paper itself allows
              repeating further
    selfcal_num -- passed to every mu-search round's totalimaging.run()
              calls
    bayesopt_n_startup_trials, bayesopt_n_search_trials -- shared by every
              lambda- and mu-search stage; see search_gain_regularizers's
              docstring for why both matter (a too-small n_search_trials
              relative to n_startup_trials means little to no actual
              Bayesian-guided search ever happens)
    outlier_amp_bounds, outlier_phase_bound_deg, outlier_penalty_scale --
              passed to every mu-search round's gain_outlier_penalty();
              see search_gain_regularizers's docstring for why this
              matters on top of the target dispersion alone

    Returns the final round's own winning (l1, ltsv, mu_sq, mu_abs,
    gains, image) -- these are mutually consistent in the sense that the
    final lambda search's own winning trial is imaged with that round's
    gain correction already applied, so no separate non-optimized
    "production" run is needed afterward.
    """
    stage0 = search_imaging_regularizers(
        imager, l1_exp_range=l1_exp_range, ltsv_exp_range=ltsv_exp_range,
        bayesopt_n_startup_trials=bayesopt_n_startup_trials,
        bayesopt_n_search_trials=bayesopt_n_search_trials,
        imaging_maxiter=imaging_maxiter, imaging_eps=imaging_eps,
        ellipse_th=ellipse_th, cos_th=cos_th,
        nonnegative=nonnegative, scalehyperparam=scalehyperparam,
        imageprefix=imageprefix + '_stage0',
    )
    stages = {'stage0': stage0}

    current_l1, current_ltsv = stage0.l1, stage0.ltsv
    cur_mu_sq_range, cur_mu_abs_range = mu_sq_exp_range, mu_abs_exp_range
    mu_stage = None
    lambda_stage = stage0

    for round_idx in range(n_refine_rounds):
        mu_stage = search_gain_regularizers(
            imager, antenna1, antenna2, time, vis_org, sigma,
            l1=current_l1, ltsv=current_ltsv, target=target,
            mu_sq_exp_range=cur_mu_sq_range, mu_abs_exp_range=cur_mu_abs_range,
            selfcal_num=selfcal_num,
            bayesopt_n_startup_trials=bayesopt_n_startup_trials,
            bayesopt_n_search_trials=bayesopt_n_search_trials,
            imaging_maxiter=imaging_maxiter, imaging_eps=imaging_eps,
            selfcal_maxiter=selfcal_maxiter, selfcal_eps=selfcal_eps, total_eps=total_eps,
            outlier_amp_bounds=outlier_amp_bounds,
            outlier_phase_bound_deg=outlier_phase_bound_deg,
            outlier_penalty_scale=outlier_penalty_scale,
            nonnegative=nonnegative, scalehyperparam=scalehyperparam, nthreads=nthreads,
        )
        stages[f'mu_round{round_idx}'] = mu_stage

        vis_cal = update_visibility(mu_stage.gains, vis_org)
        imager.working_set.rdata = vis_cal.real.copy()
        imager.working_set.idata = vis_cal.imag.copy()

        log_l1 = _nearest_exponent(current_l1)
        log_ltsv = _nearest_exponent(current_ltsv)
        lambda_stage = search_imaging_regularizers(
            imager,
            l1_exp_range=(log_l1 - narrow_width, log_l1 + narrow_width),
            ltsv_exp_range=(log_ltsv - narrow_width, log_ltsv + narrow_width),
            bayesopt_n_startup_trials=bayesopt_n_startup_trials,
            bayesopt_n_search_trials=bayesopt_n_search_trials,
            imaging_maxiter=imaging_maxiter, imaging_eps=imaging_eps,
            ellipse_th=ellipse_th, cos_th=cos_th,
            nonnegative=nonnegative, scalehyperparam=scalehyperparam,
            imageprefix=f'{imageprefix}_round{round_idx}',
        )
        stages[f'lambda_round{round_idx}'] = lambda_stage

        current_l1, current_ltsv = lambda_stage.l1, lambda_stage.ltsv
        log_mu_sq = _nearest_exponent(mu_stage.mu_sq)
        log_mu_abs = _nearest_exponent(mu_stage.mu_abs)
        cur_mu_sq_range = (log_mu_sq - narrow_width, log_mu_sq + narrow_width)
        cur_mu_abs_range = (log_mu_abs - narrow_width, log_mu_abs + narrow_width)

    return StagedSearchResult(
        l1=current_l1, ltsv=current_ltsv,
        mu_sq=mu_stage.mu_sq, mu_abs=mu_stage.mu_abs,
        gains=mu_stage.gains, image=lambda_stage.image,
        stages=stages,
    )

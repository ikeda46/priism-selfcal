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
TotalImaging equivalent: alternate between imaging (priism's MFISTA) and
self-calibration until the combined cost (imaging cost + gain regularizer)
stops improving.

Ported from old/python/sparseimaging/totalimaging.py::TotalImaging.run,
using priism's own solve() (which already warm-starts each imaging solve
from the previous result via its internal initialimage, matching the old
code's xinit=x) and priism_selfcal's already-validated run_self_calibration
in place of the old code's C++-backed SelfCalibration.run.

One difference from the old reference: priism's solve() re-estimates its
own Lipschitz constant (cinit) each call rather than carrying over the
previous solve's converged value (the old code did
`imaging_params.Lip_const = mfista_result.Lip_const`) -- priism does not
currently expose a hook for this. Since it only affects how many MFISTA
iterations are needed to reconverge after a small visibility change, not
correctness, this is left as a possible future optimization rather than
a change to priism itself.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

from priism.core import pysparseimaging
from priism.core.sparseimagingnufft import SparseImagingInputsNUFFT

from .gains import Gains
from .selfcal import run_self_calibration, update_visibility

TotalImagingResult = namedtuple(
    'TotalImagingResult',
    ['image', 'cost', 'converged', 'num_selfcal_iterations']
)


def _imaging_cost(imager, vis_cal, sigma, l1, ltsv, nonnegative, nthreads):
    nx, ny = imager.imparam.imsize
    u_rad, v_rad = SparseImagingInputsNUFFT.convert_uv(
        imager.imparam, imager.working_set.u, imager.working_set.v
    )
    x = np.squeeze(imager.imagearray.data)
    _, _, _, _, final_cost = pysparseimaging.calc_costs_nufft(
        M=vis_cal.size, Nx=nx, Ny=ny, u_dx=u_rad, v_dy=v_rad,
        vis_r=vis_cal.real, vis_i=vis_cal.imag, vis_std=sigma,
        lambda_l1=l1, lambda_tv=0.0, lambda_tsv=ltsv, nonneg=nonnegative,
        xvec=x, nthreads=nthreads
    )
    return x, final_cost


def run(
        imager,
        vis_org: np.ndarray,
        sigma: np.ndarray,
        gains: Gains,
        l1: float,
        ltsv: float,
        mu_sq: float,
        mu_abs: float,
        rho_init: float | None = None,
        selfcal_num: int = 10,
        imaging_maxiter: int = 500,
        imaging_eps: float = 1.0e-4,
        selfcal_maxiter: int = 2000,
        selfcal_eps: float = 1.0e-6,
        total_eps: float = 1.0e-2,
        nonnegative: bool = True,
        scalehyperparam: bool = False,
        nthreads: int = 1,
) -> TotalImagingResult:
    """
    Alternate imaging <-> self-calibration.

    Args:
        imager -- a priism SparseModelingImager (or subclass) with
                  .working_set and .imparam already set; .working_set.u/v
                  are used as fixed pixel-grid uv coordinates throughout,
                  and .working_set.rdata/idata are overwritten in place
                  each round with the current gain-corrected visibility.
        vis_org -- raw (not gain-corrected) visibility, length M, held
                   fixed throughout (self-calibration always corrects a
                   copy of this, never the running vis_cal)
        sigma -- per-visibility noise std dev, length M
        gains -- a Gains instance built from the same M visibilities'
                 station1/station2/time; its .gain is updated in place
                 and reflects the final solution on return
        l1, ltsv -- imaging regularization weights, passed to
                    imager.solve() every round (scalehyperparam controls
                    whether they're used as-is or priism-rescaled)
        mu_sq, mu_abs -- gain-smoothness / gain-amplitude-smoothness
                         regularization weights (mu1/mu2 in Ikeda et al. 2025)
        rho_init -- ADMM penalty for self-calibration; None estimates it
                    from the data scale each self-cal call (see
                    pyselfcal.estimate_rho_init)
        selfcal_num -- maximum number of imaging<->self-cal rounds.
                       Increasing this past the default of 10 is not
                       guaranteed to help: self-calibration has enough
                       freedom in the per-station gains to keep "explaining
                       away" real image structure as gain/phase error if
                       allowed to alternate too many times, which can drift
                       the image in a worse direction even as the tracked
                       cost keeps decreasing. The old reference
                       (old/python/example_HD142527.py) always ran with a
                       small fixed budget (10) rather than iterating to
                       convergence; treat selfcal_num as a deliberately
                       conservative cap, not a knob to maximize.
        imaging_maxiter, imaging_eps -- passed to imager.solve()
        selfcal_maxiter, selfcal_eps -- passed to run_self_calibration()
        total_eps -- relative-change threshold on (imaging cost + gain
                     regularizer) for declaring convergence (early-exit
                     safety valve). In the old reference, the equivalent
                     default (1e-6) was tight enough that it was rarely, if
                     ever, actually reached within the 10-round budget in
                     practice (empirically ~40+ rounds needed for the cost
                     to flatten to that level on a representative synthetic
                     problem, 2026-08-21) -- i.e. selfcal_num, not total_eps,
                     was the real stopping mechanism. This port defaults to
                     a looser 1e-2 so the early-exit can actually fire when
                     the cost has genuinely flattened, but it remains a
                     secondary safety valve on top of selfcal_num, not a
                     substitute for it -- do not rely on loosening total_eps
                     further as a way to run more effective rounds; raise
                     selfcal_num deliberately instead, and inspect the
                     resulting gains/image (see gains.plot_gains) before
                     trusting the result.
        nonnegative, scalehyperparam -- passed to imager.solve()
        nthreads -- threads finufft may use per NUFFT call

    Returns:
        TotalImagingResult(image, cost, converged, num_selfcal_iterations)
    """
    vis_cal = update_visibility(gains, vis_org)
    imager.working_set.rdata = vis_cal.real.copy()
    imager.working_set.idata = vis_cal.imag.copy()
    imager.solve(l1, ltsv, maxiter=imaging_maxiter, eps=imaging_eps,
                 nonnegative=nonnegative, scalehyperparam=scalehyperparam,
                 storeinitialimage=True, overwriteinitialimage=True)
    x, cost = _imaging_cost(imager, vis_cal, sigma, l1, ltsv, nonnegative, nthreads)

    converged = False
    num_iterations = 0

    for i in range(selfcal_num):
        num_iterations = i + 1

        run_self_calibration(
            gains=gains,
            u_pix=imager.working_set.u, v_pix=imager.working_set.v,
            vis=vis_org, sigma=sigma, xin=x, imsize=tuple(imager.imparam.imsize),
            mu_sq=mu_sq, mu_abs=mu_abs, rho_init=rho_init,
            maxiter=selfcal_maxiter, eps=selfcal_eps, nthreads=nthreads,
        )
        reg = gains.regularizers()
        gain_regularizer = mu_sq * reg.sq_term + mu_abs * reg.abs_term

        vis_cal = update_visibility(gains, vis_org)
        imager.working_set.rdata = vis_cal.real.copy()
        imager.working_set.idata = vis_cal.imag.copy()
        imager.solve(l1, ltsv, maxiter=imaging_maxiter, eps=imaging_eps,
                     nonnegative=nonnegative, scalehyperparam=scalehyperparam,
                     storeinitialimage=True, overwriteinitialimage=True)
        x, imaging_cost = _imaging_cost(imager, vis_cal, sigma, l1, ltsv, nonnegative, nthreads)

        newcost = imaging_cost + gain_regularizer

        if abs(cost - newcost) / newcost < total_eps:
            converged = True
            cost = newcost
            break

        cost = newcost

    return TotalImagingResult(
        image=x, cost=cost, converged=converged, num_selfcal_iterations=num_iterations
    )

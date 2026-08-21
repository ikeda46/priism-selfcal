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
SelfCalibration.run equivalent: given the current image estimate, compute
the model visibility via priism's NUFFT forward operator and solve for
updated gains via the pyselfcal engine (Phase B).

Mirrors old/python/sparseimaging/selfcal.py::SelfCalibration.run and its
x2y_nufft helper, but calls into priism's own (already bug-fixed, already
validated) pysparseimaging.x2y_nufft instead of a separate C++ .so, and
priism_selfcal.pyselfcal.self_calibration (the Phase-B NumPy port) instead
of libselfcal.so.
"""
from __future__ import annotations

import logging
from collections import namedtuple
from types import SimpleNamespace

import numpy as np

from priism.core import pysparseimaging
from priism.core.sparseimagingnufft import SparseImagingInputsNUFFT

from .gains import Gains
from .pyselfcal import self_calibration, modify_visibility

logger = logging.getLogger(__name__)

SelfCalibrationResult = namedtuple('SelfCalibrationResult', ['finalcost', 'converged'])


def run_self_calibration(
        gains: Gains,
        u_pix: np.ndarray,
        v_pix: np.ndarray,
        vis: np.ndarray,
        sigma: np.ndarray,
        xin: np.ndarray,
        imsize: tuple[int, int],
        mu_sq: float,
        mu_abs: float,
        rho_init: float | None = None,
        maxiter: int = 1000,
        eps: float = 1.0e-5,
        nthreads: int = 1
) -> SelfCalibrationResult:
    """
    Update `gains.gain` in place from the current image estimate `xin`.

    Args:
        gains -- Gains instance (its .gain array is updated in place)
        u_pix, v_pix -- visibility (u, v) in priism's pixel-grid convention
                        (VisibilityWorkingSet.u/v), length M
        vis -- observed visibility (length M), raw (not gain-corrected)
        sigma -- per-visibility noise std dev (length M)
        xin -- current image estimate, shape imsize
        imsize -- (nx, ny), matching priism's imageparam.imsize convention
        mu_sq, mu_abs -- gain-smoothness / gain-amplitude-smoothness
                         regularization weights (mu1/mu2 in Ikeda et al. 2025;
                         these are NOT the imaging L1/TSV weights)
        rho_init -- initial ADMM penalty weight. None (default) estimates it
                    from the data scale (see pyselfcal.estimate_rho_init).
        maxiter, eps -- self-calibration solver iteration controls (eps is
                        relative, see pyselfcal.self_calibration)
        nthreads -- threads finufft may use for the forward NUFFT call

    Returns:
        SelfCalibrationResult(finalcost, converged). A warning is logged
        if the solver did not converge within maxiter/MAXOUTER -- callers
        should not silently trust gains.gain in that case.
    """

    imageparam = SimpleNamespace(imsize=imsize)
    u_rad, v_rad = SparseImagingInputsNUFFT.convert_uv(imageparam, u_pix, v_pix)
    y = pysparseimaging.x2y_nufft(u_rad, v_rad, xin, nthreads=nthreads)

    result = self_calibration(
        vis=vis,
        vis_std=sigma,
        y=y,
        ginit=gains.gain,
        gid_adj_t=gains.gid_adj_t,
        vid2gid_st=gains.vid2gid_st,
        time_tbl=gains.time_tbl,
        Stnum=gains.st_num,
        Tnum=gains.time_num,
        lambda_1=mu_sq,
        lambda_2=mu_abs,
        rho_init=rho_init,
        maxiter=maxiter,
        eps=eps
    )

    gains.gain = result.gain

    if not result.converged:
        logger.warning(
            "self-calibration did not converge (mu_sq=%s, mu_abs=%s, maxiter=%d, eps=%s); "
            "gains.gain reflects the last iterate, not a verified solution.",
            mu_sq, mu_abs, maxiter, eps
        )

    return SelfCalibrationResult(finalcost=result.finalcost, converged=result.converged)


def update_visibility(gains: Gains, vis: np.ndarray) -> np.ndarray:
    """
    Apply the current gains to the raw visibility.
    Mirrors old/python/sparseimaging/totalimaging.py::update_visibility;
    identical to pyselfcal.modify_visibility, exposed here under the name
    used by the (upcoming) TotalImaging-equivalent orchestration loop.
    """
    return modify_visibility(vis, gains.gain, gains.vid2gid_st)

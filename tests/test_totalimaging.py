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
End-to-end test of totalimaging.run() (the TotalImaging equivalent):
alternating imaging <-> self-calibration on a synthetic problem, using
priism's real pure-Python NUFFT solver (pymfista_nufft, no compiled .so
needed) throughout.

Verified manually against real HD142527 data (2026-08-21, SPW0): 3 rounds
of alternation reduced the imaging mean-squared-error from ~1559 (single
MFISTA solve, no self-cal) to ~387, with gains staying physically
plausible (clustered near 1+0j, ~3% amplitude spread) -- see the project
memory for the full numbers. This test checks the same qualitative
behavior (improves over a single-pass baseline, converges within a
reasonable number of rounds) on a small synthetic problem so it can run
without CASA/real MS access.
"""
import numpy as np

from priism.core import datacontainer, paramcontainer, imager
from priism.core.sparseimagingnufft import SparseImagingInputsNUFFT
from priism.core import pysparseimaging

from priism_selfcal.gains import Gains
from priism_selfcal import totalimaging
from priism_selfcal.selfcal import update_visibility


def _make_problem(nx=32, ny=32, n_stations=10, n_times=8, sigma_val=0.03, seed=0):
    rng = np.random.default_rng(seed)

    truth = np.zeros((nx, ny))
    truth[nx // 2, ny // 2] = 1.0
    truth[nx // 2 + 4, ny // 2 - 3] = 0.6

    time_vals = np.arange(n_times, dtype=np.float64) * 60.0
    st1_list, st2_list, time_list = [], [], []
    for t in time_vals:
        for i in range(n_stations):
            for j in range(i + 1, n_stations):
                st1_list.append(i)
                st2_list.append(j)
                time_list.append(t)
    st1 = np.array(st1_list)
    st2 = np.array(st2_list)
    time_arr = np.array(time_list)
    M = st1.size

    gains_meta = Gains(st1, st2, time_arr)
    true_gain = (1.0 + 0.08 * rng.standard_normal(gains_meta.gain_num)) * \
        np.exp(1j * 0.2 * rng.standard_normal(gains_meta.gain_num))

    u_pix = rng.uniform(2, nx - 2, M)
    v_pix = rng.uniform(2, ny - 2, M)

    class _ImageParam:
        imsize = (nx, ny)

    u_rad, v_rad = SparseImagingInputsNUFFT.convert_uv(_ImageParam, u_pix, v_pix)
    true_model_vis = pysparseimaging.x2y_nufft(u_rad, v_rad, truth, nthreads=1)

    g1 = true_gain[gains_meta.vid2gid_st[:, 0]]
    g2 = true_gain[gains_meta.vid2gid_st[:, 1]]
    noise = sigma_val * (rng.standard_normal(M) + 1j * rng.standard_normal(M))
    vis_org = true_model_vis / (g1 * np.conj(g2)) + noise  # see test_selfcal.py's convention note
    sigma = np.full(M, sigma_val)

    return dict(
        nx=nx, ny=ny, truth=truth, u_pix=u_pix, v_pix=v_pix,
        vis_org=vis_org, sigma=sigma, true_gain=true_gain,
        st1=st1, st2=st2, time_arr=time_arr, true_model_vis=true_model_vis,
    )


def _make_imager(p):
    ws = datacontainer.VisibilityWorkingSet(
        data_id=0, u=p["u_pix"], v=p["v_pix"],
        rdata=p["vis_org"].real.copy(), idata=p["vis_org"].imag.copy(),
        weight=1.0 / p["sigma"] ** 2,
    )
    im = imager.SparseModelingImager(solver='pymfista_nufft')
    im.working_set = ws
    im.imparam = paramcontainer.SimpleImageParamContainer(imsize=[p["nx"], p["ny"]])
    return im


def test_totalimaging_improves_on_single_pass_and_converges():
    p = _make_problem()

    # baseline: a single MFISTA solve on the raw (uncorrected) visibility,
    # no self-calibration at all
    im_baseline = _make_imager(p)
    im_baseline.solve(l1=1e-2, ltsv=1e-2, maxiter=300, eps=1e-4,
                       nonnegative=True, scalehyperparam=False)
    x_baseline = np.squeeze(im_baseline.imagearray.data)
    u_rad, v_rad = SparseImagingInputsNUFFT.convert_uv(
        im_baseline.imparam, p["u_pix"], p["v_pix"]
    )
    _, _, _, _, baseline_cost = pysparseimaging.calc_costs_nufft(
        M=p["vis_org"].size, Nx=p["nx"], Ny=p["ny"], u_dx=u_rad, v_dy=v_rad,
        vis_r=p["vis_org"].real, vis_i=p["vis_org"].imag, vis_std=p["sigma"],
        lambda_l1=1e-2, lambda_tv=0.0, lambda_tsv=1e-2, nonneg=True,
        xvec=x_baseline, nthreads=1,
    )

    # TotalImaging: alternating imaging <-> self-calibration.
    #
    # The round-to-round relative cost change decreases roughly
    # geometrically (empirically ~0.7x per round for this problem, not a
    # bug -- traced step by step on 2026-08-21), so total_eps=1e-6 (the
    # old reference's own default) is not practically reachable within a
    # handful of rounds. The old reference scripts themselves only ever
    # ran a small fixed selfcal_num (3-10) rather than iterating to that
    # threshold. total_eps=1e-2 here is loose enough to actually trigger
    # the "converged" branch within a modest round budget, while still
    # requiring real (not immediate) convergence.
    im = _make_imager(p)
    gains = Gains(p["st1"], p["st2"], p["time_arr"])
    result = totalimaging.run(
        imager=im, vis_org=p["vis_org"], sigma=p["sigma"], gains=gains,
        l1=1e-2, ltsv=1e-2, mu_sq=1e-2, mu_abs=1e-2,
        selfcal_num=20, imaging_maxiter=300, imaging_eps=1e-4,
        selfcal_maxiter=2000, selfcal_eps=1e-6, total_eps=1e-2,
        nonnegative=True, scalehyperparam=False,
    )

    assert result.converged, f"did not converge within {result.num_selfcal_iterations} rounds"
    assert result.num_selfcal_iterations > 1, "converged suspiciously fast -- check total_eps isn't trivially satisfied"
    assert result.cost < baseline_cost, \
        f"TotalImaging cost {result.cost} not better than single-pass baseline {baseline_cost}"

    # gains should end up physically plausible (clustered near 1+0j)
    amp = np.abs(gains.gain)
    assert amp.max() < 1.5 and amp.min() > 0.5

    rel_err = np.linalg.norm(np.abs(gains.gain) - np.abs(p["true_gain"])) / np.linalg.norm(np.abs(p["true_gain"]))
    assert rel_err < 0.1, f"gain amplitude recovery too poor: {rel_err}"


def test_totalimaging_working_set_reflects_final_gain_correction():
    p = _make_problem(n_stations=6, n_times=4, seed=1)
    im = _make_imager(p)
    gains = Gains(p["st1"], p["st2"], p["time_arr"])

    totalimaging.run(
        imager=im, vis_org=p["vis_org"], sigma=p["sigma"], gains=gains,
        l1=1e-2, ltsv=1e-2, mu_sq=1e-2, mu_abs=1e-2,
        selfcal_num=5, imaging_maxiter=200, imaging_eps=1e-4,
        selfcal_maxiter=2000, selfcal_eps=1e-6, total_eps=1e-6,
        nonnegative=True, scalehyperparam=False,
    )

    expected_vis_cal = update_visibility(gains, p["vis_org"])
    assert np.allclose(im.working_set.rdata, expected_vis_cal.real)
    assert np.allclose(im.working_set.idata, expected_vis_cal.imag)

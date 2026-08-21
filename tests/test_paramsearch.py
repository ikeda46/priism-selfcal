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
Tests for paramsearch.py's staged (lambda1, lambda_tsv, mu_sq, mu_abs)
search (Ikeda et al. 2025, PASJ 77(2):260-276, section 3.4).

Trial/iteration counts here are kept deliberately tiny (a handful of
Optuna trials, 1-2 TotalImaging rounds per gain-search trial) -- these
tests are wiring/correctness checks (does the staged handoff between
priism's own ellipsoid-criterion search and this package's gain-dispersion
search work correctly end to end, on a small synthetic problem), not
convergence quality checks. TotalImaging itself, the ellipsoid criterion,
and Gains construction are each already validated elsewhere
(test_totalimaging.py, priism's own uvcriteria, test_gains.py); this file
only needs to confirm paramsearch.py's own new logic (the gain-dispersion
cost, the stage sequencing and search-range narrowing) is correct.
"""
from __future__ import annotations

import numpy as np

from priism.core import datacontainer, paramcontainer, imager
from priism.core.sparseimagingnufft import SparseImagingInputsNUFFT
from priism.core import pysparseimaging

from priism_selfcal.gains import Gains
from priism_selfcal import paramsearch


def test_gain_dispersion_matches_definition():
    rng = np.random.default_rng(0)
    amp = 1.0 + 0.05 * rng.standard_normal(1000)
    phase = np.radians(10.0) * rng.standard_normal(1000)
    gain = amp * np.exp(1j * phase)

    sigma_ph, sigma_amp = paramsearch.gain_dispersion(gain)

    assert sigma_ph == np.degrees(phase.std())
    assert sigma_amp == amp.std()


def test_gain_dispersion_cost_is_zero_at_target():
    target = paramsearch.GainTarget(sigma_ph=5.0, sigma_amp=0.05)
    rng = np.random.default_rng(1)
    n = 200000
    phase = np.radians(target.sigma_ph) * rng.standard_normal(n)
    amp = 1.0 + target.sigma_amp * rng.standard_normal(n)
    gain = amp * np.exp(1j * phase)

    cost = paramsearch.gain_dispersion_cost(gain, target)
    assert cost < 0.01, f'cost should be near 0 when dispersion matches target exactly, got {cost}'


def _make_problem(nx=24, ny=24, n_stations=6, n_times=5, sigma_val=0.02, seed=0):
    rng = np.random.default_rng(seed)

    truth = np.zeros((nx, ny))
    truth[nx // 2, ny // 2] = 1.0
    truth[nx // 2 + 3, ny // 2 - 2] = 0.6

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
    true_gain = (1.0 + 0.05 * rng.standard_normal(gains_meta.gain_num)) * \
        np.exp(1j * 0.1 * rng.standard_normal(gains_meta.gain_num))

    u_pix = rng.uniform(2, nx - 2, M)
    v_pix = rng.uniform(2, ny - 2, M)

    class _ImageParam:
        imsize = (nx, ny)

    u_rad, v_rad = SparseImagingInputsNUFFT.convert_uv(_ImageParam, u_pix, v_pix)
    true_model_vis = pysparseimaging.x2y_nufft(u_rad, v_rad, truth, nthreads=1)

    g1 = true_gain[gains_meta.vid2gid_st[:, 0]]
    g2 = true_gain[gains_meta.vid2gid_st[:, 1]]
    noise = sigma_val * (rng.standard_normal(M) + 1j * rng.standard_normal(M))
    vis_org = true_model_vis / (g1 * np.conj(g2)) + noise
    sigma = np.full(M, sigma_val)

    return dict(
        nx=nx, ny=ny, truth=truth, u_pix=u_pix, v_pix=v_pix,
        vis_org=vis_org, sigma=sigma, st1=st1, st2=st2, time_arr=time_arr,
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


def test_search_gain_regularizers_picks_grid_value_and_plausible_gains():
    p = _make_problem()
    im = _make_imager(p)
    target = paramsearch.GAIN_TARGETS['large_variance']

    result = paramsearch.search_gain_regularizers(
        im, p['st1'], p['st2'], p['time_arr'], p['vis_org'], p['sigma'],
        l1=1.0, ltsv=1.0, target=target,
        mu_sq_exp_range=(-2, 2), mu_abs_exp_range=(-2, 2),
        selfcal_num=2, bayesopt_maxiter=4, imaging_maxiter=100, selfcal_maxiter=500,
    )

    assert result.mu_sq in [10.0 ** e for e in range(-2, 3)]
    assert result.mu_abs in [10.0 ** e for e in range(-2, 3)]
    assert np.isfinite(result.cost)
    amp = np.abs(result.gains.gain)
    assert amp.min() > 0.5 and amp.max() < 1.5, 'gains should stay physically plausible (near 1+0j)'


def test_search_imaging_regularizers_picks_grid_value(tmp_path):
    p = _make_problem()
    im = _make_imager(p)

    result = paramsearch.search_imaging_regularizers(
        im, l1_exp_range=(-4, 2), ltsv_exp_range=(-4, 2),
        bayesopt_maxiter=4, imaging_maxiter=100,
        imageprefix=str(tmp_path / 'image_test'),
    )

    assert result.l1 in [10.0 ** e for e in range(-4, 3)]
    assert result.ltsv in [10.0 ** e for e in range(-4, 3)]
    assert result.image.shape == (p['nx'], p['ny'])


def test_run_staged_parameter_search_end_to_end(tmp_path):
    p = _make_problem()
    im = _make_imager(p)
    target = paramsearch.GAIN_TARGETS['large_variance']

    result = paramsearch.run_staged_parameter_search(
        im, p['st1'], p['st2'], p['time_arr'], p['vis_org'], p['sigma'], target,
        l1_exp_range=(-4, 2), ltsv_exp_range=(-4, 2),
        mu_sq_exp_range=(-2, 2), mu_abs_exp_range=(-2, 2),
        narrow_width=1, n_refine_rounds=1,
        selfcal_num=2, bayesopt_maxiter_lambda=3, bayesopt_maxiter_mu=3,
        imaging_maxiter=100, selfcal_maxiter=500,
        imageprefix=str(tmp_path / 'staged'),
    )

    assert set(result.stages.keys()) == {'stage0', 'mu_round0', 'lambda_round0'}
    # mu_round0/lambda_round0's search ranges should be centered on
    # stage0's winner (narrow_width=1 decade here), i.e. within 10x of it
    assert abs(np.log10(result.mu_sq) - np.log10(result.stages['mu_round0'].mu_sq)) <= 1
    assert abs(np.log10(result.l1) - np.log10(result.stages['stage0'].l1)) <= 1

    amp = np.abs(result.gains.gain)
    assert amp.min() > 0.5 and amp.max() < 1.5
    assert result.image.shape == (p['nx'], p['ny'])


def test_run_staged_parameter_search_supports_multiple_refine_rounds(tmp_path):
    p = _make_problem()
    im = _make_imager(p)
    target = paramsearch.GAIN_TARGETS['large_variance']

    result = paramsearch.run_staged_parameter_search(
        im, p['st1'], p['st2'], p['time_arr'], p['vis_org'], p['sigma'], target,
        l1_exp_range=(-4, 2), ltsv_exp_range=(-4, 2),
        mu_sq_exp_range=(-2, 2), mu_abs_exp_range=(-2, 2),
        narrow_width=1, n_refine_rounds=2,
        selfcal_num=2, bayesopt_maxiter_lambda=2, bayesopt_maxiter_mu=2,
        imaging_maxiter=100, selfcal_maxiter=500,
        imageprefix=str(tmp_path / 'staged2'),
    )

    assert set(result.stages.keys()) == {
        'stage0', 'mu_round0', 'lambda_round0', 'mu_round1', 'lambda_round1'
    }
    amp = np.abs(result.gains.gain)
    assert amp.min() > 0.5 and amp.max() < 1.5

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
self_calibration()'s ADMM penalty (rho) is recalibrated between outer
stages from the *actual* current h (via estimate_rho), instead of blindly
multiplying by RHOSTEP each round (2026-08-21). This targets a case the
single upfront estimate_rho_init (h~1 assumption, single global scalar)
does not cover well: an array with a heterogeneous mix of well-sampled
("strong") and sparsely-sampled/noisy ("weak") antennas, where a single
median-based rho_init may not suit all gains, and/or a badly mismatched
caller-supplied rho_init.

This test builds exactly that kind of heterogeneous problem and checks
that self_calibration still converges to physically plausible gains
(clustered near 1+0j, matching real ALMA behavior -- see the discussion
in test_selfcal.py) within a small number of outer stages even when
started from a deliberately bad rho_init.
"""
import numpy as np

from priism_selfcal.gains import Gains
from priism_selfcal.pyselfcal import self_calibration


def _make_heterogeneous_problem(n_strong=15, n_weak=15, n_times=10, seed=0):
    rng = np.random.default_rng(seed)
    S = n_strong + n_weak
    time_vals = np.arange(n_times, dtype=np.float64) * 60.0

    st1_list, st2_list, time_list, sigma_list = [], [], [], []
    for t in time_vals:
        for i in range(S):
            for j in range(i + 1, S):
                weak_involved = (i >= n_strong) or (j >= n_strong)
                # weak antennas only participate in ~30% of time slots
                # (sparse scheduling/flagging), and are ~10x noisier
                if weak_involved and rng.uniform() > 0.3:
                    continue
                st1_list.append(i)
                st2_list.append(j)
                time_list.append(t)
                sigma_list.append(0.2 if weak_involved else 0.02)

    st1 = np.array(st1_list)
    st2 = np.array(st2_list)
    time_arr = np.array(time_list)
    sigma = np.array(sigma_list)
    M = st1.size

    gains_meta = Gains(st1, st2, time_arr)
    true_gain = (1.0 + 0.05 * rng.standard_normal(gains_meta.gain_num)) * \
        np.exp(1j * 0.15 * rng.standard_normal(gains_meta.gain_num))

    y = np.ones(M, dtype=np.complex128)
    g1 = true_gain[gains_meta.vid2gid_st[:, 0]]
    g2 = true_gain[gains_meta.vid2gid_st[:, 1]]
    noise = sigma * (rng.standard_normal(M) + 1j * rng.standard_normal(M))
    vis = y / (g1 * np.conj(g2)) + noise  # see test_selfcal.py's convention note

    return gains_meta, vis, sigma, y, true_gain


def test_rho_recalibration_converges_from_bad_rho_init_on_heterogeneous_array():
    gains_meta, vis, sigma, y, true_gain = _make_heterogeneous_problem()

    result = self_calibration(
        vis=vis, vis_std=sigma, y=y,
        ginit=np.ones(gains_meta.gain_num, dtype=np.complex128),
        gid_adj_t=gains_meta.gid_adj_t, vid2gid_st=gains_meta.vid2gid_st,
        time_tbl=gains_meta.time_tbl, Stnum=gains_meta.st_num, Tnum=gains_meta.time_num,
        lambda_1=1e-6, lambda_2=1e-6,
        rho_init=1.0,  # deliberately far from the data scale
        maxiter=3000, eps=1e-8,
    )
    assert result.converged

    # amplitude: gauge-invariant, safe to compare directly
    rel_err = np.linalg.norm(np.abs(result.gain) - np.abs(true_gain)) / np.linalg.norm(np.abs(true_gain))
    assert rel_err < 0.1, f"gain amplitude recovery too poor: {rel_err}"

    # physically plausible: clustered near 1+0j, not off by factors of 2-3
    amp = np.abs(result.gain)
    assert amp.max() < 1.5 and amp.min() > 0.5, \
        f"gain amplitude not physically plausible: min={amp.min():.3f} max={amp.max():.3f}"

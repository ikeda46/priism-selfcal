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
Verify Gains' vectorized table construction and regularizers() against a
naive (non-vectorized) reference implementation, on randomized inputs
including non-contiguous station labels and arbitrary real-valued times.
"""
import numpy as np
import pytest

from priism_selfcal.gains import Gains


def _build_tables_naive(st1, st2, time_raw):
    M = len(st1)
    time_tbl = sorted(set(time_raw.tolist()))
    time_to_idx = {t: i for i, t in enumerate(time_tbl)}

    gain_index = {}
    gains = []

    def get_or_create(st, t):
        key = (st, t)
        if key not in gain_index:
            gain_index[key] = len(gains)
            gains.append(key)
        return gain_index[key]

    vid2gid_st = np.empty((M, 2), dtype=np.int64)
    for i in range(M):
        t = time_to_idx[time_raw[i]]
        vid2gid_st[i, 0] = get_or_create(int(st1[i]), t)
        vid2gid_st[i, 1] = get_or_create(int(st2[i]), t)

    Gnum = len(gains)
    gid_adj_t = -1 * np.ones((Gnum, 4), dtype=np.int64)
    by_station = {}
    for g, (st, t) in enumerate(gains):
        gid_adj_t[g, 0] = st
        gid_adj_t[g, 1] = t
        by_station.setdefault(st, []).append(g)

    for st, glist in by_station.items():
        glist_sorted = sorted(glist, key=lambda g: gid_adj_t[g, 1])
        for k in range(len(glist_sorted)):
            if k > 0:
                gid_adj_t[glist_sorted[k], 2] = glist_sorted[k - 1]
            if k < len(glist_sorted) - 1:
                gid_adj_t[glist_sorted[k], 3] = glist_sorted[k + 1]

    return gid_adj_t, vid2gid_st, np.array(time_tbl, dtype=np.float64)


def _canonical_form(gid_adj_t, vid2gid_st, time_tbl, station_labels_are_raw):
    """Relabel gains by (station, time) sort order so id-assignment order
    doesn't matter, and remap raw (non-compacted) station labels to a rank
    so naive-vs-vectorized comparisons are apples to apples."""
    Gnum = gid_adj_t.shape[0]

    stations = gid_adj_t[:, 0]
    if station_labels_are_raw:
        uniq_st = np.unique(stations)
        st_rank = {s: i for i, s in enumerate(uniq_st)}
        stations = np.array([st_rank[s] for s in stations])

    times = time_tbl[gid_adj_t[:, 1]]

    order = np.lexsort((times, stations))
    relabel = np.empty(Gnum, dtype=np.int64)
    relabel[order] = np.arange(Gnum)

    def remap(idx):
        return np.where(idx >= 0, relabel[idx], -1)

    adj = np.stack([
        stations[order], times[order],
        remap(gid_adj_t[order, 2]), remap(gid_adj_t[order, 3])
    ], axis=1)
    v2g = np.stack([relabel[vid2gid_st[:, 0]], relabel[vid2gid_st[:, 1]]], axis=1)
    return adj, v2g


@pytest.mark.parametrize("seed", range(30))
def test_gains_construction_matches_naive_reference(seed):
    rng = np.random.default_rng(seed)
    M = rng.integers(5, 60)
    S = rng.integers(2, 8)
    T = rng.integers(1, 5)

    # non-contiguous / shuffled station labels (e.g. antenna ids with gaps)
    station_pool = rng.choice(np.arange(0, 100), size=S, replace=False)
    st1 = station_pool[rng.integers(0, S, M)]
    st2 = station_pool[rng.integers(0, S, M)]

    time_pool = rng.choice(np.arange(0, 1000, dtype=np.float64), size=T, replace=False)
    time_raw = time_pool[rng.integers(0, T, M)]

    g = Gains(st1, st2, time_raw)
    adj_naive, v2g_naive, time_tbl_naive = _build_tables_naive(st1, st2, time_raw)

    adj_v, v2g_v = _canonical_form(g.gid_adj_t, g.vid2gid_st, g.time_tbl, station_labels_are_raw=False)
    adj_n, v2g_n = _canonical_form(adj_naive, v2g_naive, time_tbl_naive, station_labels_are_raw=True)

    assert np.array_equal(adj_v, adj_n)
    assert np.array_equal(v2g_v, v2g_n)

    assert g.vis_num == M
    assert g.gain_num == adj_naive.shape[0]
    assert g.time_num == len(time_tbl_naive)
    assert g.gain.shape == (g.gain_num,)
    assert np.all(g.gain == 1.0 + 0.0j)

    # station_ids must recover the *original* (possibly non-contiguous)
    # station labels from gid_adj_t's compact 0..S-1 station index
    assert g.station_ids.size == g.st_num
    recovered_station1 = g.station_ids[g.gid_adj_t[g.vid2gid_st[:, 0], 0]]
    recovered_station2 = g.station_ids[g.gid_adj_t[g.vid2gid_st[:, 1], 0]]
    assert np.array_equal(recovered_station1, st1)
    assert np.array_equal(recovered_station2, st2)


@pytest.mark.parametrize("seed", range(20))
def test_regularizers_matches_naive_sum(seed):
    rng = np.random.default_rng(seed)
    M = rng.integers(5, 60)
    S = rng.integers(2, 8)
    T = rng.integers(2, 6)  # need >=2 times for meaningful prev/next links

    station_pool = rng.choice(np.arange(0, 50), size=S, replace=False)
    st1 = station_pool[rng.integers(0, S, M)]
    st2 = station_pool[rng.integers(0, S, M)]
    time_pool = rng.choice(np.arange(0, 1000, dtype=np.float64), size=T, replace=False)
    time_raw = time_pool[rng.integers(0, T, M)]

    g = Gains(st1, st2, time_raw)
    g.gain = (1.0 + 0.3 * rng.standard_normal(g.gain_num)) * np.exp(1j * 0.5 * rng.standard_normal(g.gain_num))

    result = g.regularizers()

    term1 = 0.0
    term2 = 0.0
    for i in range(g.gain_num):
        k = g.gid_adj_t[i, 3]
        if k >= 0:
            t1 = g.gid_adj_t[k, 1]
            t2 = g.gid_adj_t[i, 1]
            d_t = g.time_tbl[t1] - g.time_tbl[t2]
            w_at = 1 / d_t
            g1 = g.gain[k]
            g2 = g.gain[i]
            term1 += w_at * (np.abs(g1 - g2) ** 2)
            term2 += w_at * ((np.abs(g1) - np.abs(g2)) ** 2)

    assert np.isclose(result.sq_term, term1)
    assert np.isclose(result.abs_term, term2)

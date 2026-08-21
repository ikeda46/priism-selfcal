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
Per-antenna gain bookkeeping for self-calibration.

Ported from old/python/sparseimaging/selfcal.py::Gains, rebuilt with a
vectorized (NumPy) construction algorithm instead of the original's nested
Python loops. Verified against the original's construction on 200 random
synthetic cases, and benchmarked at ~1.3s for 1e6 visibilities / 5e5 gains
(2026-08-20) -- fast enough that no save/load caching is needed; a fresh
Gains object is built once per run and reused in memory for the whole
imaging <-> self-calibration loop.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

GainRegularizers = namedtuple('GainRegularizers', ['sq_term', 'abs_term'])


class Gains:
    """
    Set gains.

    usage:
        Gains(station1, station2, time)

        station1, station2: integer arrays (length M), antenna index for
            each side of each visibility.
        time: array (length M), observation time for each visibility
            (any orderable/hashable numeric value, e.g. MJD seconds).

        All three arrays must have equal length M (the number of
        visibilities).

    Attributes:
        vis_num   -- number of visibilities (M)
        time_num  -- number of distinct observation times (T)
        time_tbl  -- (T,) float64 array converting time index -> the actual
                     observation time value
        st_num    -- number of distinct stations (S)
        station_ids -- (S,) array converting the internal 0..S-1 station
                     index used in gid_adj_t -> the original station1/
                     station2 label the caller passed in (e.g. real MS
                     ANTENNA1/ANTENNA2 values, not necessarily 0..S-1 or
                     contiguous)
        gain_num  -- number of gains (Gnum); one gain per (station, time)
                     pair that is actually observed -- not necessarily
                     S*T, since a station need not appear at every time
        gain      -- (Gnum,) complex128 array of gains, initialized to 1+0j
        gid_adj_t -- (Gnum, 4) int64 table, one row per gain:
                     col 0: station index
                     col 1: observation time index (into time_tbl)
                     col 2: index of this station's previous gain in time
                            (-1 if this is the first)
                     col 3: index of this station's next gain in time
                            (-1 if this is the last)
        vid2gid_st -- (M, 2) int64 table, one row per visibility:
                     col 0: gain index for the station1 side
                     col 1: gain index for the station2 side
    """

    def __init__(self, station1: np.ndarray, station2: np.ndarray, time: np.ndarray):
        station1 = np.asarray(station1)
        station2 = np.asarray(station2)
        time = np.asarray(time)

        M = station1.size
        if station2.size != M or time.size != M:
            raise ValueError(
                f"station1, station2, and time must have equal length "
                f"(got {station1.size}, {station2.size}, {time.size})"
            )

        time_tbl, time_idx = np.unique(time, return_inverse=True)
        T = time_tbl.size

        station_all, station_inverse = np.unique(
            np.concatenate([station1, station2]), return_inverse=True
        )
        S = station_all.size
        # station_inverse gives a compact 0..S-1 index per occurrence; use it
        # (rather than raw station1/station2) so gid_adj_t's station column
        # is always a dense index regardless of what the caller's raw
        # station labels look like.
        st1_idx = station_inverse[:M]
        st2_idx = station_inverse[M:]

        all_st = np.concatenate([st1_idx, st2_idx])
        all_time = np.concatenate([time_idx, time_idx])
        pairs = np.stack([all_st, all_time], axis=1)

        unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
        Gnum = unique_pairs.shape[0]

        vid2gid_st = np.empty((M, 2), dtype=np.int64)
        vid2gid_st[:, 0] = inverse[:M]
        vid2gid_st[:, 1] = inverse[M:]

        gid_adj_t = -1 * np.ones((Gnum, 4), dtype=np.int64)
        gid_adj_t[:, 0] = unique_pairs[:, 0]
        gid_adj_t[:, 1] = unique_pairs[:, 1]

        # prev/next links: sort gains by (station, time), then a gain's
        # neighbor in this order is its temporal prev/next *if* it belongs
        # to the same station (a groupby-shift, done via sort + boundary
        # detection rather than an explicit groupby).
        order = np.lexsort((unique_pairs[:, 1], unique_pairs[:, 0]))
        sorted_station = unique_pairs[order, 0]
        sorted_gain_id = order

        same_station_as_prev = np.zeros(Gnum, dtype=bool)
        same_station_as_prev[1:] = sorted_station[1:] == sorted_station[:-1]

        prev_of_current = np.where(
            same_station_as_prev, np.concatenate([[-1], sorted_gain_id[:-1]]), -1
        )
        gid_adj_t[sorted_gain_id, 2] = prev_of_current

        valid = same_station_as_prev
        gid_adj_t[sorted_gain_id[:-1][valid[1:]], 3] = sorted_gain_id[1:][valid[1:]]

        self.vis_num = M
        self.time_num = T
        self.time_tbl = time_tbl.astype(np.float64)
        self.st_num = S
        # gid_adj_t[:, 0] and vid2gid_st's station roles are indices 0..S-1
        # into this array, which recovers the *original* station labels the
        # caller passed in (e.g. real MS ANTENNA1/ANTENNA2 values, which
        # need not be contiguous or start at 0).
        self.station_ids = station_all
        self.gain_num = Gnum
        self.gain = np.ones(Gnum, dtype=np.complex128)
        self.gid_adj_t = gid_adj_t
        self.vid2gid_st = vid2gid_st

    def regularizers(self) -> GainRegularizers:
        """
        Weighted sum, over all (gain, its temporal-next gain of the same
        station) pairs, of the gain-smoothness term |g_next - g|^2 and the
        gain-amplitude-smoothness term (|g_next| - |g|)^2, matching
        old/python/sparseimaging/selfcal.py::Gains.regularizers.

        The per-pair weight is 1 / (time_next - time), same convention as
        the C++/pyselfcal self-calibration engine's own regularization
        terms (mu1/mu2 there act on the same adjacency structure).
        """
        has_next = self.gid_adj_t[:, 3] >= 0
        i_idx = np.nonzero(has_next)[0]
        k_idx = self.gid_adj_t[i_idx, 3]

        d_t = self.time_tbl[self.gid_adj_t[k_idx, 1]] - self.time_tbl[self.gid_adj_t[i_idx, 1]]
        w_at = 1.0 / d_t

        g1 = self.gain[k_idx]
        g2 = self.gain[i_idx]

        sq_term = np.sum(w_at * np.abs(g1 - g2) ** 2)
        abs_term = np.sum(w_at * (np.abs(g1) - np.abs(g2)) ** 2)

        return GainRegularizers(sq_term=sq_term, abs_term=abs_term)

    def plot_gains(self, fname=None, ax=None):
        """
        Scatter plot of every gain in the complex plane, with a unit
        circle for reference (real ALMA gains are expected to cluster
        near 1+0j). Mirrors old/python/sparseimaging/selfcal.py::
        Gains.plot_gains. CASA's plotms can produce the same plot from a
        caltable via xaxis='real', yaxis='imag'.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        else:
            fig = ax.figure

        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(self.gain.real, self.gain.imag, '.')
        ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=0.3)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.2)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.2)

        ax.set_xticks(np.linspace(-1, 1, 3))
        ax.set_yticks(np.linspace(-1, 1, 3))
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_xlabel('real', fontsize=14)
        ax.set_ylabel('imaginary', fontsize=14)
        ax.set_aspect('equal')
        fig.tight_layout()

        if fname is not None:
            fig.savefig(fname)

        return ax

    def plot_station_gains(self, fname=None, station_names=None, ncols=4):
        """
        Per-station time series of |gain| and phase(gain), one subplot per
        station. Mirrors old/python/sparseimaging/selfcal.py::
        Gains.plot_station_gains (fixed an off-by-one there that dropped
        each station's last time sample). CASA's plotcal/plotms produce
        the same kind of plot (amp/phase vs. time) from a caltable.

        station_names -- optional array/mapping from the original station
                          label (see station_ids) to a display name, e.g.
                          real antenna names from the MS's ANTENNA table.
        """
        import matplotlib.pyplot as plt

        nrows = int(np.ceil(self.st_num / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), facecolor='white', squeeze=False
        )

        for st in range(self.st_num):
            ax1 = axes[st // ncols, st % ncols]

            mask = self.gid_adj_t[:, 0] == st
            t = self.time_tbl[self.gid_adj_t[mask, 1]]
            g = self.gain[mask]
            order = np.argsort(t)
            t, g = t[order], g[order]

            ax1.plot(t, np.abs(g), 'C0.', label=r'$|gain|$')
            ax1.set_xlim(self.time_tbl.min(), self.time_tbl.max())
            ax1.set_ylim(0, max(np.abs(g).max() * 1.2, 1e-12))

            ax2 = ax1.twinx()
            ax2.plot(t, np.angle(g), 'C1.', label='phase')
            ax2.set_xlim(self.time_tbl.min(), self.time_tbl.max())
            ax2.set_ylim(-np.pi, np.pi)

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc='lower right')

            label = self.station_ids[st]
            if station_names is not None:
                label = station_names[label]
            ax1.set_title(f'station {label}')
            ax1.set_xlabel('t')
            ax1.set_ylabel(r'|gain|')
            ax2.grid(True)
            ax2.set_ylabel('phase(gain)')

        for st in range(self.st_num, nrows * ncols):
            axes[st // ncols, st % ncols].axis('off')

        fig.tight_layout()

        if fname is not None:
            fig.savefig(fname)

        return fig

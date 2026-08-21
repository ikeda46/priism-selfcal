from __future__ import annotations

import logging
from collections import namedtuple

import numpy as np

logger = logging.getLogger(__name__)

# ADMM penalty continuation schedule, matching c++/selfcal.hpp.
MAXOUTER = 30
MINITER = 100
RHOSTEP = 100
GH_DIFF_TH = 1.0e-10

# tuple holding self-calibration results
PySelfCalResults = namedtuple(
    'PySelfCalResults',
    ['gain', 'finalcost', 'converged']
)


def adjust_g_st(
        gid_adj_t: np.ndarray,
        Stnum: int,
        ainv: np.ndarray,
        z: np.ndarray
) -> tuple[np.ndarray, int]:
    """Per-station projection so that, station by station, the surviving
    (non-negative) entries of z sum to the number of gains at that station.

    Mirrors c++/selfcal.cpp::adjust_g_st. `gid_adj_t` is the (Gnum, 4)
    table [station_idx, time_idx, prev_gid, next_gid]; only column 0 is
    used here.
    """
    z = z.copy()
    st = gid_adj_t[:, 0]

    z_tmp = np.bincount(st, weights=z, minlength=Stnum)
    ainv_tmp = np.bincount(st, weights=ainv, minlength=Stnum)
    gst_num = np.bincount(st, minlength=Stnum).astype(np.float64)

    eta = (z_tmp - gst_num) / ainv_tmp

    z = z - eta[st] * ainv

    mask = z >= 0
    zsum = np.bincount(st, weights=np.where(mask, z, 0.0), minlength=Stnum)

    z = z * mask
    z = z * (gst_num[st] / zsum[st])

    return z, int(round(mask.sum()))


def adjust_g(ainv: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, int]:
    """Global (non-per-station) simplex-like projection, kept for parity
    with c++/selfcal.cpp::adjust_g. Not used by self_calibration() (which
    always calls adjust_g_st), but ported for completeness.
    """
    Gnum = z.size
    z = z.copy()
    mask = np.ones(Gnum, dtype=bool)

    eta = (z.sum() - Gnum) / ainv.sum()
    zmin = z.min()

    while zmin < eta:
        mask &= (z >= eta)
        z = z * mask

        active = z[mask]
        zmin = active.max() if active.size > 0 else z.max()
        # smallest strictly-positive entry, matching the C++ loop
        positive = z[(z < zmin) & (z > 0.0)]
        if positive.size > 0:
            zmin = positive.min()

        eta = (z.sum() - Gnum) / ainv.sum()

    z = z - eta * ainv
    z = z * mask

    return z, int(round(mask.sum()))


def _time_diff_and_station(gid_adj_t: np.ndarray, time_tbl: np.ndarray):
    """For every gain index i with a valid "next" link k = gid_adj_t[i, 3],
    return (i_idx, k_idx, st, d_t) arrays over the valid links only.
    """
    k = gid_adj_t[:, 3]
    has_next = k >= 0
    i_idx = np.nonzero(has_next)[0]
    k_idx = k[has_next]
    st = gid_adj_t[i_idx, 0]
    d_t = time_tbl[gid_adj_t[k_idx, 1]] - time_tbl[gid_adj_t[i_idx, 1]]
    return i_idx, k_idx, st, d_t


def cost_g(
        wvis: np.ndarray,
        wy: np.ndarray,
        sig: np.ndarray,
        g: np.ndarray,
        gid_adj_t: np.ndarray,
        vid2gid_st: np.ndarray,
        time_tbl: np.ndarray,
        lambda_1: float,
        lambda_2: float,
        prt: bool = False
) -> float:
    gid1 = vid2gid_st[:, 0]
    gid2 = vid2gid_st[:, 1]

    term0 = np.sum(np.abs(wvis * g[gid1] * np.conj(g[gid2]) - wy) ** 2) / 2.0

    i_idx, k_idx, st, d_t = _time_diff_and_station(gid_adj_t, time_tbl)
    w_at = 1.0 / (sig[st] * d_t)

    term1 = np.sum(w_at * np.abs(g[k_idx] - g[i_idx]) ** 2)
    term2 = np.sum(w_at * (np.abs(g[k_idx]) - np.abs(g[i_idx])) ** 2)

    cost = term0 + lambda_1 * term1 + lambda_2 * term2

    if prt:
        logger.debug("weighted chi-sq %s", 2 * term0)
        logger.debug("lambda_1: %s  (gt-gt-1)^2 %s", lambda_1, term1)
        logger.debug("lambda_2: %s  (|gt|-|gt-1|)^2 %s", lambda_2, term2)
        logger.debug("total cost %s", cost)

    return cost


def cost_gh(
        wvis: np.ndarray,
        wy: np.ndarray,
        sig: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        gid_adj_t: np.ndarray,
        vid2gid_st: np.ndarray,
        time_tbl: np.ndarray,
        lambda_1: float,
        lambda_2: float,
        rho: float,
        prt: bool = False
) -> float:
    gid1 = vid2gid_st[:, 0]
    gid2 = vid2gid_st[:, 1]

    tmp0 = (wvis * g[gid1] * np.conj(h[gid2]) + wvis * h[gid1] * np.conj(g[gid2])) / 2.0 - wy
    term0 = np.sum(np.abs(tmp0) ** 2)

    i_idx, k_idx, st, d_t = _time_diff_and_station(gid_adj_t, time_tbl)
    w_at = 1.0 / (sig[st] * d_t)

    term1 = np.sum(w_at * (np.abs(h[k_idx] - g[i_idx]) ** 2 + np.abs(g[k_idx] - h[i_idx]) ** 2))
    term2 = np.sum(
        w_at * (
            (np.abs(h[k_idx]) - np.abs(g[i_idx])) ** 2
            + (np.abs(g[k_idx]) - np.abs(h[i_idx])) ** 2
        )
    )

    term3 = np.sum(np.abs(g - h) ** 2)

    cost = (term0 + lambda_1 * term1 + lambda_2 * term2 + rho * term3) / 2.0

    if prt:
        logger.debug("(g-h)^2 %s", term3)

    return cost


def update_g(
        pow_wvis: np.ndarray,
        wyvis_conj: np.ndarray,
        sig: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        gid_adj_t: np.ndarray,
        vid2gid_st: np.ndarray,
        time_tbl: np.ndarray,
        Gnum: int,
        Stnum: int,
        lambda_1: float,
        lambda_2: float,
        rho: float
) -> np.ndarray:
    """One ADMM sub-step: solve for the new value of `g` given the current
    `h`, closed-form per-gain (mirrors c++/selfcal.cpp::update_g).
    """
    gid1 = vid2gid_st[:, 0]
    gid2 = vid2gid_st[:, 1]

    h_pow = np.abs(h) ** 2
    h_abs = np.abs(h)

    a_vec = np.zeros(Gnum, dtype=np.float64)
    b_vec = np.zeros(Gnum, dtype=np.complex128)
    c_vec = np.zeros(Gnum, dtype=np.float64)

    np.add.at(a_vec, gid1, pow_wvis * h_pow[gid2])
    np.add.at(b_vec, gid1, wyvis_conj * h[gid2])
    np.add.at(a_vec, gid2, pow_wvis * h_pow[gid1])
    np.add.at(b_vec, gid2, np.conj(wyvis_conj) * h[gid1])

    a_vec /= 2.0
    a_vec += rho
    b_vec = b_vec / 2.0 + rho * h

    i_idx, k_idx, st, d_t = _time_diff_and_station(gid_adj_t, time_tbl)
    w_at = 1.0 / (sig[st] * d_t)

    np.add.at(a_vec, i_idx, (lambda_1 + lambda_2) * w_at)
    np.add.at(b_vec, i_idx, lambda_1 * h[k_idx] * w_at)
    np.add.at(c_vec, i_idx, lambda_2 * h_abs[k_idx] * w_at)

    np.add.at(a_vec, k_idx, (lambda_1 + lambda_2) * w_at)
    np.add.at(b_vec, k_idx, lambda_1 * h[i_idx] * w_at)
    np.add.at(c_vec, k_idx, lambda_2 * h_abs[i_idx] * w_at)

    ainv_vec = 1.0 / a_vec
    b_abs = np.abs(b_vec)

    z_vec = (c_vec + b_abs) * ainv_vec

    z_vec, active = adjust_g_st(gid_adj_t, Stnum, ainv_vec, z_vec)

    g_zeros = Gnum - active
    if g_zeros > 0:
        logger.debug("%d components were set to 0", g_zeros)

    gnew = np.zeros(Gnum, dtype=np.complex128)
    nonzero = b_abs > 0
    gnew[nonzero] = z_vec[nonzero] * b_vec[nonzero] / b_abs[nonzero]

    return gnew


def estimate_rho(wvis: np.ndarray, vid2gid_st: np.ndarray, h: np.ndarray, scale: float = 1.0) -> float:
    """Estimate a rho comparable in scale to the data term's own per-gain
    curvature at the *current* h -- i.e. the data-only contribution to
    update_g's a_vec (a_vec(gid1) += pow_wvis(i)*h_pow(gid2)), evaluated
    exactly as update_g would, rather than leaving rho at an arbitrary
    fixed value that may be many orders of magnitude away from the data
    scale (e.g. at high SNR / small vis_std, the data term can be ~1e5-1e6
    while a naive rho=1 needs ~3 outer x100 jumps just to become
    comparable -- see priism-selfcal's notes on the ADMM penalty
    continuation, 2026-08-20).

    Used both to pick rho_init (call with h=ginit) and, per outer stage
    with the *actual* current h instead of blindly multiplying by
    RHOSTEP, to recalibrate rho between stages (2026-08-21) -- the
    initial-h estimate assumes h_pow~1, which is only exactly right at
    the very first call; recomputing from the real h after each stage
    corrects for that and for any per-gain curvature the single upfront
    estimate didn't capture.

    `scale` lets the caller push the estimate up/down; 1.0 matches the
    data term's own typical magnitude.
    """
    Gnum = h.size
    pow_wvis = np.abs(wvis) ** 2
    h_pow = np.abs(h) ** 2
    gid1 = vid2gid_st[:, 0]
    gid2 = vid2gid_st[:, 1]

    a_data = np.zeros(Gnum, dtype=np.float64)
    np.add.at(a_data, gid1, pow_wvis * h_pow[gid2])
    np.add.at(a_data, gid2, pow_wvis * h_pow[gid1])
    a_data *= 0.5  # matches update_g's a_vec /= 2.0 scaling of the data term

    nonzero = a_data[a_data > 0]
    if nonzero.size == 0:
        return scale
    return float(scale * np.median(nonzero))


def estimate_rho_init(wvis: np.ndarray, vid2gid_st: np.ndarray, Gnum: int, scale: float = 1.0) -> float:
    """Convenience wrapper for estimate_rho() at h_pow=1 (e.g. ginit=1+0j,
    the usual starting point). See estimate_rho's docstring.
    """
    return estimate_rho(wvis, vid2gid_st, np.ones(Gnum, dtype=np.complex128), scale=scale)


def self_calibration(
        vis: np.ndarray,
        vis_std: np.ndarray,
        y: np.ndarray,
        ginit: np.ndarray,
        gid_adj_t: np.ndarray,
        vid2gid_st: np.ndarray,
        time_tbl: np.ndarray,
        Stnum: int,
        Tnum: int,
        lambda_1: float,
        lambda_2: float,
        rho_init: float | None,
        maxiter: int,
        eps: float
) -> PySelfCalResults:
    """Estimate per-gain complex gains by ADMM-style splitting, mirroring
    c++/selfcal.cpp::self_calibration.

    `lambda_1`/`lambda_2` here are the gain-smoothness/amplitude
    regularization weights (mu1/mu2 in Ikeda et al. 2025), not the
    imaging L1/TSV weights used elsewhere in this package.

    `rho_init`: if None, estimated from the data scale via
    estimate_rho_init (recommended default, see that function's docstring);
    pass an explicit float to pin it (matches the original C++ behavior,
    which always required a caller-supplied value). Between outer
    (rho-continuation) stages, instead of blindly multiplying by RHOSTEP,
    rho is recalibrated via estimate_rho() using the actual current h --
    but never allowed to grow slower than the RHOSTEP schedule would, so
    the original guarantee of eventual (g, h) consensus within MAXOUTER
    stages still holds (2026-08-21).

    `eps` is applied as a *relative* threshold on the inner loop's cost
    change (abs(cost - cost_new) < eps * max(1, abs(cost_new))), not an
    absolute one. At large absolute cost values (e.g. high-SNR data, where
    the weighted chi-square term can reach 1e5-1e6), an absolute eps of
    1e-10 or so is at or below double-precision's representable resolution
    at that magnitude, causing the inner loop to report "converged" almost
    immediately even though the solution is still far from optimal --
    confirmed reproducing bit-for-bit with the original C++ engine on
    identical inputs, so this was a property of the absolute-eps criterion
    itself, not a porting bug (see priism-selfcal tests/, 2026-08-20).
    """
    M = vis.size
    Gnum = ginit.size
    maxiter = max(maxiter, MINITER)

    wvis = vis / vis_std
    wy = y / vis_std

    if rho_init is None:
        rho_init = estimate_rho_init(wvis, vid2gid_st, Gnum)

    g = ginit.astype(np.complex128).copy()
    h = g.copy()

    pow_wvis = np.abs(wvis) ** 2
    wyvis_conj = wy * np.conj(wvis)

    sig = np.ones(Stnum, dtype=np.float64)

    logger.debug(
        "Self calibration: M=%d Stnum=%d Tnum=%d Gnum=%d lambda_1=%s lambda_2=%s rho_init=%s "
        "maxiter=%d eps=%s",
        M, Stnum, Tnum, Gnum, lambda_1, lambda_2, rho_init, maxiter, eps
    )

    cost = cost_g(wvis, wy, sig, g, gid_adj_t, vid2gid_st, time_tbl, lambda_1, lambda_2, prt=True)

    rho = rho_init
    converged = False
    cost_new = 0.0

    for _outer in range(MAXOUTER):
        logger.debug("rho is: %s", rho)
        flag = False
        for i in range(maxiter):
            g = update_g(
                pow_wvis, wyvis_conj, sig, g, h, gid_adj_t, vid2gid_st, time_tbl,
                Gnum, Stnum, lambda_1, lambda_2, rho
            )
            h = update_g(
                pow_wvis, wyvis_conj, sig, h, g, gid_adj_t, vid2gid_st, time_tbl,
                Gnum, Stnum, lambda_1, lambda_2, rho
            )

            cost_new = cost_gh(
                wvis, wy, sig, g, h, gid_adj_t, vid2gid_st, time_tbl,
                lambda_1, lambda_2, rho, prt=False
            )

            if i % 100 == 0:
                logger.debug("Iteration %d: cost is %f", i + 1, cost_new)

            if i > MINITER and abs(cost - cost_new) < eps * max(1.0, abs(cost_new)):
                flag = True
                break
            cost = cost_new

        if flag and np.sum(np.abs(g - h) ** 2) / Gnum < GH_DIFF_TH:
            cost = cost_gh(
                wvis, wy, sig, g, h, gid_adj_t, vid2gid_st, time_tbl,
                lambda_1, lambda_2, rho, prt=True
            )
            converged = True
            break

        # Recalibrate rho from the actual current h rather than blindly
        # multiplying by RHOSTEP -- but never let it grow slower than the
        # RHOSTEP schedule would, to keep the guarantee of eventually
        # reaching (g, h) consensus within MAXOUTER stages.
        rho = max(estimate_rho(wvis, vid2gid_st, h), RHOSTEP * rho)
        cost = cost_new

    if not converged:
        logger.debug("not converged")

    g = (g + h) / 2.0

    cost = cost_g(wvis, wy, sig, g, gid_adj_t, vid2gid_st, time_tbl, lambda_1, lambda_2, prt=True)

    return PySelfCalResults(gain=g, finalcost=cost, converged=converged)


def modify_visibility(
        vis: np.ndarray,
        gain: np.ndarray,
        vid2gid_st: np.ndarray
) -> np.ndarray:
    """Apply estimated gains to visibilities: vis_i * g[a] * conj(g[b]).

    Mirrors c++/selfcal.cpp::modify_visibility.
    """
    gid1 = vid2gid_st[:, 0]
    gid2 = vid2gid_st[:, 1]
    return vis * gain[gid1] * np.conj(gain[gid2])

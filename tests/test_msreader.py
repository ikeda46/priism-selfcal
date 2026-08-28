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
Real-data test of msreader.py against an actual Measurement Set.

Unlike this package's other tests (all synthetic), read_ms_for_selfcal/
read_multi_ms_for_selfcal wrap priism's real MS-reading path
(AlmaSparseModelingImager.readvis(with_gain_metadata=True), on priism's
"main" branch as of PR #71, 2026-08-27) and so need a real CASA install
and a real MS to exercise. Set
PRIISM_SELFCAL_TEST_MS to point at one; tests are skipped (not failed) if
it's unset/missing, so this file doesn't break environments without
CASA/test data (e.g. plain CI).

Validated manually against data/concat.ms.cal.HD142527.avg60 (2026-08-21):
- single-MS read: antenna1/antenna2/time line up 1:1 with an independent
  direct MS table read (see priism's own commit history for that
  validation -- msreader.py itself adds no new row-correspondence logic,
  it only consumes readvis(with_gain_metadata=True)'s already-validated
  output).
- multi-MS antenna_offset: reading the same MS twice as if it were two
  separate MSs gives disjoint station_ids ranges ([0, N) and [N, 2N) for
  N=ANTENNA subtable row count), and write_gaintable(antenna_offset=N)
  correctly restores the second "MS"'s ANTENNA1 back to [0, N).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from priism_selfcal.msreader import (
    count_antennas, read_ms_for_selfcal, read_multi_ms_for_selfcal
)
from priism_selfcal.casa_gaintable import write_gaintable

MSNAME = os.environ.get(
    'PRIISM_SELFCAL_TEST_MS',
    '/Users/shiro/pyenv/selfcal/data/concat.ms.cal.HD142527.avg60'
)
HAS_TEST_MS = os.path.exists(MSNAME)
_SKIP_REASON = (
    f'test MS not found at "{MSNAME}" -- set PRIISM_SELFCAL_TEST_MS to a '
    'real MS to run this test'
)


def _skip_if_no_test_ms():
    if not HAS_TEST_MS:
        pytest.skip(_SKIP_REASON)


def test_read_ms_for_selfcal_builds_consistent_gains():
    _skip_if_no_test_ms()

    result = read_ms_for_selfcal(
        MSNAME, spw='0', imsize=512, cell='0.01arcsec', field='0', datacolumn='data'
    )

    n_ant = count_antennas(MSNAME)
    assert result.antenna_offset == 0
    assert result.gains.station_ids.min() >= 0
    assert result.gains.station_ids.max() < n_ant
    assert result.gains.vis_num == len(result.imager.working_set)
    # every visibility's gain indices must point at valid gain rows
    assert result.gains.vid2gid_st.max() < result.gains.gain_num
    assert result.gains.vid2gid_st.min() >= 0


def test_read_multi_ms_for_selfcal_disambiguates_antenna_numbering():
    _skip_if_no_test_ms()

    # same MS "twice" stands in for two distinct MSs here; antenna_offset
    # only depends on ANTENNA subtable size, not on which MS it actually is
    specs = [dict(msname=MSNAME, spw='0'), dict(msname=MSNAME, spw='0')]
    results = read_multi_ms_for_selfcal(specs, imsize=512, cell='0.01arcsec')

    n_ant = count_antennas(MSNAME)
    assert results[0].antenna_offset == 0
    assert results[1].antenna_offset == n_ant

    assert results[0].gains.station_ids.max() < n_ant
    assert results[1].gains.station_ids.min() >= n_ant
    assert results[1].gains.station_ids.max() < 2 * n_ant

    # the two MSSelfCalData's gains are independent objects (no shared state)
    assert results[0].gains is not results[1].gains


def test_write_gaintable_antenna_offset_round_trip(tmp_path):
    _skip_if_no_test_ms()

    specs = [dict(msname=MSNAME, spw='0'), dict(msname=MSNAME, spw='0')]
    results = read_multi_ms_for_selfcal(specs, imsize=512, cell='0.01arcsec')

    out_path = str(tmp_path / 'offset_test.g')
    write_gaintable(
        results[1].gains, MSNAME, out_path, spw=0, field=0,
        overwrite=True, antenna_offset=results[1].antenna_offset
    )

    import casatools
    tb = casatools.table()
    tb.open(out_path)
    try:
        ant1 = tb.getcol('ANTENNA1')
    finally:
        tb.close()

    n_ant = count_antennas(MSNAME)
    assert ant1.min() >= 0
    assert ant1.max() < n_ant

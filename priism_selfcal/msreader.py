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
Read one or more Measurement Sets for imaging + self-calibration.

This is a thin wrapper around priism's own MS-reading path
(AlmaSparseModelingImager.readvis(with_gain_metadata=True), on priism's
pysparseimaging branch) --
imaging arrays (u/v/rdata/idata/weight) and gain-metadata arrays
(antenna1/antenna2/time) come from the exact same MS scan, so their
row-by-row correspondence is guaranteed by construction rather than by
independently reproducing priism's row-filtering/channel logic here (see
priism-selfcal's project notes on why "two separate reads" was rejected
as a design, 2026-08-21).

Design choice: imaging ingestion and gain-metadata ingestion are exposed
as separate concerns at the API level (with_gain_metadata defaults False
in priism, so plain-imaging callers never pay for or see this), but for
multi-MS self-calibration, each MS is read with its own
AlmaSparseModelingImager/readvis() call rather than relying on priism's
own multi-visparam concatenation -- real ALMA antenna IDs are local to
each MS (not globally comparable), so antenna numbering must be
disambiguated per MS (see antenna_offset below) before any cross-MS
Gains bookkeeping is meaningful.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import casatools

from priism.alma.imager import AlmaSparseModelingImager

from .gains import Gains

MSSelfCalData = namedtuple(
    'MSSelfCalData',
    ['msname', 'imager', 'gains', 'antenna_offset']
)


def count_antennas(msname: str) -> int:
    """
    Number of rows in msname's ANTENNA subtable.

    Used as the per-MS antenna-numbering block size for
    read_multi_ms_for_selfcal's cumulative antenna_offset -- deliberately
    the *subtable* row count, not the number of distinct antennas actually
    appearing in the (possibly flagged/selected) visibility data, so the
    offset is a fixed property of the MS and doesn't shift if the data
    selection changes.
    """
    tb = casatools.table()
    tb.open(msname + '/ANTENNA')
    try:
        return tb.nrows()
    finally:
        tb.close()


def read_ms_for_selfcal(
        msname: str,
        spw,
        imsize,
        cell,
        field='0',
        datacolumn: str = 'data',
        nchan: int = 1,
        start: str = '',
        width: str = '',
        antenna_offset: int = 0,
        solver: str = 'mfista_nufft',
) -> MSSelfCalData:
    """
    Read one MS's visibility data for imaging + self-calibration.

    Args:
        msname -- path to the Measurement Set
        spw, field, datacolumn -- passed to AlmaSparseModelingImager.selectdata()
        imsize, cell -- passed to AlmaSparseModelingImager.defineimage();
                        phasecenter is set to `field` (this package only
                        supports specifying phasecenter by field ID, same
                        restriction as priism's own defineimage())
        nchan, start, width -- passed to defineimage(); default (nchan=1,
                        start='', width='') maps all selected visibility
                        channels into one continuum image channel
        antenna_offset -- added to this MS's local ANTENNA1/ANTENNA2
                        indices before building Gains. Real ALMA antenna
                        IDs are local to each MS (antenna 5 in one MS need
                        not be the same physical station as antenna 5 in
                        another), so combining multiple MSs into one
                        self-calibration problem requires giving each MS's
                        antennas a disjoint numeric range first --
                        see read_multi_ms_for_selfcal, which manages this
                        automatically. Pass 0 (the default) for single-MS
                        use, where local IDs need no adjustment.
        solver -- passed to AlmaSparseModelingImager(solver=...)

    Returns:
        MSSelfCalData(msname, imager, gains, antenna_offset)
        -- imager.working_set holds u/v/rdata/idata/weight for imaging,
           unaffected by antenna_offset; gains is a Gains instance built
           from this MS's (offset-adjusted) antenna1/antenna2 and time,
           ready for run_self_calibration(). To recover this MS's
           original (local) antenna numbering later (e.g. when writing a
           CASA gaintable back out for this MS), subtract antenna_offset
           from gains.station_ids -- see casa_gaintable.write_gaintable's
           antenna_offset parameter.
    """
    im = AlmaSparseModelingImager(solver=solver)
    im.selectdata(vis=msname, spw=str(spw), field=str(field), datacolumn=datacolumn)
    im.defineimage(imsize=imsize, cell=cell, phasecenter=str(field),
                   nchan=nchan, start=start, width=width)
    im.readvis(with_gain_metadata=True)

    ws = im.working_set
    global_antenna1 = ws.antenna1.astype(np.int64) + antenna_offset
    global_antenna2 = ws.antenna2.astype(np.int64) + antenna_offset
    gains = Gains(global_antenna1, global_antenna2, ws.time)

    return MSSelfCalData(msname=msname, imager=im, gains=gains, antenna_offset=antenna_offset)


def read_multi_ms_for_selfcal(
        specs,
        imsize,
        cell,
        solver: str = 'mfista_nufft',
) -> list[MSSelfCalData]:
    """
    Read multiple MSs for a joint self-calibration problem, assigning each
    MS a disjoint block of antenna numbers so their Gains bookkeeping
    never collides (see read_ms_for_selfcal's antenna_offset).

    Args:
        specs -- list of dicts, each describing one MS with key 'msname'
                 (required) and optionally 'spw' (default '0'), 'field'
                 (default '0'), 'datacolumn' (default 'data'), 'nchan'
                 (default 1), 'start' (default ''), 'width' (default '')
                 -- see read_ms_for_selfcal for their meaning.
        imsize, cell -- shared imaging grid, passed to every MS's
                        defineimage() (all MSs must image onto the same
                        pixel grid for their visibilities to be
                        comparable/combinable)
        solver -- passed to every AlmaSparseModelingImager(solver=...)

    Returns:
        list of MSSelfCalData, one per spec, in the same order, with
        cumulative antenna_offset values (0 for the first MS, then
        increasing by each preceding MS's ANTENNA subtable row count).

        Self-calibration itself is still run per MS (each MSSelfCalData's
        gains is independent); if imaging wants a single combined
        visibility array across MSs, concatenate
        result.imager.working_set.{u,v,rdata,idata,weight} yourself and
        keep track of which MS each row came from (e.g. via a parallel
        ms-index array) so update_visibility() can be applied with the
        right MS's gains to the right rows.
    """
    results = []
    offset = 0
    for spec in specs:
        msname = spec['msname']
        result = read_ms_for_selfcal(
            msname=msname,
            spw=spec.get('spw', '0'),
            imsize=imsize, cell=cell,
            field=spec.get('field', '0'),
            datacolumn=spec.get('datacolumn', 'data'),
            nchan=spec.get('nchan', 1),
            start=spec.get('start', ''),
            width=spec.get('width', ''),
            antenna_offset=offset,
            solver=solver,
        )
        results.append(result)
        offset += count_antennas(msname)
    return results

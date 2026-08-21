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
Write a Gains object out as a CASA "G Jones" calibration table, so that
familiar CASA tools (plotcal, applycal, browsetable, etc.) can be used on
self-calibration results produced by this package.

A CASA calibration table's exact low-level structure (column descriptors,
subtable linkage, VisCal/ParType/MSName keywords) is set by CASA's own C++
calibration-table writer and is not practical to hand-build reliably from
scratch. Instead, this module has CASA itself create a correctly-formed
template (via a throwaway casatasks.gaincal solve on the same MS/field/spw)
and then overwrites its rows with the Gains object's own data -- this
guarantees the output is byte-for-byte a valid CASA caltable structurally,
differing from a "real" gaincal solution only in its data values.

Caveat: our self-calibration solves a single complex gain per (antenna,
time) from Stokes-I-combined visibilities, whereas CASA's G Jones tables
store one gain *per polarization* (CPARAM shape (2, 1, nrow) for typical
dual-polarization ALMA data). We do not have separate per-polarization
gain estimates, so the same Stokes-I-derived gain value is written into
both polarization slots.
"""
from __future__ import annotations

import logging
import os
import shutil

import numpy as np
import casatools
import casatasks

from .gains import Gains

logger = logging.getLogger(__name__)


def _make_template(msname: str, field: str, spw: str, refant: str, template_path: str) -> None:
    """A fast, throwaway gaincal solve purely to get a correctly-structured
    CASA caltable (subtables, keywords, column descriptors) to overwrite.
    Its solved values are discarded entirely.
    """
    if os.path.exists(template_path):
        shutil.rmtree(template_path)
    casatasks.gaincal(
        vis=msname, caltable=template_path, field=field, spw=spw,
        solint='inf', calmode='ap', refant=refant, minsnr=0.0,
    )


def write_gaintable(
        gains: Gains,
        msname: str,
        output_path: str,
        spw: int = 0,
        field: int = 0,
        scan_number: int = -1,
        observation_id: int = 0,
        refant: str | None = None,
        overwrite: bool = False,
) -> None:
    """
    Write `gains` out to `output_path` as a CASA G-Jones calibration table.

    Args:
        gains -- a Gains instance, already solved (gains.gain populated)
        msname -- path to the MS the gains were derived from (used only to
                  build the structural template; not modified)
        output_path -- path for the new calibration table
        spw, field -- SPECTRAL_WINDOW_ID/FIELD_ID to record for every row
                      (this package currently solves one spw/field at a time)
        scan_number -- SCAN_NUMBER to record for every row; -1 (CASA's usual
                        "unspecified/all scans" convention) if not tracked
        observation_id -- OBSERVATION_ID to record for every row
        refant -- antenna name to use for the throwaway structural template
                  solve (arbitrary choice, does not affect the written
                  values); defaults to the first antenna in the MS
        overwrite -- remove an existing table at output_path first if True
    """
    if os.path.exists(output_path):
        if not overwrite:
            raise RuntimeError(f'"{output_path}" already exists (pass overwrite=True to replace it)')
        shutil.rmtree(output_path)

    if refant is None:
        tb = casatools.table()
        tb.open(msname + '/ANTENNA')
        refant = tb.getcol('NAME')[0]
        tb.close()

    template_path = output_path + '.template'
    _make_template(msname, field=str(field), spw=str(spw), refant=refant, template_path=template_path)

    Gnum = gains.gain_num

    tb = casatools.table()
    tb.open(template_path, nomodify=False)
    current_nrow = tb.nrows()
    if current_nrow < Gnum:
        tb.addrows(Gnum - current_nrow)
    elif current_nrow > Gnum:
        tb.removerows(list(range(Gnum, current_nrow)))

    station = gains.station_ids[gains.gid_adj_t[:, 0]]
    time_col = gains.time_tbl[gains.gid_adj_t[:, 1]]

    tb.putcol('TIME', time_col)
    tb.putcol('FIELD_ID', np.full(Gnum, field, dtype=np.int32))
    tb.putcol('SPECTRAL_WINDOW_ID', np.full(Gnum, spw, dtype=np.int32))
    tb.putcol('ANTENNA1', station.astype(np.int32))
    tb.putcol('ANTENNA2', np.zeros(Gnum, dtype=np.int32))
    tb.putcol('INTERVAL', np.zeros(Gnum, dtype=np.float64))
    tb.putcol('SCAN_NUMBER', np.full(Gnum, scan_number, dtype=np.int32))
    tb.putcol('OBSERVATION_ID', np.full(Gnum, observation_id, dtype=np.int32))

    npol = tb.getcell('CPARAM', 0).shape[0]  # (2, 1) for a standard dual-pol G table
    cparam = np.broadcast_to(gains.gain[np.newaxis, np.newaxis, :], (npol, 1, Gnum)).copy()
    tb.putcol('CPARAM', cparam)
    tb.putcol('FLAG', np.zeros((npol, 1, Gnum), dtype=bool))
    tb.putcol('PARAMERR', np.zeros((npol, 1, Gnum), dtype=np.float64))
    tb.putcol('SNR', np.zeros((npol, 1, Gnum), dtype=np.float64))
    if 'WEIGHT' in tb.colnames():
        # WEIGHT is optional/not always populated per-row in a gaincal
        # template (e.g. rows corresponding to flagged solutions); skip it
        # rather than fail the whole write over non-essential metadata.
        try:
            weight_cell_shape = tb.getcell('WEIGHT', 0).shape
            tb.putcol('WEIGHT', np.ones(weight_cell_shape + (Gnum,), dtype=np.float64))
        except RuntimeError:
            logger.debug("could not determine WEIGHT column shape; leaving it as the template had it")

    tb.close()

    os.rename(template_path, output_path)

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
priism's "sakura" alignment/gridding helper library (libsakurapy) is a
separate, sometimes hard-to-obtain C extension that priism-selfcal's own
NUFFT-based imaging path (solver='pymfista_nufft') doesn't actually need --
only priism.core.datacontainer/alma.gridder import it at module load time.
Stub it out here, before any test module gets a chance to import priism,
so the whole suite is runnable without it installed. If it *is* installed,
this stub is simply never used (the real import wins).
"""
import sys
import types

import numpy as np

try:
    import priism.external.sakura  # noqa: F401
except ImportError:
    sakura_stub = types.ModuleType('priism.external.sakura')
    sakura_stub.empty_aligned = lambda shape, dtype=np.float64: np.empty(shape, dtype=dtype)
    sakura_stub.empty_like_aligned = lambda a: np.empty_like(a)
    sys.modules['priism.external.sakura'] = sakura_stub

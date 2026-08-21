# priism-selfcal

Self-calibration for ALMA sparse-modeling imaging (Ikeda et al. 2025),
standalone from [priism](https://github.com/tnakazato/priism) (a runtime
dependency, not a fork). Currently depends on priism's
`selfcal-gain-metadata` branch (adds the `antenna1`/`antenna2`/`time`
metadata self-cal needs to `readvis()`) rather than `pysparseimaging`;
see that branch's own history for details.

## Usage

### Reading a Measurement Set

`msreader.read_ms_for_selfcal()` wraps priism's own MS-reading path and
returns both an imaging-ready `imager` and a `Gains` instance built from
the same visibilities:

```python
from priism_selfcal.msreader import read_ms_for_selfcal

data = read_ms_for_selfcal(
    msname='mydata.ms', spw='0', imsize=512, cell='0.01arcsec', field='0',
    datacolumn='data', solver='pymfista_nufft',
)
# data.imager        -- a priism AlmaSparseModelingImager, working_set already set
# data.gains         -- a Gains instance (one complex gain per station/integration)
# data.antenna_offset -- 0 for single-MS use; see read_multi_ms_for_selfcal for multiple MSs
```

`read_multi_ms_for_selfcal()` does the same for a list of MSs, assigning
each a disjoint block of antenna numbers so their `Gains` never collide
(real ALMA antenna IDs are local to each MS).

### Self-calibration building blocks

`Gains` (`gains.py`) holds the per-(station, integration-time) complex
gains and the bookkeeping tables that relate them to visibilities.
`run_self_calibration()`/`update_visibility()` (`selfcal.py`) are the
low-level primitives: given a current image, solve for the gains that
best explain the visibility residual, then apply them:

```python
import numpy as np
from priism_selfcal.selfcal import run_self_calibration, update_visibility

ws = data.imager.working_set
vis_org = ws.rdata + 1j * ws.idata
sigma = 1.0 / np.sqrt(ws.weight)

# xin: current best image estimate (e.g. from a plain MFISTA solve)
result = run_self_calibration(
    gains=data.gains, u_pix=ws.u, v_pix=ws.v, vis=vis_org, sigma=sigma,
    xin=xin, imsize=(512, 512), mu_sq=1e8, mu_abs=1e7,
)
# data.gains.gain is updated in place; result.converged reports whether
# the ADMM solve actually converged (see run_self_calibration's docstring)

vis_cal = update_visibility(data.gains, vis_org)  # gain-corrected visibility
```

Real ALMA gains should stay clustered near `1+0j` in the complex plane
-- check with `data.gains.plot_gains()` (complex-plane scatter) or
`data.gains.plot_station_gains()` (per-station amplitude/phase vs. time)
after any solve; values far from `1+0j` indicate a problem (usually bad
imaging parameters, not a self-cal bug -- see the TotalImaging section
below).

### Making an image (alternating imaging <-> self-calibration)

Most users don't want to hand-roll the `run_self_calibration()` /
re-image / repeat loop themselves -- `totalimaging.run()` does exactly
that, alternating priism's own MFISTA solve with a self-calibration
round each pass, for **caller-supplied** `l1`, `ltsv`, `mu_sq`, `mu_abs`:

```python
from priism_selfcal import totalimaging

result = totalimaging.run(
    imager=data.imager, vis_org=vis_org, sigma=sigma, gains=data.gains,
    l1=100.0, ltsv=100.0, mu_sq=1e8, mu_abs=1e7,
)
# result.image      -- the final image (numpy array)
# result.cost       -- final combined imaging+gain cost
# result.converged  -- whether the alternation's own stopping criterion fired
# data.gains.gain is updated in place to the final self-cal solution
```

If you don't already know good values for `l1`/`ltsv`/`mu_sq`/`mu_abs`,
use `paramsearch.run_staged_parameter_search()` instead (below) -- it
finds them automatically and returns the final image and gains directly,
without you needing to call `totalimaging.run()` yourself at all.

### Writing a CASA gaintable

`casa_gaintable.write_gaintable()` exports a solved `Gains` instance as a
CASA "G Jones" calibration table, so familiar CASA tools (`plotcal`,
`applycal`, `browsetable`) can work with the result:

```python
from priism_selfcal.casa_gaintable import write_gaintable

write_gaintable(
    data.gains, msname='mydata.ms', output_path='mydata.selfcal.gaincal',
    spw=0, field=0, antenna_offset=data.antenna_offset,  # 0 for single-MS use
)
```

## TotalImaging: choosing `selfcal_num` / `total_eps`

`totalimaging.run()` alternates imaging (MFISTA) and self-calibration
(gain estimation) until either `selfcal_num` rounds have run or the
tracked cost (imaging cost + gain regularizer) stops changing by more
than `total_eps` between rounds.

**Increasing `selfcal_num` beyond the default (10) is not guaranteed to
produce a better image.** Self-calibration has enough freedom in the
per-station complex gains to keep "explaining away" real image structure
as gain/phase error if allowed to alternate too many times -- the tracked
cost can keep decreasing while the image itself drifts in a worse
direction (e.g. phase gets over-fit to noise or to structure that should
have stayed in the image). The original reference implementation
(`old/python/example_HD142527.py`) always ran with a small, fixed round
budget rather than iterating to convergence, and its default `total_eps`
(1e-6) was, in practice, rarely tight enough to actually trigger before
that budget ran out -- the round cap was the real stopping mechanism, not
the cost threshold.

If you do raise `selfcal_num`, treat it as an experiment: inspect the
resulting gains with `Gains.plot_gains()` (real ALMA gains should stay
clustered near `1+0j`) and compare the image against a smaller-budget run
before trusting it.

## paramsearch: choosing (lambda1, lambda_tsv, mu_sq, mu_abs)

`paramsearch.run_staged_parameter_search()` searches all four imaging/
self-cal regularization weights, following the staged Bayesian-optimization
procedure in Ikeda et al. 2025 (PASJ 77(2):260-276, section 3.4) rather
than a joint 4-D search (which the paper explicitly rejects as too slow,
and which self-calibration's own dynamics -- gains chasing whatever image
is currently on hand -- make a poor fit for anyway):

```python
from priism_selfcal import paramsearch

target = paramsearch.GAIN_TARGETS['large_variance']  # or 'small_variance', or a custom GainTarget(sigma_ph=..., sigma_amp=...)

result = paramsearch.run_staged_parameter_search(
    data.imager, ws.antenna1, ws.antenna2, ws.time, vis_org, sigma, target,
)

print(result.l1, result.ltsv, result.mu_sq, result.mu_abs)
result.gains.plot_gains()  # sanity check: should cluster near 1+0j
```

The search alternates two independent 2-parameter stages rather than one
joint 4-D one, following the paper's own explicit section 3.4 procedure:

1. **stage 0** -- search `(lambda1, lambda_tsv)` with gains fixed at
   `1+0j` (i.e. no self-calibration at all yet), via priism's own
   `optimizeparameters(criterion='ellipsoid', optimizer='bayesian')`
   (eq. 16) on the raw, uncorrected visibility.
2. **mu_round0** -- fix `(lambda1, lambda_tsv)` to stage 0's winner,
   search `(mu_sq, mu_abs)` from scratch (gains re-initialized to `1+0j`
   every trial). Each trial runs a full `totalimaging.run()` and is
   scored by how close the resulting gains' phase/amplitude standard
   deviation come to `target` (`paramsearch.gain_dispersion_cost`,
   eq. 17).
3. **lambda_round0** -- fix gains to `mu_round0`'s winner, re-search
   `(lambda1, lambda_tsv)`, narrowed to stage 0's winner +/-
   `narrow_width` decades (default 3, matching the historical scripts).

By default (`n_refine_rounds=1`) the search stops there, matching what
the historical reference scripts (`old/python/*_step0.py` ..
`*_step3.py`) actually ran in practice (traced via their
hardcoded-constant provenance chain, 2026-08-21) -- even though the
paper itself says steps 2-3 can be repeated further "since the image
does not change greatly after a few iterations". Pass
`n_refine_rounds=2` to repeat the mu/lambda refinement one more time
(`mu_round1`, `lambda_round1`), narrowing further around the previous
round's winners each time.

`search_gain_regularizers()` and `search_imaging_regularizers()` (the
two per-round stage functions) are also usable standalone if you only
need one of them -- e.g. to re-tune `mu_sq`/`mu_abs` alone with
`lambda1`/`lambda_tsv` already decided some other way.

Each stage runs `bayesopt_maxiter_lambda`/`bayesopt_maxiter_mu` Optuna
trials (both default to 30, matching the paper); each mu-search trial
runs a full `totalimaging.run()`, so the whole procedure can take a
while on real data (on 512x512/~50000-visibility HD142527 data, roughly
an hour for the default single-round search) -- start with small values
while testing your setup, following this package's test suite
(`tests/test_paramsearch.py`) as a template for a fast, synthetic-data
smoke test before running on a real MS.

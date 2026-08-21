# priism-selfcal

Self-calibration for ALMA sparse-modeling imaging (Ikeda et al. 2025),
standalone from [priism](https://github.com/tnakazato/priism) (a runtime
dependency, not a fork).

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
from priism.core import imager as core_imager
from priism_selfcal.msreader import read_ms_for_selfcal
from priism_selfcal import paramsearch

data = read_ms_for_selfcal(
    msname='mydata.ms', spw='0', imsize=512, cell='0.01arcsec', field='0',
)

target = paramsearch.GAIN_TARGETS['large_variance']  # or 'small_variance', or a custom GainTarget(sigma_ph=..., sigma_amp=...)

result = paramsearch.run_staged_parameter_search(
    data.imager, data.imager.working_set.antenna1, data.imager.working_set.antenna2,
    data.imager.working_set.time, data.imager.working_set.rdata + 1j * data.imager.working_set.idata,
    1.0 / np.sqrt(data.imager.working_set.weight),
    target,
)

print(result.l1, result.ltsv, result.mu_sq, result.mu_abs)
result.gains.plot_gains()  # sanity check: should cluster near 1+0j
```

The search alternates two independent 2-parameter stages rather than one
joint 4-D one:

1. **stage 0** -- search `(mu_sq, mu_abs)` from scratch (gains start at
   `1+0j` every trial), with `lambda1 = lambda_tsv = 1.0` fixed. Each
   trial runs a full `totalimaging.run()` and is scored by how close the
   resulting gains' phase/amplitude standard deviation come to `target`
   (`paramsearch.gain_dispersion_cost`, eq. 17 of the paper).
2. **stage 1** -- fix gains to stage 0's winner, search `(lambda1,
   lambda_tsv)` via priism's own `optimizeparameters(criterion=
   'ellipsoid', optimizer='bayesian')` (eq. 16) -- no self-calibration in
   this stage, just imaging.
3. **stage 2** -- fix `(lambda1, lambda_tsv)` to stage 1's winner,
   re-search `(mu_sq, mu_abs)` narrowed to stage 0's winner +/-
   `narrow_width` decades (default 3, matching the historical scripts).
4. **stage 3** -- fix `(mu_sq, mu_abs)` to stage 2's winner, re-search
   `(lambda1, lambda_tsv)` narrowed to stage 1's winner +/- `narrow_width`
   decades.

The search stops after stage 3 (0 -> 1 -> 2 -> 1, not a repeating loop).
This matches what the historical reference scripts
(`old/python/*_step0.py` .. `*_step3.py`) actually ran, traced via their
hardcoded-constant provenance chain -- not the paper's own text, whose
"we repeated steps 2 and 3 for 10 iterations" turned out to reuse wording
from a *different* iteration count elsewhere in the paper (the inner
TotalImaging imaging<->gain loop, unrelated to this outer parameter
search) rather than describing a literal 10x repeat of the outer search.

`search_gain_regularizers()` and `search_imaging_regularizers()` (the two
stages) are also usable standalone if you only need one of them -- e.g.
to re-tune `mu_sq`/`mu_abs` alone with `lambda1`/`lambda_tsv` already
decided some other way.

Each stage runs `bayesopt_maxiter_stageN` Optuna trials (default 30,
matching the paper); each gain-search trial runs a full
`totalimaging.run()`, so the whole procedure can take a while on real
data -- start with small values while testing your setup, following this
package's test suite (`tests/test_paramsearch.py`) as a template for a
fast, synthetic-data smoke test before running on a real MS.

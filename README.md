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

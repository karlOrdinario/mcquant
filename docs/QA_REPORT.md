# QA report

This report covers how `mcquant` was verified, what broke along the way, and what is still a known limitation. Every defect below has a regression test, so it cannot quietly return.

## How the engine was verified

| Layer | What was done | Result |
|---|---|---|
| Unit and analytic tests | Closed forms checked against textbook values, put-call parity, in/out barrier parity, and finite differences of the pricer | pass |
| Statistical tests | Every Monte Carlo price within 4 SE of an independent reference: Black-Scholes, Heston characteristic function, Merton series, BGK barrier formula, Bermudan binomial tree | pass |
| SE honesty | 95% confidence-interval coverage measured over 300 seeds for each sampler | pass |
| Bias study | 13 benchmarks × 40 seeds; mean z-score per benchmark | no benchmark biased (all \|t\| < 2) |
| Property-based tests | Hypothesis: 200 random parameter sets checked against no-arbitrage bounds | pass |
| Code review | Structured review of correctness, numerical stability, edge cases and performance | 7 issues found, all fixed |
| Edge-case probes | Zero/negative inputs, singular covariance, extreme jump rates, deep OTM and short-dated Heston, S0 on the barrier | 4 issues found, all fixed |
| Memory profiling | `tracemalloc` peak per simulation | 2 issues found, both fixed |
| UI tests | Streamlit AppTest on every model × contract combination, plus both buttons | pass |
| Visual QA | Headless Chrome screenshots of all 6 tabs, inspected by eye | 6 layout issues found, all fixed |
| Lint and format | ruff (pyflakes, bugbear, numpy rules, import order) | clean |

**Final state:** 197 tests passing, 96% line coverage, lint clean.

## Defects found and fixed

| # | Severity | Defect | How it was found | Fix | Guarded by |
|---|---|---|---|---|---|
| 1 | Critical | **Merton series double-counted a discount factor**: each Poisson term was multiplied by $e^{(r_n-r)T}$, but weighting with $\lambda' = \lambda(1+k)$ already absorbs it | Re-deriving the formula during the review | Removed the extra factor | `test_merton_vs_series`, `test_no_jumps_is_black_scholes` |
| 2 | High | **Control-variate coefficient fitted on the wrong samples**: with antithetic sampling, β was fitted on raw draws, but the estimator averages pairs. Antithetic + CV gave SE 0.0303, *worse* than CV alone at 0.0218 | Variance-reduction table in the dashboard looked wrong | Fit β on the pair means; SE fell to 0.0107 | `test_antithetic_plus_control_variate_beats_each_alone` |
| 3 | High | **Randomised-QMC confidence intervals under-covered**: 16 scrambles with a normal 1.96 quantile covered only 89–92% instead of 95%, and \|z\| > 4 appeared in 1–1.6% of runs | 40-seed bias study (z std 1.71), then a 1,000-replication coverage study | 32 scrambles + Student-t quantile, giving 93–94% coverage | `test_rqmc_interval_uses_student_t_and_covers`, `test_confidence_interval_coverage[sobol]` |
| 4 | High | **Memory blow-up**: Heston peaked at 9× the path-matrix size and GBM at 5×. At 100k × 252 Heston needs ~1.8 GB, which crashes a 1 GB Streamlit Cloud container | `tracemalloc` profiling | In-place NumPy operations, correlating one column per step, freeing intermediates. Heston 4×, GBM 2× | `test_simulation_memory_budget` |
| 5 | Medium | **Silent NaN**: `bs_price` with σ ≤ 0, K ≤ 0 or S ≤ 0 returned NaN with no error | Edge-case probes with `-W error` | Input validation that raises `ValueError` | `test_black_scholes_rejects_invalid_inputs_instead_of_nan` |
| 6 | Medium | **Perfectly correlated assets rejected**: Cholesky fails on a positive *semi*-definite covariance | Edge-case probes | Eigen-decomposition square root as a fallback | `test_singular_covariance_is_accepted` |
| 7 | Medium | **Merton series truncated** at a hard 200 terms, which is wrong when λT is large | Code review | Term count adapts to mean + 12 std of the Poisson | `test_merton_series_converges_for_frequent_jumps` |
| 8 | Medium | **Synthetic-market defaults had an infinite 4th moment** (GARCH α=0.08, β=0.90, t₅: β² + 2αβ + α²κ = 1.01 > 1), so sample vols badly missed their targets | `test_synthetic_market_shape_and_stats` failed | Defaults α=0.06, β=0.92, t₆ (0.978 < 1), plus a warning when the condition is violated | `test_synthetic_market_shape_and_stats` |
| 9 | Low | **S₀ column off by round-off**: `exp(log(100)) = 100.00000000000001`, which matters for a barrier sitting exactly at spot | `test_martingale` | Pin column 0 to S₀ exactly | `test_martingale` |
| 10 | Low | **Stress shock ≤ −100%** produced NaN instead of an error | Edge-case probes | Validation | `test_stress_shock_below_minus_100pct_rejected` |

Also fixed, outside the engine:

- **Benchmark script:** the "short-volatility book" accidentally held 2,000 long shares, so it was not delta-neutral and muddied the delta-normal comparison. It is now a clean sold straddle.
- **Benchmark script:** the first speed comparison used a single-step European, the best case for a Python loop. It now uses a 52-step Asian, where vectorisation actually matters, and checks that the two implementations agree.
- **Dashboard:**
  - chart legends overlapped titles
  - a 1-step European drew straight-line "paths"
  - log-axis tick labels were unreadable
  - NaN cells showed as "None"
  - the VaR table was truncated
  - the correlation heatmap's diagonal had low contrast

## Bias study (13 benchmarks × 40 seeds, 16,384 paths)

If an estimator is unbiased and its SE is honest, the z-scores (error / SE) average about 0 with a standard deviation of about 1. A mean whose |t| exceeds 2 would indicate bias.

| Benchmark | mean z | t-stat | std z |
|---|---|---|---|
| European call (GBM) | 0.00 | 0.02 | 1.12 |
| European put (antithetic) | −0.23 | −1.44 | 1.02 |
| European call (Sobol) | −0.45 | −1.67 | **1.71** → defect 3 |
| Digital call | 0.15 | 0.93 | 1.06 |
| Geometric Asian | −0.10 | −0.73 | 0.85 |
| Up-and-out call | −0.18 | −1.06 | 1.05 |
| Down-and-in put | −0.04 | −0.29 | 0.88 |
| Heston call | 0.02 | 0.11 | 0.93 |
| Merton call | −0.10 | −0.63 | 1.02 |
| American put (LSM) | −0.07 | −0.45 | 0.94 |
| Delta (pathwise) | −0.28 | −1.71 | 1.02 |
| Vega (pathwise) | −0.23 | −1.31 | 1.10 |
| Gamma (LR) | −0.14 | −0.90 | 0.98 |

No benchmark shows significant bias. The one anomaly, the Sobol z-scores with a standard deviation of 1.71, led to defect 3.

## RQMC coverage study (1,000 replications, European call K=110)

| Scrambles | Paths | Coverage of nominal 95% | Runs with \|z\| > 4 | RMSE |
|---|---|---|---|---|
| 16, normal quantile (old) | 65,536 | 90.8% | 16 | 0.00066 |
| 16, t quantile | 65,536 | 92.2% | 16 | 0.00066 |
| **32, t quantile (new default)** | 65,536 | **94.0%** | 2 | 0.00095 |
| 64, t quantile | 65,536 | 92.7% | 0 | 0.00137 |

The cause is a known property of randomised QMC. Each scramble places one point in the extreme tail cell of the inverse normal CDF, where the payoff is unbounded, so the batch means are slightly heavy-tailed and t-intervals under-cover a little (L'Ecuyer, Nakayama, Owen & Tuffin, 2023). 32 scrambles is the measured sweet spot: 64 doubles the error without covering better.

## Known limitations (accepted)

- **Heston discretisation bias.** Full-truncation Euler has O(Δt) bias, under 1 cent at 200 steps per year for the test parameters. The validation suite allows 2 cents. Andersen's QE scheme would reduce this.
- **RQMC intervals** cover about 93–94% at the nominal 95% level (see above).
- **Longstaff-Schwartz** prices a Bermudan option on the simulation grid. The gap to the true American price shrinks as exercise dates are added; at 50 dates per year it is about 0.009 for the standard test case.
- **Multi-day Student-t scenarios** scale a one-period t-distribution by √h, which is an approximation: sums of t variables are not t-distributed.
- **The BGK barrier correction** is an approximation with error of order 1e-3 relative. Tests allow 1 cent for it.

## Reproduce

```bash
pytest                                  # full suite, about 1 minute
pytest --cov=mcquant                    # coverage report
python scripts/run_benchmarks.py        # figures and docs/results.json
```

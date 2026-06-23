# Reverse-Engineering the Leaderboard — Findings

Target: best adjusted = **2.25e-7**, top-5 = **2.79e-7**. Our estimator
(whitened antithetic MC) = **3.72e-7** (multi-MLP avg); **5.12e-7** on the
single hard MLP `megan-chang` we could fully grade locally.

## Real grader numbers (single MLP, confirmed)

| Quantity | Value |
|---|---|
| Adjusted final-layer score | 5.12e-7 |
| Raw final-layer MSE | 5.08e-6 |
| Mean score multiplier | **0.10088** (at the floor) |
| flops_used | 2.66e10 |
| effective_compute | 2.74e10 = 0.1009·B |
| residual wall time | 7.9 ms |

`multiplier = max(0.1, effective_compute / B)`. We are 0.9% above the floor —
optimally tuned; trimming to exactly 0.1 changes the score <0.2%.

## The metric forces the problem onto variance-per-FLOP

At the floor, `adjusted ≈ V · flops_per_sample / (VR_factor · B)` — **independent
of sample count** (variance drop from more samples exactly cancels the multiplier
rise). The only levers are: lower per-sample variance `V`, higher `VR_factor`,
or a bias-free analytic term. The gap to the leader is **1.6× lower MSE at the
floor** — purely a variance-per-FLOP gap, not a budget-tuning gap.

## Whitening+antithetic = exact cubature for all degree-≤2 polynomials

Antithetic kills every odd-degree term; ZCA whitening forces batch mean=0 and
cov=I exactly, killing every degree-2 term. So the residual MC variance lives
entirely in **degree-≥4** even components. This is the structural ceiling, and
it explains why every affine/quadratic surrogate control gets annihilated.

## Every standard route — closed by direct measurement

| Hypothesis for the leaders' edge | Test | Result | Verdict |
|---|---|---|---|
| Degree-4 polynomial control | `vr_deg4.py` | R²=**0.004**, 1.00× | Dead — residual variance is degree-≥6, not low-order |
| Analytic + MC blend (skew/kurtosis) | `vr_edgeworth.py` | analytic MSE moves 3% (1.35e-4→1.31e-4) | Dead — blend caps ~3.6e-7 |
| Exact bivariate covariance | `vr_cv_gh_verify.py` | same E[a_L] error as gain-approx | Dead |
| Deep control variate (oracle) | `vr_cv_oracle_verify.py` | 4.99× ceiling, robust | Real but unrealizable |
| Deep CV via budget-split MLMC | `vr_mlmc_deepcv.py` | best **1.00×** (all splits net-negative) | Dead — E[a_L] cost always exceeds CV saving |
| Cheaper FLOP-counted forward | flopscope profile | matmul = honest 2·M·K·N | Dead — no counting trick |
| More compute | grader | multiplier cancels the gain | Dead |

## The structural theorem (why whitening is the ceiling)

**There is no cheaply-known-expectation control beyond degree-2 input moments.**
Every richer control's expectation is itself an unknown *deep-layer* expectation
(E[a_L], E[z_32], …), which can only be estimated — and estimating it
independently is always net-negative (measured: MLMC best 1.00×), while
estimating it from the same samples gives an identically-zero control. Whitening
works precisely because the degree-2 input expectation (I) is the one rich
control whose value is known a priori. Nothing else qualifies.

## Last-layer Rao-Blackwell — built and benchmarked, net-negative

`vr_lastlayer_rb.py` implements and grades four variants vs a 2M-sample GT
(5 seeds, 40 trials, N=5926):

| Method | MSE | vs direct |
|---|---|---|
| direct (current) | 5.495e-6 | 1.00× |
| gauss_rb (biased replacement) | 6.904e-6 | 0.80× |
| rb_cv (optimal λ≈0.35, unbiased) | 5.999e-6 | 0.92× |
| rb_debiased_split (unbiased) | 1.976e-5 | 0.28× |

**Theoretical per-neuron variance ceiling: 0.855× — below 1.0×.** Deep ReLU
pre-activations z_32 carry enough excess kurtosis that the empirical σ̂ is
*noisier* than the direct ReLU sample-mean, so even the perfectly-unbiased RB
has higher variance than direct. This lever cannot break even.

## Base-sampler search + realized improvement

Searched for a better variance-reduction constant; **none beats anti+ZCA**:
PCA-whiten (0.94×), radial stratification (0.98×), Sobol QMC (0.95×), ZCA-order
swaps (0.98×), multi-axis antithetic K=2…8 (0.80–0.96×; the apparent double-anti
1.065× was noise), and active-subspace quadrature (not viable: a_32 is output-rank
~2–3 but each mode depends on ~55–74 *input* directions). anti+ZCA is the
moment-cubature ceiling.

Realized change: **fold the ZCA transform into W₁** (`w0 = inv_sqrt @ W₁`) —
numerically identical output (diff 2e-6), but it removes a full (N×n) matmul,
trims per-sample cost ~2.8%, and lets us sit **safely at the 0.1 floor** (the old
estimator ran at 0.10088, slightly *over*). Net ≈1–2% lower adjusted score, with
a ~12ms wall-time margin so no MLP is pushed over the floor.

**Per-MLP MSE is high-variance** (rank-2 collapse → MSE averages over only ~2–3
independent mode-errors), so single-MLP grader scores swing ~2× run-to-run; only
the multi-MLP leaderboard average is meaningful.

## Conclusion

The leaders' ~1.6× edge is **not** from a control variate, a better analytic, a
blend, a last-layer Rao-Blackwell, or a FLOP-counting trick — all ruled out by
direct measurement. The remaining structurally-possible sources are narrow:

1. A **modestly better base-sampler constant** than ZCA+antithetic.
2. **Favorable benchmark averaging** — per-MLP MSE varies 2–9e-6; the headline
   average may reflect a method only ~1.2–1.3× better than ours.

Our 3.72e-7 is at the practical ceiling for the standard toolkit. Reaching
2.25e-7 requires a technique outside the (now exhaustively mapped) space of
moment-matching, control variates, analytic propagation, and MLMC.

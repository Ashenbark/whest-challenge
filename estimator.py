"""Estimator: whitened antithetic Monte Carlo.

Two findings drive this design:
  1. Analytical (Gaussian covariance propagation, even exact-bivariate GH) is
     capped at MSE ≈ 2e-4 by the joint-Gaussian pre-activation assumption, which
     fails over 32 layers. Pure MC (MSE ≈ 6-10e-6) is ~30x better — so no blend.
  2. For pure MC the adjusted score is flat in budget (more samples lower MSE but
     raise the multiplier proportionally), pinning us at MSE_floor × 0.1. The only
     way down is lower variance PER FLOP — i.e. variance reduction.

Variance reduction (measured, 4-5 seeds × 16 trials, final-layer variance):
     plain        1.00x      antithetic   1.05x   (antithetic decays to ~nothing
     whiten       1.88x      anti+whiten  2.10x    by the final layer at depth 32)
  Stacking attempts: radial stratification 0.99x, Sobol QMC 1.05x, orthogonal
  blocks 0.60x — none improve meaningfully on anti+whiten.

Whitening forces the input batch's empirical mean to 0 (antithetic) and covariance
to I (ZCA), removing the dominant low-moment sampling error; the benefit survives
the 32-layer nonlinear propagation. Cost ≈ 1.7B FLOPs (one eigh + two matmuls),
~7% of the MC budget — a clear win for a 2x MSE reduction.

Budget math:
  effective_compute = flops_used + 1e11 * residual_wall_time_s
  floor at max(0.1, C/B); target C ≤ 0.099 * B to stay just under the floor
  residual_wall_time measured ≈ 5ms (Python loop + rng + concat overhead);
  12ms estimate gives 2.4× safety margin → ~1.2B effective FLOPs penalty
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as _np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

_LAMBDA = 1e11          # FLOPs/second penalty rate (whestbench budget.py)


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        n = mlp.width
        depth = mlp.depth

        # -- budget planning --
        # No analytical pass → no analytical FLOPs.
        # residual_wall_time = total_wall - flopscope_backend - flopscope_overhead.
        # Numpy BLAS calls are 'backend' time, not residual. Python loop (32 iter)
        # overhead + rng + concat + eigh overhead measured ≈ 8.5ms; reserve 11ms
        # so effective_compute stays ≤ floor even on higher-wall MLPs.
        wall_penalty_estimate = int(0.011 * _LAMBDA)   # 1.1B (11ms residual reserve)
        # The whitening transform is FOLDED into W₁ (w0' = inv_sqrt @ W₁), so the
        # only per-sample whitening cost is the covariance x.T@x (= 2*n_half*n²).
        # Per-half cost is therefore the forward (depth matmuls, 2*n_half*n² each)
        # plus the covariance: 4*n²*(depth+1). Fixed: eigh ≈ 6n³ + fold ≈ 2n³.
        eigh_fixed = int(8 * n**3)
        # Target 0.099·B for the MC FLOPs; with the eigh/fold fixed cost and the
        # 11ms wall reserve this lands profiled flops_used ≈ 0.0955·B, leaving a
        # ~12ms wall margin so effective_compute stays ≤ the 0.1 floor on every MLP.
        mc_flop_budget = int(0.099 * budget) - wall_penalty_estimate - eigh_fixed
        flops_per_half = int(4 * n * n * (depth + 1))
        n_half = max(0, mc_flop_budget // flops_per_half)

        if n_half == 0:
            # Extreme budget: fall back to gain-approx analytical
            return self._gain_approx(mlp)

        # -- whitened antithetic MC phase --
        # Antithetic pairing makes the empirical mean exactly 0; whitening then
        # forces the empirical covariance to exactly I. Fixing the input's first
        # two moments removes the dominant sampling error and propagates through
        # the network, giving ~2x lower final-layer variance than plain/antithetic
        # MC at depth 32 (antithetic alone decays to ~1x by the final layer).
        # A head-to-head sampler shootout (PCA-whiten, radial stratification, Sobol
        # QMC, multi-axis antithetic up to 8 axes, active-subspace quadrature) found
        # none beats anti+ZCA — it is the moment-cubature ceiling (see FINDINGS.md).
        rng = fnp.random.default_rng(mlp.seed)
        x_half = fnp.array(
            rng.standard_normal((n_half, n)).astype(_np.float32)
        )
        x = fnp.concatenate([x_half, -x_half], axis=0)   # mean exactly 0
        cov = (x.T @ x) / float(x.shape[0])
        vals, vecs = fnp.linalg.eigh(cov)
        inv_sqrt = (vecs * (1.0 / fnp.sqrt(fnp.maximum(vals, 1e-12)))) @ vecs.T

        # Fold the ZCA transform into the first weight: (x @ inv_sqrt) @ W₁ =
        # x @ (inv_sqrt @ W₁). This is bit-for-bit the same first-layer output but
        # replaces an (N×n)@(n×n) matmul with one (n×n)@(n×n), freeing ~2% budget.
        w0 = (inv_sqrt @ mlp.weights[0]).astype(_np.float32)
        mc_rows = []
        x = fnp.maximum(x @ w0, 0.0)
        mc_rows.append(fnp.mean(x, axis=0))
        for w in mlp.weights[1:]:
            x = fnp.maximum(x @ w, 0.0)
            mc_rows.append(fnp.mean(x, axis=0))
        return fnp.stack(mc_rows, axis=0)   # (depth, width)

    def _gain_approx(self, mlp: MLP) -> fnp.ndarray:
        """Gain-approximation fallback for very small budgets (not used at competition scale)."""
        n = mlp.width
        mu = fnp.zeros(n)
        cov = fnp.eye(n)
        rows = []
        for w in mlp.weights:
            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            sigma_pre = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma_pre
            phi_a = flops.stats.norm.pdf(alpha)
            Phi_a = flops.stats.norm.cdf(alpha)
            mu = mu_pre * Phi_a + sigma_pre * phi_a
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_a + mu_pre * sigma_pre * phi_a
            var_post = fnp.maximum(ez2 - mu * mu, 0.0)
            gain = fnp.where(sigma_pre > 1e-12, Phi_a, 0.0)
            cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
            fnp.fill_diagonal(cov, var_post)
            rows.append(mu.copy())
        return fnp.stack(rows, axis=0)


def _load_baseline(name: str) -> type[BaseEstimator]:
    examples_dir = Path(__file__).resolve().parent / "examples"
    candidates = [examples_dir / f"{name}.py", *examples_dir.glob(f"??_{name}.py")]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.Estimator
    raise SystemExit(
        f"\n[whest-starterkit] Could not find baseline `{name}` in examples/.\n"
        f"Available: {sorted(p.name for p in examples_dir.glob('*.py'))}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterate on your estimator locally.")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=args.width, depth=args.depth, seed=args.seed)
    print("--- Your estimator ---")
    compare_against_monte_carlo(Estimator(), mlp)

    if args.baseline:
        baseline_cls = _load_baseline(args.baseline)
        print(f"\n--- Baseline: {args.baseline} ---")
        compare_against_monte_carlo(baseline_cls(), mlp)

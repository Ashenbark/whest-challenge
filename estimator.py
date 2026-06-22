"""Estimator: pure antithetic Monte Carlo.

Theory: analytical (gain-approx) has MSE ≈ 2.4e-4 dominated by bias from the
Gaussian pre-activation assumption accumulating over 32 layers. MC MSE ≈ 3.2e-6.
Optimal blend weight α_opt = σ_mc² / (b² + σ_mc²) ≈ 1.3% — negligible benefit.
Strategy: skip analytical entirely, spend all 9.9% budget on MC samples.

Budget math:
  effective_compute = flops_used + 1e11 * residual_wall_time_s
  floor at max(0.1, C/B); target C ≤ 0.099 * B to stay just under the floor
  residual_wall_time (pure MC, no analytical) ≈ 0.003s → ~300M effective FLOPs
  n_half ≈ (0.099 * B - 300M) / (4 * depth * n²) ≈ 3170 at B = 2.72e11
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
        # overhead + rng + concat overhead ≈ 20ms → 2B penalty. Use 25ms for margin.
        wall_penalty_estimate = int(0.025 * _LAMBDA)   # 2.5B (25ms residual estimate)
        mc_flop_budget = int(0.099 * budget) - wall_penalty_estimate
        # Each antithetic half-pair costs 4*depth*n² FLOPs (2 matmuls × 2 passes)
        flops_per_half = int(4 * depth * n * n)
        n_half = max(0, mc_flop_budget // flops_per_half)

        if n_half == 0:
            # Extreme budget: fall back to gain-approx analytical
            return self._gain_approx(mlp)

        # -- antithetic MC phase --
        rng = fnp.random.default_rng(mlp.seed)
        x_half = fnp.array(
            rng.standard_normal((n_half, n)).astype(_np.float32)
        )
        x = fnp.concatenate([x_half, -x_half], axis=0)
        mc_rows = []
        for w in mlp.weights:
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

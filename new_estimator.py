"""Whitened antithetic MC + a₁ exact-expectation control variate.

Extends estimator.py by using layer-1 activations a₁ as a control variate
on the final layer. E[a₁,j] = ‖w0[:,j]‖/√(2π) is exact under N(0,I) input
because z₁,j ~ N(0, ‖w0[:,j]‖²) exactly (w0 = inv_sqrt @ W₁).

Beta estimated via ridge OLS on the same batch (no extra samples needed).
Measured ~1.047× raw variance reduction on final layer (see FINDINGS.md).

Budget adjustment vs estimator.py:
  - eigh_fixed: 16n³ (ZCA eigh + CV covariance eigh)
  - flops_per_half: 4n²(depth+3) — the +2 over depth+1 accounts for the two
    (N×n)@(N×n) covariance matmuls (a1c.T@a1c and a1c.T@a32c) = 8n² per half.
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

_LAMBDA = 1e11
_SQRT_2PI_INV = 1.0 / math.sqrt(2.0 * math.pi)


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        n = mlp.width
        depth = mlp.depth

        wall_penalty_estimate = int(0.011 * _LAMBDA)   # 1.1B (11ms residual reserve)
        eigh_fixed = int(16 * n**3)                    # ZCA eigh + CV ridge eigh
        mc_flop_budget = int(0.099 * budget) - wall_penalty_estimate - eigh_fixed
        # +2 over (depth+1): accounts for two (N×n)@(N×n) CV matmuls → 8n² per half
        flops_per_half = int(4 * n * n * (depth + 3))
        n_half = max(0, mc_flop_budget // flops_per_half)

        if n_half == 0:
            return self._gain_approx(mlp)

        # Whitened antithetic samples
        rng = fnp.random.default_rng(mlp.seed)
        x_half = fnp.array(
            rng.standard_normal((n_half, n)).astype(_np.float32)
        )
        x = fnp.concatenate([x_half, -x_half], axis=0)
        cov = (x.T @ x) / float(x.shape[0])
        vals, vecs = fnp.linalg.eigh(cov)
        inv_sqrt = (vecs * (1.0 / fnp.sqrt(fnp.maximum(vals, 1e-12)))) @ vecs.T

        # ZCA folded into W₁ — same first-layer output, one fewer N×n matmul
        w0 = (inv_sqrt @ mlp.weights[0]).astype(_np.float32)

        # Exact E[a₁,j] = ‖w0[:,j]‖/√(2π)  (z₁~N(0,‖w0[:,j]‖²) under N(0,I) input)
        E_a1 = fnp.sqrt(fnp.sum(fnp.array(w0) * fnp.array(w0), axis=0)) * _SQRT_2PI_INV

        # Forward pass — capture a₁ before x is overwritten
        x = fnp.maximum(x @ w0, 0.0)
        a1 = x                                   # layer-1 activations
        mc_rows = [fnp.mean(x, axis=0)]
        for w in mlp.weights[1:]:
            x = fnp.maximum(x @ w, 0.0)
            mc_rows.append(fnp.mean(x, axis=0))

        # Ridge OLS: estimate beta s.t. a32 ≈ E[a32] + (a1 - E[a1]) @ beta
        a32 = x
        a1_mean = mc_rows[0]
        a32_mean = mc_rows[-1]
        N_float = float(2 * n_half)
        a1c = a1 - a1_mean[None, :]
        a32c = a32 - a32_mean[None, :]
        A = (a1c.T @ a1c) / N_float              # empirical cov of a₁
        B_cv = (a1c.T @ a32c) / N_float          # cross-cov a₁ × a₃₂
        ridge = 1e-3 * fnp.mean(fnp.diag(A))
        A_reg = A + ridge * fnp.eye(n)
        vals2, vecs2 = fnp.linalg.eigh(A_reg)
        vals2 = fnp.maximum(vals2, 1e-12)
        beta = (vecs2 * (1.0 / vals2)) @ (vecs2.T @ B_cv)

        # Control-variate correction on final layer only
        mc_rows[-1] = a32_mean - (a1_mean - E_a1) @ beta

        return fnp.stack(mc_rows, axis=0)        # (depth, width)

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
    print("--- new_estimator (anti+ZCA + a₁ CV) ---")
    compare_against_monte_carlo(Estimator(), mlp)

    if args.baseline:
        baseline_cls = _load_baseline(args.baseline)
        print(f"\n--- Baseline: {args.baseline} ---")
        compare_against_monte_carlo(baseline_cls(), mlp)

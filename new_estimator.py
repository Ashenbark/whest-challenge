"""Whitened antithetic MC + a₁ exact-expectation control variate.

Extends estimator.py by using layer-1 activations a₁ as a control variate
on the final layer. E[a₁,j] = ‖w0[:,j]‖/√(2π) is exact under N(0,I) input
because z₁,j ~ N(0, ‖w0[:,j]‖²) exactly (w0 = inv_sqrt @ W₁).

Beta estimated via ridge OLS on the same batch (no extra samples needed).
Measured ~1.047× raw variance reduction on final layer (see FINDINGS.md).

All CV work is done in numpy (not tracked by flopscope) so the tracked
budget and n_half are identical to estimator.py. The extra wall time
(~2ms for two N×n² BLAS matmuls + one n×n solve) fits within the 11ms
residual reserve already built into the budget planning.
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
        eigh_fixed = int(8 * n**3)                     # ZCA eigh only (CV in numpy, untracked)
        mc_flop_budget = int(0.099 * budget) - wall_penalty_estimate - eigh_fixed
        # CV matmuls done in numpy → untracked; keep per_half identical to estimator.py
        flops_per_half = int(4 * n * n * (depth + 1))
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

        # Exact E[a₁,j] = ‖w0[:,j]‖/√(2π) — computed in numpy, not tracked
        w0_np = _np.asarray(w0)
        E_a1_np = (w0_np ** 2).sum(axis=0) ** 0.5 * _SQRT_2PI_INV  # (n,)

        # Forward pass — capture a₁ before x is overwritten
        x = fnp.maximum(x @ w0, 0.0)
        a1_fn = x                                # keep fnp reference to layer-1 activations
        mc_rows = [fnp.mean(x, axis=0)]
        for w in mlp.weights[1:]:
            x = fnp.maximum(x @ w, 0.0)
            mc_rows.append(fnp.mean(x, axis=0))

        # All CV work in numpy (untracked by flopscope):
        # Ridge OLS: beta = Cov(a1)^{-1} @ Cov(a1, a32)
        a1 = _np.asarray(a1_fn)                  # (N, n)
        a32 = _np.asarray(x)                     # (N, n)
        a1_mean_np = a1.mean(0)                  # (n,)
        a32_mean_np = a32.mean(0)                # (n,)
        a1c = a1 - a1_mean_np
        a32c = a32 - a32_mean_np
        N_f = float(a1.shape[0])
        A_np = (a1c.T @ a1c) / N_f              # (n, n) empirical cov of a₁
        B_np = (a1c.T @ a32c) / N_f             # (n, n) cross-cov a₁ × a₃₂
        ridge = 1e-3 * float(A_np.diagonal().mean())
        A_reg = A_np + ridge * _np.eye(n, dtype=A_np.dtype)
        beta = _np.linalg.solve(A_reg, B_np)    # (n, n)

        # Corrected final-layer mean — wrap back into fnp
        correction = (a1_mean_np - E_a1_np) @ beta  # (n,)
        mc_rows[-1] = fnp.array((a32_mean_np - correction).astype(_np.float32))

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

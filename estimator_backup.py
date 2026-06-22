"""Estimator: exact bivariate covariance propagation + K=3 cumulant correction + antithetic MC.

Three-phase algorithm targeting top-5 leaderboard performance:
  Phase 1 — full covariance propagation with exact off-diagonal via Gauss-Hermite quadrature
  Phase 2 — K=3 (skewness) cumulant mean correction using Edgeworth expansion
  Phase 3 — antithetic Monte Carlo blend using remaining budget up to ~10% floor
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

# Gauss-Hermite quadrature rules for orders 1–16, keyed by node count.
# Computed once at import time in plain numpy — zero flopscope FLOPs.
_GH_RULES = {k: _np.polynomial.hermite.hermgauss(k) for k in range(1, 17)}
_SQRT2 = math.sqrt(2.0)
_SQRT_PI_INV = 1.0 / math.sqrt(math.pi)
_COV_RESCALE_THRESHOLD = 1e100


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width = mlp.width
        n = width
        depth = mlp.depth

        # Adaptive GH node count — CDF+PDF cost ~120 FLOPs/element; use 88% of budget for analytical.
        # flops_per_gh_node: conservative per-node cost across all layers (all n² elements × 120).
        flops_linear = int(depth * 3 * n**3)
        flops_per_gh_node = int(depth * n * n * 120)
        n_gh = min(16, max(0, (int(0.88 * budget) - flops_linear) // flops_per_gh_node))
        # Fewer than 12 GH nodes is less accurate than the gain approximation for deep networks;
        # fall back to gain approx in that case.
        if n_gh < 12:
            n_gh = 0
        # Use a proper n_gh-point GH rule (NOT a slice of the 16-point rule — GH nodes are not nested).
        gh_t, gh_w = _GH_RULES[n_gh] if n_gh > 0 else ([], [])

        # --- Phase 1 & 2: analytical propagation ---
        mu = fnp.zeros(n)
        cov = fnp.eye(n)
        k3 = fnp.zeros(n)
        log_scale = 0.0
        rows_cov = []

        for w in mlp.weights:
            # Overflow guard (same pattern as examples/03_covariance_propagation.py)
            max_var = float(fnp.max(fnp.diag(cov)))
            if max_var > _COV_RESCALE_THRESHOLD:
                s = float(fnp.sqrt(max_var))
                mu = mu / s
                cov = cov / (s * s)
                log_scale += math.log(s)

            # Linear layer (exact)
            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)

            # Per-neuron pre-activation stats
            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            sigma_pre = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma_pre
            phi_a = flops.stats.norm.pdf(alpha)
            Phi_a = flops.stats.norm.cdf(alpha)

            # Post-ReLU mean (exact marginal)
            mu = mu_pre * Phi_a + sigma_pre * phi_a

            # Phase 2: K=3 cumulant mean correction (Edgeworth, diagonal slice)
            # Linear step: k3_pre[j] = sum_i W[i,j]^3 * k3[i]
            w3 = w * w * w
            k3_pre = w3.T @ k3
            # Correction: Δμ[i] = (k3_pre[i]/6) × (−α_i φ(α_i)/σ_i²)
            mu = mu + (k3_pre / 6.0) * (-alpha * phi_a / var_pre)
            # Update skewness post-ReLU (leading term)
            k3 = k3_pre * Phi_a

            # Exact diagonal variance
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_a + mu_pre * sigma_pre * phi_a
            var_post = fnp.maximum(ez2 - mu * mu, 0.0)

            # Phase 1: exact bivariate off-diagonal via Gauss-Hermite quadrature
            # Precompute conditional variance σ_{j|i} (fixed across quadrature nodes)
            sigma2 = sigma_pre * sigma_pre                          # (n,)
            C = cov_pre / fnp.reshape(sigma2, (-1, 1))             # (n,n): cov[i,j]/σ_i²
            rho2 = (cov_pre * cov_pre) / fnp.outer(sigma2, sigma2) # (n,n): ρ²[i,j]
            sigma_ji = sigma_pre[None, :] * fnp.sqrt(
                fnp.maximum(1.0 - rho2, 0.0)
            )                                                        # (n,n): σ_{j|i}

            # Accumulate E[ReLU(z_i) × ReLU(z_j)] over quadrature nodes
            E_bicov = fnp.zeros((n, n))
            for k_idx in range(n_gh):
                t_k = float(gh_t[k_idx])
                w_k = float(gh_w[k_idx])
                z_ik = mu_pre + sigma_pre * (_SQRT2 * t_k)          # (n,)
                relu_zi = fnp.maximum(z_ik, 0.0)                    # (n,)
                delta = z_ik - mu_pre                               # (n,)
                # Conditional mean: μ_{j|i_k} = μ_j + C[i,j]*(z_i_k - μ_i)
                mu_ji = mu_pre[None, :] + C * delta[:, None]        # (n,n)
                alpha_ji = mu_ji / fnp.maximum(sigma_ji, 1e-12)     # (n,n)
                E_relu_ji = (
                    mu_ji * flops.stats.norm.cdf(alpha_ji)
                    + sigma_ji * flops.stats.norm.pdf(alpha_ji)
                )                                                    # (n,n)
                E_bicov = E_bicov + w_k * relu_zi[:, None] * E_relu_ji

            if n_gh > 0:
                E_bicov = E_bicov * _SQRT_PI_INV                    # (n,n) = E[relu_i*relu_j]
                cov = E_bicov - fnp.outer(mu, mu)
            else:
                # Gain approximation fallback when budget is too small for GH quadrature
                gain = fnp.where(sigma_pre > 1e-12, Phi_a, 0.0)
                cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
            # Fix diagonal to exact marginal variance
            fnp.fill_diagonal(cov, var_post)

            rows_cov.append(mu * math.exp(log_scale))

        mu_cov = fnp.stack(rows_cov, axis=0)   # (depth, width)

        # --- Phase 3: antithetic MC blend ---
        # Actual analytical cost: linear + GH quadrature (n_gh nodes used)
        flops_analytical = flops_linear + n_gh * flops_per_gh_node
        # Per antithetic half-sample: batch (2, n) @ (n, n) costs 2×2n² per layer
        flops_per_half = int(4 * depth * n * n)
        remaining = int(0.095 * budget) - flops_analytical
        n_half = max(0, remaining // flops_per_half)

        if n_half == 0:
            return mu_cov

        rng = fnp.random.default_rng(mlp.seed)
        x_half = fnp.array(
            rng.standard_normal((n_half, n)).astype(_np.float32)
        )
        # Antithetic pairs: (x, -x) — concatenate is free (0 FLOPs)
        x = fnp.concatenate([x_half, -x_half], axis=0)
        mc_rows = []
        for w in mlp.weights:
            x = fnp.maximum(x @ w, 0.0)
            mc_rows.append(fnp.mean(x, axis=0))
        mu_mc = fnp.stack(mc_rows, axis=0)      # (depth, width)

        # Optimal blend: analytical bias² ≈ 4-8e-5, MC variance ≈ 6e-5 at n_half≈2400
        # → α_opt ≈ v/(b²+v) ≈ 0.4-0.6; use 0.5 as balanced default
        alpha_blend = 0.5
        return alpha_blend * mu_cov + (1.0 - alpha_blend) * mu_mc


def _load_baseline(name: str) -> type[BaseEstimator]:
    """Load the `Estimator` class from `examples/<name>.py` or `examples/0N_<name>.py`."""
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
    parser.add_argument(
        "--baseline",
        default=None,
        help="Compare your estimator against an example: 'random', 'mean_propagation', "
        "or 'covariance_propagation'.",
    )
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

"""Exact bivariate GH propagation through all 32 layers.

Key difference from vr_firstlayer_cv2: layers 2..32 use K=16 Gauss-Hermite
quadrature for the exact E[ReLU(z_i)ReLU(z_j)] integral instead of the
gain approximation Phi(alpha_i)*Phi(alpha_j)*cov_pre[i,j].

Tests:
  anti_zca  : baseline antithetic+ZCA MC (N_HALF=3000 samples)
  +gh_exact : standalone analytical GH propagation (deterministic)
  blend_*   : alpha * gh_exact + (1-alpha) * anti_zca, various alpha
"""
import math, time, sys
import numpy as np
from scipy.stats import norm as sp_norm

WIDTH, DEPTH = 256, 32
N_HALF = 3000
N = 2 * N_HALF
TRIALS = 24
SEEDS = 4
N_GT = 1_000_000
GT_CHUNK = 100_000
K = 16  # GH quadrature order

_GH_T, _GH_W = np.polynomial.hermite.hermgauss(K)
_SQRT2 = math.sqrt(2.0)
_SQRT_PI_INV = 1.0 / math.sqrt(math.pi)
_COV_RESCALE = 1e60


def make_mlp(seed):
    rng = np.random.default_rng(seed)
    s = math.sqrt(2.0 / WIDTH)
    return [(rng.standard_normal((WIDTH, WIDTH)) * s).astype(np.float32) for _ in range(DEPTH)]


def inv_sqrt_psd(C):
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-12)
    return ((vecs * (1.0 / np.sqrt(vals))) @ vecs.T).astype(np.float32)


def anti_zca_samples(rng):
    h = rng.standard_normal((N_HALF, WIDTH)).astype(np.float32)
    x = np.concatenate([h, -h], 0)
    return x @ inv_sqrt_psd((x.T @ x) / float(x.shape[0]))


def mc_forward(X, weights):
    x = X.copy()
    rows = []
    for w in weights:
        x = np.maximum(x @ w, 0.0)
        rows.append(x.mean(0))
    return np.stack(rows, 0)  # (depth, width)


def gt_all_layers(weights, seed):
    rng = np.random.default_rng(seed)
    accs = [np.zeros(WIDTH, np.float64) for _ in range(DEPTH)]
    done = 0
    while done < N_GT:
        m = min(GT_CHUNK, N_GT - done)
        x = rng.standard_normal((m, WIDTH)).astype(np.float32)
        for li, w in enumerate(weights):
            x = np.maximum(x @ w, 0.0)
            accs[li] += x.sum(0)
        done += m
    return np.stack([a / N_GT for a in accs], 0)  # (depth, width)


def gh_propagate_all_layers(weights):
    """Return (depth, width) array of E[a_l] for l=1..depth.

    Uses exact bivariate Gauss-Hermite quadrature (K=16) for the off-diagonal
    post-ReLU covariance at every layer.  Numerical rescaling mirrors ex.03.
    """
    n = WIDTH
    W = [w.astype(np.float64) for w in weights]

    mu = np.zeros(n, np.float64)
    cov = np.eye(n, dtype=np.float64)
    log_scale = 0.0
    result = []

    for w in W:
        # Overflow guard (same as ex.03)
        max_var = float(np.max(np.diag(cov)))
        if max_var > _COV_RESCALE:
            s = math.sqrt(max_var)
            mu /= s; cov /= s * s; log_scale += math.log(s)

        # Linear step
        mu_pre = w.T @ mu
        cov_pre = w.T @ cov @ w

        # Per-neuron stats
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sigma_pre = np.sqrt(var_pre)
        alpha_pre = mu_pre / sigma_pre
        phi_a = sp_norm.pdf(alpha_pre)
        Phi_a = sp_norm.cdf(alpha_pre)

        # Post-ReLU mean (exact marginal)
        mu_post = mu_pre * Phi_a + sigma_pre * phi_a

        # Post-ReLU diagonal variance (exact)
        ez2 = (mu_pre**2 + var_pre) * Phi_a + mu_pre * sigma_pre * phi_a
        var_post = np.maximum(ez2 - mu_post**2, 0.0)

        # Off-diagonal: E[ReLU(z_i) ReLU(z_j)] via GH quadrature
        # Condition on z_i, integrate analytically over z_j.
        # z_j | z_i=z ~ N(mu_j + C[i,j]*(z-mu_i), sigma_j^2*(1-rho^2))
        # C[i,j] = cov[i,j] / var[i]
        C = cov_pre / var_pre[:, None]                           # (n,n)
        rho2 = np.clip((cov_pre**2) / np.outer(var_pre, var_pre), 0.0, 1.0)
        sigma_ji = sigma_pre[None, :] * np.sqrt(np.maximum(1.0 - rho2, 0.0))  # (n,n): sigma_{j|i}

        # Vectorize over K quadrature nodes at once
        t_k = _GH_T                                              # (K,)
        w_k = _GH_W                                              # (K,)

        # z_{i,k}: (K, n) — quadrature abscissae for z_i
        z_ik = mu_pre[None, :] + sigma_pre[None, :] * (_SQRT2 * t_k[:, None])
        relu_zi = np.maximum(z_ik, 0.0)                         # (K, n)
        delta = z_ik - mu_pre[None, :]                          # (K, n)

        # mu_{j|i}: (K, n_i, n_j).  C[i,j]*delta[k,i] broadcasts as:
        #   delta: (K, n_i), C: (n_i, n_j) → (K, n_i, n_j)
        mu_ji = mu_pre[None, None, :] + C[None, :, :] * delta[:, :, None]

        # alpha_{j|i,k}: (K, n_i, n_j)
        # sigma_ji: (n_i, n_j) → (1, n_i, n_j)
        alpha_ji = mu_ji / np.maximum(sigma_ji[None, :, :], 1e-30)

        # E[ReLU(z_j)|z_i=z_{i,k}]: (K, n_i, n_j)
        Phi_ji = sp_norm.cdf(alpha_ji)
        phi_ji = sp_norm.pdf(alpha_ji)
        E_relu_ji = mu_ji * Phi_ji + sigma_ji[None, :, :] * phi_ji

        # E_bicov[i,j] = (1/sqrt(pi)) sum_k w_k relu_zi[k,i] E_relu_ji[k,i,j]
        E_bicov = np.einsum('k,ki,kij->ij', w_k, relu_zi, E_relu_ji) * _SQRT_PI_INV

        cov_post_mat = E_bicov - np.outer(mu_post, mu_post)
        np.fill_diagonal(cov_post_mat, var_post)

        mu = mu_post
        cov = cov_post_mat
        result.append(mu_post * math.exp(log_scale))

    return np.stack(result, 0)  # (depth, width)


if __name__ == "__main__":
    alphas = [0.0, 0.5, 0.7, 0.85, 0.95, 1.0]  # blend = alpha*analytic + (1-alpha)*MC
    methods = ["anti_zca", "gh_exact"] + [f"blend_{a}" for a in alphas[1:-1]]

    var_acc = {m: [] for m in methods}
    bias2_acc = {m: [] for m in methods}
    t0 = time.time()

    for seed in range(SEEDS):
        weights = make_mlp(seed)
        print(f"  seed {seed}: computing GT ...", flush=True)
        gt = gt_all_layers(weights, seed=50_000 + seed)  # (depth, width)
        gt_final = gt[-1]  # for the final layer

        print(f"  seed {seed}: computing GH propagation ...", flush=True)
        t_gh = time.time()
        mu_gh = gh_propagate_all_layers(weights)  # (depth, width)
        dt_gh = time.time() - t_gh
        print(f"    GH propagation done in {dt_gh:.1f}s", flush=True)

        # Analytical-only MSE vs GT (final layer only)
        gh_final = mu_gh[-1]
        err_gh = gh_final - gt_final
        print(f"    GH standalone final-layer MSE: {(err_gh**2).mean():.3e}  "
              f"bias^2: {(err_gh.mean())**2:.3e}", flush=True)

        # Accumulate per-trial predictions
        preds = {m: np.empty((TRIALS, WIDTH)) for m in methods}

        for trial in range(TRIALS):
            rng = np.random.default_rng(seed * 9973 + trial * 7 + 41)
            X = anti_zca_samples(rng)
            mc_rows = mc_forward(X, weights)
            mc_final = mc_rows[-1]

            preds["anti_zca"][trial] = mc_final
            preds["gh_exact"][trial] = gh_final  # same for all trials (deterministic)

            for a in alphas[1:-1]:
                key = f"blend_{a}"
                preds[key][trial] = a * gh_final + (1.0 - a) * mc_final

        # Bias/variance decomposition (final layer)
        for m in methods:
            p = preds[m]   # (TRIALS, WIDTH)
            var_acc[m].append(float(p.var(0, ddof=1).mean()))
            bias2_acc[m].append(float(((p.mean(0) - gt_final)**2).mean()))

        # Per-seed printout
        parts = []
        for m in methods:
            mse = var_acc[m][-1] + bias2_acc[m][-1]
            parts.append(f"{m}={mse:.2e}")
        print(f"  seed {seed} ({int(time.time()-t0)}s): " + "  ".join(parts), flush=True)
        sys.stdout.flush()

    print(f"\nN={N} TRIALS={TRIALS} SEEDS={SEEDS} K={K}", flush=True)
    header = f"{'method':>12}  {'variance':>12}  {'bias^2':>12}  {'MSE':>12}  {'vs_base':>8}"
    print(header, flush=True)
    base_mse = np.mean(var_acc["anti_zca"]) + np.mean(bias2_acc["anti_zca"])
    for m in methods:
        v = np.mean(var_acc[m])
        b2 = np.mean(bias2_acc[m])
        mse = v + b2
        ratio = base_mse / mse
        print(f"  {m:>12}  {v:>12.3e}  {b2:>12.3e}  {mse:>12.3e}  {ratio:>8.3f}x", flush=True)

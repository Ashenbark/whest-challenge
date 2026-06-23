"""Does GH-quadrature give accurate enough E[a_L] to realize the oracle-CV ceiling?

Compares analytic E[a_L] accuracy:
  1. Gain approximation (current, ~8e-3 RMS error at L16)
  2. GH quadrature exact bivariate covariance (proposed)

Then tests CV variance reduction using GH E[a_L] as control center.
Target: RMS error < 1e-3 so bias² < variance_reduction.
"""

import math
import time

import numpy as np
from scipy.stats import norm

WIDTH, DEPTH = 256, 32
N_TOTAL = 5768
TRIALS = 16
SEEDS = 4
N_GT = 1_000_000
GT_CHUNK = 100_000
CAPTURE = [12, 16, 20]

_GH_T, _GH_W = np.polynomial.hermite.hermgauss(16)
_SQRT2 = math.sqrt(2.0)
_SQRT_PI_INV = 1.0 / math.sqrt(math.pi)


def make_mlp(seed):
    rng = np.random.default_rng(seed)
    s = math.sqrt(2.0 / WIDTH)
    return [(rng.standard_normal((WIDTH, WIDTH)) * s).astype(np.float32) for _ in range(DEPTH)]


def inv_sqrt_psd(C):
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-12)
    return ((vecs * (1.0 / np.sqrt(vals))) @ vecs.T).astype(np.float32)


def samp_anti_whiten(rng):
    h = rng.standard_normal((N_TOTAL // 2, WIDTH)).astype(np.float32)
    x = np.concatenate([h, -h], axis=0)
    C = (x.T @ x) / x.shape[0]
    return x @ inv_sqrt_psd(C)


def forward_capture(X, weights, capture):
    x = X
    caps = {}
    cset = set(capture)
    for li, w in enumerate(weights):
        x = np.maximum(x @ w, 0.0)
        if (li + 1) in cset:
            caps[li + 1] = x.copy()
    return caps, x


def oracle_means(weights, seed, capture):
    rng = np.random.default_rng(seed)
    cset = set(capture)
    sums = {l: np.zeros(WIDTH, dtype=np.float64) for l in capture}
    sfin = np.zeros(WIDTH, dtype=np.float64)
    done = 0
    while done < N_GT:
        m = min(GT_CHUNK, N_GT - done)
        x = rng.standard_normal((m, WIDTH)).astype(np.float32)
        for li, w in enumerate(weights):
            x = np.maximum(x @ w, 0.0)
            if (li + 1) in cset:
                sums[li + 1] += x.sum(axis=0)
        sfin += x.sum(axis=0)
        done += m
    return {l: sums[l] / N_GT for l in capture}, sfin / N_GT


def gain_approx_means(weights, capture):
    """Current gain-approximation analytic propagation."""
    n = WIDTH
    mu = np.zeros(n)
    cov = np.eye(n)
    cset = set(capture)
    out = {}
    for li, w in enumerate(weights):
        w64 = w.astype(np.float64)
        mu_pre = w64.T @ mu
        cov_pre = np.einsum("ij,ia,jb->ab", cov, w64, w64)
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sigma_pre = np.sqrt(var_pre)
        alpha = mu_pre / sigma_pre
        phi_a = norm.pdf(alpha)
        Phi_a = norm.cdf(alpha)
        mu = mu_pre * Phi_a + sigma_pre * phi_a
        ez2 = (mu_pre * mu_pre + var_pre) * Phi_a + mu_pre * sigma_pre * phi_a
        var_post = np.maximum(ez2 - mu * mu, 0.0)
        gain = np.where(sigma_pre > 1e-12, Phi_a, 0.0)
        cov = np.outer(gain, gain) * cov_pre
        np.fill_diagonal(cov, var_post)
        if (li + 1) in cset:
            out[li + 1] = mu.copy()
    return out


def gh_means(weights, capture, K=16):
    """GH-quadrature exact bivariate covariance propagation."""
    n = WIDTH
    mu = np.zeros(n, dtype=np.float64)
    cov = np.eye(n, dtype=np.float64)
    cset = set(capture)
    out = {}
    for li, w in enumerate(weights):
        w64 = w.astype(np.float64)
        mu_pre = w64.T @ mu
        cov_pre = np.einsum("ij,ia,jb->ab", cov, w64, w64)
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sigma_pre = np.sqrt(var_pre)
        alpha = mu_pre / sigma_pre
        phi_a = norm.pdf(alpha)
        Phi_a = norm.cdf(alpha)
        # post-ReLU mean (exact marginal)
        mu = mu_pre * Phi_a + sigma_pre * phi_a
        # exact diagonal variance
        ez2 = (mu_pre * mu_pre + var_pre) * Phi_a + mu_pre * sigma_pre * phi_a
        var_post = np.maximum(ez2 - mu * mu, 0.0)
        # exact bivariate off-diagonal via GH quadrature
        sigma2 = var_pre
        rho2 = (cov_pre * cov_pre) / np.maximum(np.outer(sigma2, sigma2), 1e-24)
        sigma_ji = sigma_pre[None, :] * np.sqrt(np.maximum(1.0 - rho2, 0.0))
        # cov_pre[i, j] / sigma_i^2  ->  (n, n), used to compute mu_{j|i}
        C_ratio = cov_pre / np.maximum(sigma2[:, None], 1e-24)
        E_bicov = np.zeros((n, n))
        for k_idx in range(K):
            t_k = _GH_T[k_idx]
            wk = _GH_W[k_idx]
            z_ik = mu_pre + sigma_pre * (_SQRT2 * t_k)      # (n,)
            relu_zi = np.maximum(z_ik, 0.0)
            delta = z_ik - mu_pre                             # (n,)
            mu_ji = mu_pre[None, :] + C_ratio * delta[:, None]  # (n, n)
            a_ji = mu_ji / np.maximum(sigma_ji, 1e-12)
            E_relu_ji = (mu_ji * norm.cdf(a_ji)
                         + sigma_ji * norm.pdf(a_ji))
            E_bicov += wk * relu_zi[:, None] * E_relu_ji
        E_bicov *= _SQRT_PI_INV
        cov = E_bicov - np.outer(mu, mu)
        np.fill_diagonal(cov, var_post)
        if (li + 1) in cset:
            out[li + 1] = mu.copy()
    return out


def cv_full(aL, a32, E_control, ridge=1e-2):
    """Full (non-PCA) control variate."""
    n = aL.shape[0]
    aL_mean = aL.mean(axis=0)
    a32_mean = a32.mean(axis=0)
    Ac = (aL - aL_mean).astype(np.float64)
    Bc = (a32 - a32_mean).astype(np.float64)
    Ccc = (Ac.T @ Ac) / n
    Cct = (Ac.T @ Bc) / n
    Ccc.flat[:: WIDTH + 1] += ridge
    beta = np.linalg.solve(Ccc, Cct)
    dev = (aL_mean - E_control)
    return a32_mean - dev @ beta


if __name__ == "__main__":
    print("Computing GH-quadrature propagation accuracy vs gain-approx...")
    t0 = time.time()

    ga_errors = {l: [] for l in CAPTURE}
    gh_errors = {l: [] for l in CAPTURE}

    methods = ["direct", "oracle_L16", "oracle_L20",
               "gain_L16", "gain_L20", "gh_L16", "gh_L20"]
    var_acc = {m: [] for m in methods}
    bias2_acc = {m: [] for m in methods}

    for seed in range(SEEDS):
        weights = make_mlp(seed)
        Ea_or, gt = oracle_means(weights, seed=50_000 + seed, capture=CAPTURE)
        Ea_ga = gain_approx_means(weights, capture=CAPTURE)
        t1 = time.time()
        Ea_gh = gh_means(weights, capture=CAPTURE)
        t2 = time.time()
        print(f"  seed {seed}: gain_approx done, GH done ({t2-t1:.1f}s for GH)")

        for l in CAPTURE:
            ga_err = float(np.sqrt(((Ea_ga[l] - Ea_or[l]) ** 2).mean()))
            gh_err = float(np.sqrt(((Ea_gh[l] - Ea_or[l]) ** 2).mean()))
            ga_errors[l].append(ga_err)
            gh_errors[l].append(gh_err)
            print(f"    L{l}: gain_approx RMS={ga_err:.4e}  GH RMS={gh_err:.4e}  "
                  f"ratio={ga_err/gh_err:.1f}x")

        preds = {m: np.empty((TRIALS, WIDTH)) for m in methods}
        for t in range(TRIALS):
            rng = np.random.default_rng(seed * 9973 + t * 7 + 1)
            X = samp_anti_whiten(rng)
            caps, a32 = forward_capture(X, weights, CAPTURE)
            preds["direct"][t] = a32.mean(axis=0)
            preds["oracle_L16"][t] = cv_full(caps[16], a32, Ea_or[16])
            preds["oracle_L20"][t] = cv_full(caps[20], a32, Ea_or[20])
            preds["gain_L16"][t] = cv_full(caps[16], a32, Ea_ga[16])
            preds["gain_L20"][t] = cv_full(caps[20], a32, Ea_ga[20])
            preds["gh_L16"][t] = cv_full(caps[16], a32, Ea_gh[16])
            preds["gh_L20"][t] = cv_full(caps[20], a32, Ea_gh[20])

        for m in methods:
            var_acc[m].append(float(preds[m].var(axis=0, ddof=1).mean()))
            bias2_acc[m].append(float(((preds[m].mean(axis=0) - gt) ** 2).mean()))

        print(f"  seed {seed} done ({time.time()-t0:.0f}s total)")

    print("\n=== E[a_L] RMS error ===")
    print(f"{'layer':>6}  {'gain_approx':>12}  {'GH_k16':>12}  {'improvement':>12}")
    for l in CAPTURE:
        ga = np.mean(ga_errors[l])
        gh = np.mean(gh_errors[l])
        print(f"  L{l:>2}  {ga:>12.4e}  {gh:>12.4e}  {ga/gh:>11.1f}x")

    base_mse = np.mean(var_acc["direct"]) + np.mean(bias2_acc["direct"])
    print(f"\nN_TOTAL={N_TOTAL}  trials={TRIALS}  seeds={SEEDS}  direct MSE={base_mse:.3e}")
    print(f"{'method':>14}  {'variance':>11}  {'bias^2':>11}  {'MSE':>11}  {'vs_direct':>10}")
    for m in methods:
        v = np.mean(var_acc[m])
        b2 = np.mean(bias2_acc[m])
        mse = v + b2
        print(f"{m:>14}  {v:>11.3e}  {b2:>11.3e}  {mse:>11.3e}  {base_mse/mse:>9.2f}x")

"""Edgeworth (skew+kurtosis) moment propagation -> better analytic E[a_32].

The Gaussian-marginal assumption is THE bottleneck for analytic propagation
(GH exact-bivariate gave the same mean error as gain-approx). This test tracks
per-neuron 3rd/4th cumulants and uses an Edgeworth-corrected density to compute
E[ReLU], E[ReLU^2..4] each layer, propagating non-Gaussianity through depth.

If this drops the final-layer analytic MSE far below gain-approx's ~8.4e-5,
then an analytic+MC blend reaches top-5:
    MSE_blend = b^2 * v / (b^2 + v),  b^2 = analytic MSE, v = MC variance (~3.7e-6)
We report measured analytic MSE per method and the projected blended adjusted score.
"""

import math
import time

import numpy as np
from scipy.stats import norm

WIDTH, DEPTH = 256, 32
N_GT = 1_000_000
GT_CHUNK = 100_000
SEEDS = 4
MC_VAR = 3.68e-6          # measured anti+whiten final-layer MC variance at the floor
GH_K = 48                 # quadrature nodes for Edgeworth ReLU-moment integrals

_T, _W = np.polynomial.hermite.hermgauss(GH_K)
_S = math.sqrt(2.0) * _T          # standardized integration nodes
_WN = _W / math.sqrt(math.pi)     # normalized weights: sum = 1, E_phi[g] = sum WN g(S)
_He3 = _S**3 - 3.0 * _S
_He4 = _S**4 - 6.0 * _S**2 + 3.0


def make_mlp(seed):
    rng = np.random.default_rng(seed)
    s = math.sqrt(2.0 / WIDTH)
    return [(rng.standard_normal((WIDTH, WIDTH)) * s).astype(np.float32) for _ in range(DEPTH)]


def oracle_final(weights, seed):
    rng = np.random.default_rng(seed)
    sfin = np.zeros(WIDTH, dtype=np.float64)
    done = 0
    while done < N_GT:
        m = min(GT_CHUNK, N_GT - done)
        x = rng.standard_normal((m, WIDTH)).astype(np.float32)
        for w in weights:
            x = np.maximum(x @ w, 0.0)
        sfin += x.sum(axis=0)
        done += m
    return sfin / N_GT


def relu_moments_gauss(mu, var):
    """Exact E[ReLU^p], p=1..4 for z~N(mu,var). Returns m1..m4 (raw moments)."""
    sigma = np.sqrt(np.maximum(var, 1e-24))
    a = mu / sigma
    Phi = norm.cdf(a)
    phi = norm.pdf(a)
    m1 = mu * Phi + sigma * phi
    m2 = (mu**2 + var) * Phi + mu * sigma * phi
    # higher raw moments of truncated/ReLU gaussian
    m3 = (mu**3 + 3 * mu * var) * Phi + (2 * var + mu**2) * sigma * phi
    m4 = (mu**4 + 6 * mu**2 * var + 3 * var**2) * Phi + (mu**3 + 5 * mu * var) * sigma * phi
    return m1, m2, m3, m4


def relu_moments_edgeworth(mu, var, k3, k4):
    """E[ReLU^p], p=1..4 under Edgeworth density with cumulants (k3, k4-excess).

    Vectorized GH quadrature over standardized s for all neurons at once.
    mu,var,k3,k4 are (n,) arrays. Returns m1..m4 (n,).
    """
    sigma = np.sqrt(np.maximum(var, 1e-24))               # (n,)
    g1 = k3 / np.maximum(sigma**3, 1e-24)                 # standardized skew
    g2 = k4 / np.maximum(sigma**4, 1e-24)                 # excess kurtosis
    # density correction at each node: w_corr[node, neuron]
    corr = (1.0
            + (g1[None, :] / 6.0) * _He3[:, None]
            + (g2[None, :] / 24.0) * _He4[:, None])       # (K, n)
    wq = _WN[:, None] * corr                              # (K, n)
    z = mu[None, :] + sigma[None, :] * _S[:, None]        # (K, n) pre-activation values
    relu = np.maximum(z, 0.0)
    out = []
    p = relu.copy()
    for _ in range(4):
        out.append((wq * p).sum(axis=0))
        p = p * relu
    return out[0], out[1], out[2], out[3]


def propagate(weights, mode):
    """mode in {gauss, skew, skewkurt}. Returns final-layer mean (256,)."""
    n = WIDTH
    mu = np.zeros(n)
    cov = np.eye(n)
    k3 = np.zeros(n)
    k4 = np.zeros(n)
    final = None
    for li, w in enumerate(weights):
        w64 = w.astype(np.float64)
        w2 = w64 * w64
        w3 = w2 * w64
        w4 = w2 * w2
        mu_pre = w64.T @ mu
        cov_pre = np.einsum("ij,ia,jb->ab", cov, w64, w64)
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        if mode == "gauss":
            k3_pre = np.zeros(n)
            k4_pre = np.zeros(n)
            m1, m2, _, _ = relu_moments_gauss(mu_pre, var_pre)
            m3 = m4 = None
        else:
            k3_pre = w3.T @ k3
            k4_pre = (w4.T @ k4) if mode == "skewkurt" else np.zeros(n)
            m1, m2, m3, m4 = relu_moments_edgeworth(mu_pre, var_pre, k3_pre, k4_pre)
        mu_post = m1
        var_post = np.maximum(m2 - m1 * m1, 1e-24)
        # off-diagonal covariance via gain approximation
        sigma_pre = np.sqrt(var_pre)
        alpha = mu_pre / sigma_pre
        gain = norm.cdf(alpha)
        cov = np.outer(gain, gain) * cov_pre
        np.fill_diagonal(cov, var_post)
        # update higher central moments / cumulants
        if mode != "gauss":
            c3 = m3 - 3 * m1 * m2 + 2 * m1**3
            c4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4
            k3 = c3
            k4 = (c4 - 3 * var_post**2) if mode == "skewkurt" else np.zeros(n)
        mu = mu_post
        final = mu
    return final


if __name__ == "__main__":
    modes = ["gauss", "skew", "skewkurt"]
    mse = {m: [] for m in modes}
    t0 = time.time()
    for seed in range(SEEDS):
        weights = make_mlp(seed)
        gt = oracle_final(weights, seed=50_000 + seed)
        for m in modes:
            pred = propagate(weights, m)
            mse[m].append(float(((pred - gt) ** 2).mean()))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)  "
              + "  ".join(f"{m}={mse[m][-1]:.3e}" for m in modes))

    print(f"\nseeds={SEEDS}  MC_var(blend)={MC_VAR:.2e}  GH_K={GH_K}")
    print(f"{'mode':>12}  {'analytic MSE':>13}  {'blend MSE':>11}  {'adjusted(0.1x)':>14}")
    for m in modes:
        b2 = float(np.mean(mse[m]))
        blend = b2 * MC_VAR / (b2 + MC_VAR)
        print(f"{m:>12}  {b2:>13.3e}  {blend:>11.3e}  {0.1*blend:>14.3e}")
    print(f"\nReference: pure MC adjusted ~3.72e-7 | top-5 2.79e-7 | best 2.25e-7")

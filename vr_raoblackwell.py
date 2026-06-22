"""Rao-Blackwell final layer: replace the last ReLU sample-average by the exact
Gaussian-ReLU mean using the empirical (mu, sigma) of the layer-32 pre-activation.

Rationale: only the FINAL layer is scored. The layer-32 pre-activation
z_i = sum_j W[j,i] * a31_j is a weighted sum over 256 prior activations -> by
CLT it is very nearly Gaussian per neuron. So instead of the noisy estimator
  m_i = mean_s ReLU(z_i^s)
use the analytic Gaussian-ReLU mean evaluated at the EMPIRICAL moments
  m_i = mu_i * Phi(mu_i/sig_i) + sig_i * phi(mu_i/sig_i)
which is a smooth function of two well-estimated moments -> lower variance.
The Gaussian assumption is invoked only at the last layer (max CLT accuracy),
so it does NOT accumulate over depth.

We also test an Edgeworth (skew + excess-kurtosis) corrected version to remove
the residual non-Gaussian bias, using empirical 3rd/4th central moments of z.

Measures variance (across trials) and bias^2 (vs 2M-sample GT) -> MSE.
"""

import math
import time

import numpy as np
from scipy.stats import norm

WIDTH, DEPTH = 256, 32
N_TOTAL = 5768
TRIALS = 16
SEEDS = 6
N_GT = 2_000_000
GT_CHUNK = 100_000


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


def forward_to_31(X, weights):
    """Propagate through layers 1..31 (all but the last). Returns a_31."""
    x = X
    for w in weights[:-1]:
        x = np.maximum(x @ w, 0.0)
    return x


def ground_truth_final(weights, seed):
    rng = np.random.default_rng(seed)
    s = np.zeros(WIDTH, dtype=np.float64)
    done = 0
    while done < N_GT:
        m = min(GT_CHUNK, N_GT - done)
        x = rng.standard_normal((m, WIDTH)).astype(np.float32)
        for w in weights:
            x = np.maximum(x @ w, 0.0)
        s += x.sum(axis=0)
        done += m
    return (s / N_GT).astype(np.float64)


# ---- final-layer estimators given a_31 samples and last weight W ----

def est_direct(a31, W):
    """Current estimator: sample-average of ReLU(z)."""
    z = a31 @ W
    return np.maximum(z, 0.0).mean(axis=0)


def est_gauss(a31, W):
    """Rao-Blackwell: analytic Gaussian-ReLU mean at empirical (mu, sigma)."""
    z = a31 @ W
    mu = z.mean(axis=0)
    sig = z.std(axis=0)
    sig = np.maximum(sig, 1e-12)
    t = mu / sig
    return mu * norm.cdf(t) + sig * norm.pdf(t)


def est_edgeworth(a31, W):
    """Gaussian-ReLU mean + Gram-Charlier skew & excess-kurtosis corrections.

    E[ReLU(z)] = mu*Phi(t) + sig*phi(t)
                 + sig*phi(t)*[ g1/6 * A3(t) + g2/24 * A4(t) ]
    where t = mu/sig, g1 = skew, g2 = excess kurtosis, and A3, A4 come from
    integrating ReLU against the Hermite terms of the Gram-Charlier A series.
    """
    z = a31 @ W
    mu = z.mean(axis=0)
    c = z - mu
    var = (c * c).mean(axis=0)
    sig = np.sqrt(np.maximum(var, 1e-24))
    m3 = (c ** 3).mean(axis=0)
    m4 = (c ** 4).mean(axis=0)
    g1 = m3 / np.maximum(sig ** 3, 1e-24)        # skewness
    g2 = m4 / np.maximum(sig ** 4, 1e-24) - 3.0  # excess kurtosis
    t = mu / np.maximum(sig, 1e-12)
    phi = norm.pdf(t)
    Phi = norm.cdf(t)
    base = mu * Phi + sig * phi
    # Gram-Charlier corrections to E[ReLU] (closed form via Hermite integrals).
    # He2(t)=t^2-1, He3(t)=t^3-3t. Correction integrals over z>0:
    #   skew term:  (g1/6)  * sig * phi(t) * He2(t)
    #   kurt term:  (g2/24) * sig * phi(t) * He3(t)
    corr = sig * phi * (g1 / 6.0 * (t * t - 1.0) + g2 / 24.0 * (t ** 3 - 3.0 * t))
    return base + corr


ESTIMATORS = {
    "direct": est_direct,
    "gauss_RB": est_gauss,
    "edgeworth": est_edgeworth,
}


if __name__ == "__main__":
    var_acc = {k: [] for k in ESTIMATORS}
    bias2_acc = {k: [] for k in ESTIMATORS}

    t0 = time.time()
    for seed in range(SEEDS):
        weights = make_mlp(seed)
        W_last = weights[-1]
        gt = ground_truth_final(weights, seed=50_000 + seed)
        preds = {k: np.empty((TRIALS, WIDTH)) for k in ESTIMATORS}
        for t in range(TRIALS):
            rng = np.random.default_rng(seed * 9973 + t * 7 + 1)
            X = samp_anti_whiten(rng)
            a31 = forward_to_31(X, weights)
            for name, fn in ESTIMATORS.items():
                preds[name][t] = fn(a31, W_last)
        for name in ESTIMATORS:
            var_acc[name].append(float(preds[name].var(axis=0, ddof=1).mean()))
            bias2_acc[name].append(float(((preds[name].mean(axis=0) - gt) ** 2).mean()))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)")

    print(f"\nN_TOTAL={N_TOTAL}  trials={TRIALS}  seeds={SEEDS}")
    base_mse = np.mean(var_acc["direct"]) + np.mean(bias2_acc["direct"])
    print(f"{'estimator':>12}  {'variance':>11}  {'bias^2':>11}  {'MSE':>11}  {'vs_direct':>10}")
    for name in ESTIMATORS:
        v = np.mean(var_acc[name])
        b2 = np.mean(bias2_acc[name])
        mse = v + b2
        print(f"{name:>12}  {v:>11.3e}  {b2:>11.3e}  {mse:>11.3e}  {base_mse/mse:>9.2f}x")

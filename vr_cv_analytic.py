"""THE test: control variate with ANALYTIC E[a_L] (deterministic, ~free).

Deep control variates give huge variance reduction (layer-16 -> 3.68x) but need
E[a_L]. An independent MC estimate isn't worth it (MLMC: deep control too costly).
But the analytic Gaussian propagation gives E[a_L] deterministically for ~1B FLOPs,
and a_L is captured for free during the layer-32 forward pass. That preserves the
variance reduction at the SAME sample count, at the cost of bias:
    bias_i = beta_i^T (E_analytic[a_L] - E_true[a_L]).
The analytic error at an intermediate layer L is far smaller than at layer 32,
so this bias may be tolerable. We sweep L and measure realized MSE (var + bias^2)
vs a true 1M-sample GT, comparing analytic-E against oracle-E.
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
RIDGE = 1e-3
CAPTURE = [4, 8, 12, 16, 20]


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


def analytic_means(weights, capture):
    """Gain-approximation covariance propagation -> E[a_l] for captured layers.

    Mirrors the estimator's _gain_approx fallback (numpy/float64).
    """
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


def cv_estimate(caps, a32, E_control, layer):
    a1 = caps[layer]
    n = a1.shape[0]
    a1_mean = a1.mean(axis=0)
    a32_mean = a32.mean(axis=0)
    Ac = (a1 - a1_mean).astype(np.float64)
    Bc = (a32 - a32_mean).astype(np.float64)
    C_cc = (Ac.T @ Ac) / n
    C_ct = (Ac.T @ Bc) / n
    C_cc.flat[:: WIDTH + 1] += RIDGE
    beta = np.linalg.solve(C_cc, C_ct)
    dev = (a1_mean - E_control)
    return a32_mean - dev @ beta


if __name__ == "__main__":
    methods = ["direct"] + [f"oracle_l{l}" for l in CAPTURE] + [f"analytic_l{l}" for l in CAPTURE]
    var_acc = {k: [] for k in methods}
    bias2_acc = {k: [] for k in methods}
    aerr = {l: [] for l in CAPTURE}   # analytic E[a_l] error vs oracle

    t0 = time.time()
    for seed in range(SEEDS):
        weights = make_mlp(seed)
        Ea_oracle, gt = oracle_means(weights, seed=50_000 + seed, capture=CAPTURE)
        Ea_analytic = analytic_means(weights, capture=CAPTURE)
        for l in CAPTURE:
            aerr[l].append(float(np.sqrt(((Ea_analytic[l] - Ea_oracle[l]) ** 2).mean())))
        preds = {k: np.empty((TRIALS, WIDTH)) for k in methods}
        for t in range(TRIALS):
            rng = np.random.default_rng(seed * 9973 + t * 7 + 1)
            X = samp_anti_whiten(rng)
            caps, a32 = forward_capture(X, weights, CAPTURE)
            preds["direct"][t] = a32.mean(axis=0)
            for l in CAPTURE:
                preds[f"oracle_l{l}"][t] = cv_estimate(caps, a32, Ea_oracle[l], l)
                preds[f"analytic_l{l}"][t] = cv_estimate(caps, a32, Ea_analytic[l], l)
        for k in methods:
            var_acc[k].append(float(preds[k].var(axis=0, ddof=1).mean()))
            bias2_acc[k].append(float(((preds[k].mean(axis=0) - gt) ** 2).mean()))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)")

    print(f"\nN_TOTAL={N_TOTAL}  trials={TRIALS}  seeds={SEEDS}")
    print("analytic E[a_l] RMS error vs oracle:")
    for l in CAPTURE:
        print(f"   layer {l:>2}: {np.mean(aerr[l]):.4e}")
    base_mse = np.mean(var_acc["direct"]) + np.mean(bias2_acc["direct"])
    print(f"\n{'method':>14}  {'variance':>11}  {'bias^2':>11}  {'MSE':>11}  {'vs_direct':>10}")
    for k in methods:
        v = np.mean(var_acc[k])
        b2 = np.mean(bias2_acc[k])
        mse = v + b2
        print(f"{k:>14}  {v:>11.3e}  {b2:>11.3e}  {mse:>11.3e}  {base_mse/mse:>9.2f}x")

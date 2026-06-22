"""Last shot at realizing the deep-CV potential: PCA-truncated analytic control.

Deep CV with analytic E[a_L] fails because beta amplifies the analytic error in
the LOW-variance directions of Cov(a_L) (where Cov^-1 blows up), giving bias^2 ~
9e-5 >> the ~1.6e-6 variance it removes. Fix: project the control onto its top-k
principal components. The variance-reduction signal (R^2) concentrates in the top
PCs; the bias amplification lives in the discarded low-variance PCs. Truncation
should keep most of the 3.68x while killing the bias.

Estimator: a_L centered, projected to top-k PCs of Cov(a_L); regress a_32 on the
k-dim projection (ridge); control center = analytic E[a_L] projected the same way.
Sweep (L, k, ridge). Compares against oracle-E to show the realizable fraction.
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
CAPTURE = [12, 16]
KS = [8, 16, 32, 64]
RIDGES = [1e-2, 3e-2, 1e-1]


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


def cv_pca(aL, a32, E_control, k, ridge):
    """PCA-truncated control variate. Returns corrected final-layer mean."""
    n = aL.shape[0]
    aL_mean = aL.mean(axis=0)
    a32_mean = a32.mean(axis=0)
    Ac = (aL - aL_mean).astype(np.float64)
    # top-k PCs of Cov(aL)
    C = (Ac.T @ Ac) / n
    vals, vecs = np.linalg.eigh(C)
    Vk = vecs[:, -k:]                      # (256, k) top-k eigenvectors
    P = Ac @ Vk                            # (n, k) projected control, mean 0
    Bc = (a32 - a32_mean).astype(np.float64)
    Cpp = (P.T @ P) / n                    # (k,k)
    Cpt = (P.T @ Bc) / n                   # (k, 256)
    Cpp.flat[:: k + 1] += ridge
    beta = np.linalg.solve(Cpp, Cpt)       # (k, 256)
    dev_full = (aL_mean - E_control)       # (256,)
    dev_proj = dev_full @ Vk               # (k,)
    return a32_mean - dev_proj @ beta


if __name__ == "__main__":
    methods = ["direct"]
    for L in CAPTURE:
        for k in KS:
            for r in RIDGES:
                methods.append(f"an_L{L}_k{k}_r{r:g}")
                methods.append(f"or_L{L}_k{k}_r{r:g}")
    var_acc = {m: [] for m in methods}
    bias2_acc = {m: [] for m in methods}

    t0 = time.time()
    for seed in range(SEEDS):
        weights = make_mlp(seed)
        Ea_or, gt = oracle_means(weights, seed=50_000 + seed, capture=CAPTURE)
        Ea_an = analytic_means(weights, capture=CAPTURE)
        preds = {m: np.empty((TRIALS, WIDTH)) for m in methods}
        for t in range(TRIALS):
            rng = np.random.default_rng(seed * 9973 + t * 7 + 1)
            X = samp_anti_whiten(rng)
            caps, a32 = forward_capture(X, weights, CAPTURE)
            preds["direct"][t] = a32.mean(axis=0)
            for L in CAPTURE:
                for k in KS:
                    for r in RIDGES:
                        preds[f"an_L{L}_k{k}_r{r:g}"][t] = cv_pca(caps[L], a32, Ea_an[L], k, r)
                        preds[f"or_L{L}_k{k}_r{r:g}"][t] = cv_pca(caps[L], a32, Ea_or[L], k, r)
        for m in methods:
            var_acc[m].append(float(preds[m].var(axis=0, ddof=1).mean()))
            bias2_acc[m].append(float(((preds[m].mean(axis=0) - gt) ** 2).mean()))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)")

    base_mse = np.mean(var_acc["direct"]) + np.mean(bias2_acc["direct"])
    print(f"\nN_TOTAL={N_TOTAL}  trials={TRIALS}  seeds={SEEDS}  direct MSE={base_mse:.3e}")
    print(f"{'method':>18}  {'variance':>11}  {'bias^2':>11}  {'MSE':>11}  {'vs_direct':>10}")
    rows = []
    for m in methods:
        v = np.mean(var_acc[m]); b2 = np.mean(bias2_acc[m]); mse = v + b2
        rows.append((m, v, b2, mse))
    # print direct first, then sort analytic methods by MSE (the realizable ones)
    print(f"{'direct':>18}  {rows[0][1]:>11.3e}  {rows[0][2]:>11.3e}  {rows[0][3]:>11.3e}  {1.0:>9.2f}x")
    for m, v, b2, mse in sorted(rows[1:], key=lambda r: r[3]):
        print(f"{m:>18}  {v:>11.3e}  {b2:>11.3e}  {mse:>11.3e}  {base_mse/mse:>9.2f}x")

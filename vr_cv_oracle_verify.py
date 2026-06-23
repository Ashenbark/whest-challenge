"""Larger oracle verification of deep control-variate ceiling.

Confirms the 3.68x (layer-16) oracle-CV variance reduction is stable across
more seeds. Tests L in {12, 16, 20}, k in {32, 64, 128}, and reports the
best achievable MSE per configuration.

This determines whether the oracle-CV ceiling (3.07-3.68x at layer 16) is
reliable enough to justify finding a better analytic E[a_L].
"""

import math
import time

import numpy as np
from scipy.stats import norm

WIDTH, DEPTH = 256, 32
N_TOTAL = 5768
TRIALS = 24          # more trials for tighter variance estimates
SEEDS = 10           # more seeds to confirm stability
N_GT = 1_000_000
GT_CHUNK = 100_000
CAPTURE = [12, 16, 20]
KS = [32, 64, 128]
RIDGES = [1e-2, 1e-1]


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


def cv_pca_oracle(aL, a32, E_control, k, ridge):
    """Oracle PCA-truncated control variate."""
    n = aL.shape[0]
    aL_mean = aL.mean(axis=0)
    a32_mean = a32.mean(axis=0)
    Ac = (aL - aL_mean).astype(np.float64)
    C = (Ac.T @ Ac) / n
    vals, vecs = np.linalg.eigh(C)
    k_use = min(k, vecs.shape[1])
    Vk = vecs[:, -k_use:]
    P = Ac @ Vk
    Bc = (a32 - a32_mean).astype(np.float64)
    Cpp = (P.T @ P) / n
    Cpt = (P.T @ Bc) / n
    Cpp.flat[:: k_use + 1] += ridge
    beta = np.linalg.solve(Cpp, Cpt)
    dev_full = (aL_mean - E_control)
    dev_proj = dev_full @ Vk
    return a32_mean - dev_proj @ beta


def cv_full_oracle(aL, a32, E_control, ridge):
    """Full (non-PCA) oracle control variate for comparison."""
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
    methods = ["direct"]
    for L in CAPTURE:
        methods.append(f"or_full_L{L}")
    for L in CAPTURE:
        for k in KS:
            for r in RIDGES:
                methods.append(f"or_L{L}_k{k}_r{r:g}")

    var_acc = {m: [] for m in methods}
    bias2_acc = {m: [] for m in methods}

    t0 = time.time()
    for seed in range(SEEDS):
        weights = make_mlp(seed)
        Ea_or, gt = oracle_means(weights, seed=50_000 + seed, capture=CAPTURE)
        preds = {m: np.empty((TRIALS, WIDTH)) for m in methods}
        for t in range(TRIALS):
            rng = np.random.default_rng(seed * 9973 + t * 7 + 1)
            X = samp_anti_whiten(rng)
            caps, a32 = forward_capture(X, weights, CAPTURE)
            preds["direct"][t] = a32.mean(axis=0)
            for L in CAPTURE:
                preds[f"or_full_L{L}"][t] = cv_full_oracle(caps[L], a32, Ea_or[L], 1e-2)
            for L in CAPTURE:
                for k in KS:
                    for r in RIDGES:
                        preds[f"or_L{L}_k{k}_r{r:g}"][t] = cv_pca_oracle(
                            caps[L], a32, Ea_or[L], k, r)

        for m in methods:
            var_acc[m].append(float(preds[m].var(axis=0, ddof=1).mean()))
            bias2_acc[m].append(float(((preds[m].mean(axis=0) - gt) ** 2).mean()))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)")

    base_mse = np.mean(var_acc["direct"]) + np.mean(bias2_acc["direct"])
    print(f"\nN_TOTAL={N_TOTAL}  trials={TRIALS}  seeds={SEEDS}  direct MSE={base_mse:.3e}")
    print(f"{'method':>22}  {'variance':>11}  {'bias^2':>11}  {'MSE':>11}  {'vs_direct':>10}")

    rows = []
    for m in methods:
        v = np.mean(var_acc[m])
        b2 = np.mean(bias2_acc[m])
        mse = v + b2
        rows.append((m, v, b2, mse))

    direct_row = rows[0]
    print(f"{'direct':>22}  {direct_row[1]:>11.3e}  {direct_row[2]:>11.3e}  {direct_row[3]:>11.3e}  {1.0:>9.2f}x")
    for m, v, b2, mse in sorted(rows[1:], key=lambda r: r[3]):
        print(f"{m:>22}  {v:>11.3e}  {b2:>11.3e}  {mse:>11.3e}  {base_mse/mse:>9.2f}x")

    # Per-seed stability report for key methods
    key_methods = ["direct", "or_full_L16", "or_full_L20",
                   f"or_L16_k64_r0.01", f"or_L20_k64_r0.01"]
    print("\n--- Per-seed MSE for key methods ---")
    print(f"{'seed':>4}", end="")
    for m in key_methods:
        print(f"  {m[:14]:>14}", end="")
    print()
    for s in range(SEEDS):
        print(f"{s:>4}", end="")
        for m in key_methods:
            v = var_acc[m][s]
            b2 = bias2_acc[m][s]
            mse = v + b2
            print(f"  {mse:>14.3e}", end="")
        print()

"""Variance-reduction diagnostic: which sampling scheme beats plain antithetic MC?

Compares input-distribution techniques at FIXED sample count (fixed compute) on
256x32 He-init ReLU MLPs:
  - plain      : iid N(0,I)
  - antithetic : (x, -x) pairs  (empirical mean exactly 0)
  - whiten     : iid, then ZCA so empirical mean=0 and cov=I exactly
  - anti+whiten: antithetic then whiten the residual covariance
  - control    : antithetic, plus per-neuron linear control variate using the
                 previous-layer MC mean (Stein-style, unbiased coupling)
Measured against a 2M-sample ground truth.  Reports final-layer MSE (the scored
quantity) averaged over seeds x trials.
"""

import math
import time

import numpy as np

WIDTH, DEPTH = 256, 32
N_TOTAL = 5800          # ~ our budget-floor sample count
N_GT = 2_000_000
GT_CHUNK = 100_000


def make_mlp(seed):
    rng = np.random.default_rng(seed)
    s = math.sqrt(2.0 / WIDTH)
    return [rng.standard_normal((WIDTH, WIDTH)).astype(np.float64) * s for _ in range(DEPTH)]


def forward_means(X, weights):
    x = X
    rows = []
    for w in weights:
        x = np.maximum(x @ w, 0.0)
        rows.append(x.mean(axis=0))
    return np.stack(rows)


def ground_truth(weights, seed=12345):
    rng = np.random.default_rng(seed)
    sums = [np.zeros(WIDTH) for _ in range(DEPTH)]
    done = 0
    while done < N_GT:
        m = min(GT_CHUNK, N_GT - done)
        x = rng.standard_normal((m, WIDTH))
        for li, w in enumerate(weights):
            x = np.maximum(x @ w, 0.0)
            sums[li] += x.sum(axis=0)
        done += m
    return np.stack([s / N_GT for s in sums])


def inv_sqrt_psd(C):
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-12)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


# ---- sampling schemes (return an (N_TOTAL, WIDTH) input batch) ----

def samp_plain(rng):
    return rng.standard_normal((N_TOTAL, WIDTH))


def samp_antithetic(rng):
    h = rng.standard_normal((N_TOTAL // 2, WIDTH))
    return np.concatenate([h, -h], axis=0)


def samp_whiten(rng):
    x = rng.standard_normal((N_TOTAL, WIDTH))
    x = x - x.mean(axis=0, keepdims=True)
    C = (x.T @ x) / x.shape[0]
    return x @ inv_sqrt_psd(C)


def samp_anti_whiten(rng):
    h = rng.standard_normal((N_TOTAL // 2, WIDTH))
    x = np.concatenate([h, -h], axis=0)          # mean exactly 0
    C = (x.T @ x) / x.shape[0]
    return x @ inv_sqrt_psd(C)                    # now cov = I exactly, mean stays 0


def mse_final(pred, gt):
    return float(np.mean((pred[-1] - gt[-1]) ** 2))


def mse_all(pred, gt):
    return float(np.mean((pred - gt) ** 2))


if __name__ == "__main__":
    schemes = {
        "plain": samp_plain,
        "antithetic": samp_antithetic,
        "whiten": samp_whiten,
        "anti+whiten": samp_anti_whiten,
    }
    seeds = range(6)
    trials = 4

    results = {k: [] for k in schemes}
    results_all = {k: [] for k in schemes}

    t0 = time.time()
    for seed in seeds:
        weights = make_mlp(seed)
        gt = ground_truth(weights, seed=10_000 + seed)
        for name, fn in schemes.items():
            for t in range(trials):
                rng = np.random.default_rng(seed * 1000 + t)
                X = fn(rng)
                pred = forward_means(X, weights)
                results[name].append(mse_final(pred, gt))
                results_all[name].append(mse_all(pred, gt))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)")

    print(f"\nN_TOTAL={N_TOTAL}  seeds={len(list(seeds))}  trials={trials}")
    print(f"{'scheme':>14}  {'final_MSE':>12}  {'vs_plain':>9}  {'all_MSE':>12}")
    base = np.mean(results["plain"])
    for name in schemes:
        m = np.mean(results[name])
        ma = np.mean(results_all[name])
        print(f"{name:>14}  {m:>12.3e}  {base/m:>8.2f}x  {ma:>12.3e}")

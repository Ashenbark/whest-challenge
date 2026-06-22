"""Fast VR diagnostic: measure estimator VARIANCE across trials (no GT needed).

Variance reduction is the whole game. For unbiased schemes variance == MSE.
For (mildly biased) whitening we also estimate bias^2 via a 400k-sample GT and
report total MSE = variance + bias^2.

All float32, BLAS-threaded.  Reports the per-neuron variance of the final-layer
mean estimate, averaged over neurons, and the reduction factor vs plain MC.
"""

import math
import time

import numpy as np

WIDTH, DEPTH = 256, 32
N_TOTAL = 5800
TRIALS = 16            # trials per (seed, scheme) to estimate estimator variance
SEEDS = 5
N_GT = 400_000         # for bias estimation only
GT_CHUNK = 100_000


def make_mlp(seed):
    rng = np.random.default_rng(seed)
    s = math.sqrt(2.0 / WIDTH)
    return [(rng.standard_normal((WIDTH, WIDTH)) * s).astype(np.float32) for _ in range(DEPTH)]


def forward_final(X, weights):
    x = X
    for w in weights:
        x = np.maximum(x @ w, 0.0)
    return x.mean(axis=0)          # (WIDTH,) final-layer mean


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
    return (s / N_GT).astype(np.float32)


def inv_sqrt_psd(C):
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-12)
    return ((vecs * (1.0 / np.sqrt(vals))) @ vecs.T).astype(np.float32)


def samp_plain(rng):
    return rng.standard_normal((N_TOTAL, WIDTH)).astype(np.float32)


def samp_antithetic(rng):
    h = rng.standard_normal((N_TOTAL // 2, WIDTH)).astype(np.float32)
    return np.concatenate([h, -h], axis=0)


def samp_whiten(rng):
    x = rng.standard_normal((N_TOTAL, WIDTH)).astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    C = (x.T @ x) / x.shape[0]
    return x @ inv_sqrt_psd(C)


def samp_anti_whiten(rng):
    h = rng.standard_normal((N_TOTAL // 2, WIDTH)).astype(np.float32)
    x = np.concatenate([h, -h], axis=0)
    C = (x.T @ x) / x.shape[0]
    return x @ inv_sqrt_psd(C)


SCHEMES = {
    "plain": samp_plain,
    "antithetic": samp_antithetic,
    "whiten": samp_whiten,
    "anti+whiten": samp_anti_whiten,
}


if __name__ == "__main__":
    # variance[scheme] = list over seeds of (mean-over-neurons of per-neuron variance)
    var_acc = {k: [] for k in SCHEMES}
    bias2_acc = {k: [] for k in SCHEMES}

    t0 = time.time()
    for seed in range(SEEDS):
        weights = make_mlp(seed)
        gt = ground_truth_final(weights, seed=50_000 + seed)
        for name, fn in SCHEMES.items():
            preds = np.empty((TRIALS, WIDTH), dtype=np.float64)
            for t in range(TRIALS):
                rng = np.random.default_rng(seed * 10_000 + t * 7 + 3)
                preds[t] = forward_final(fn(rng), weights)
            # per-neuron variance across trials, averaged over neurons
            var_acc[name].append(float(preds.var(axis=0, ddof=1).mean()))
            # bias^2: (mean over trials - gt)^2 averaged over neurons
            bias2_acc[name].append(float(((preds.mean(axis=0) - gt) ** 2).mean()))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)")

    print(f"\nN_TOTAL={N_TOTAL}  trials={TRIALS}  seeds={SEEDS}")
    print(f"{'scheme':>14}  {'variance':>11}  {'bias^2':>11}  {'MSE=var+b2':>11}  {'VR_factor':>9}")
    base_var = np.mean(var_acc["plain"])
    base_mse = base_var + np.mean(bias2_acc["plain"])
    for name in SCHEMES:
        v = np.mean(var_acc[name])
        b2 = np.mean(bias2_acc[name])
        mse = v + b2
        print(f"{name:>14}  {v:>11.3e}  {b2:>11.3e}  {mse:>11.3e}  {base_mse/mse:>8.2f}x")

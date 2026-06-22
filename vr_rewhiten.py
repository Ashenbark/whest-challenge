"""VR diagnostic: does re-whitening the batch at intermediate layers help?

anti+whiten guarantees cov=I at the input; by layer 32 that structure has
partially eroded. Re-whitening after every k layers would force correct
statistics deeper, potentially extending the 2x benefit — but introduces
bias because the target covariance after ReLU is not I.

We measure both variance (no GT needed) and bias^2 (via 400k-sample GT)
to get MSE = variance + bias^2.
"""

import math
import time

import numpy as np

WIDTH, DEPTH = 256, 32
N_TOTAL = 5642
TRIALS = 16
SEEDS = 4
N_GT = 400_000
GT_CHUNK = 100_000


def make_mlp(seed):
    rng = np.random.default_rng(seed)
    s = math.sqrt(2.0 / WIDTH)
    return [(rng.standard_normal((WIDTH, WIDTH)) * s).astype(np.float32) for _ in range(DEPTH)]


def forward_final(X, weights, rewhiten_every=0):
    x = X
    for i, w in enumerate(weights):
        x = np.maximum(x @ w, 0.0)
        if rewhiten_every > 0 and (i + 1) % rewhiten_every == 0 and i < len(weights) - 1:
            mu = x.mean(axis=0, keepdims=True)
            x = x - mu
            C = (x.T @ x) / x.shape[0]
            vals, vecs = np.linalg.eigh(C)
            vals = np.maximum(vals, 1e-12)
            inv_sqrt = ((vecs * (1.0 / np.sqrt(vals))) @ vecs.T).astype(np.float32)
            x = x @ inv_sqrt
    return x.mean(axis=0)


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


def whiten(x):
    C = (x.T @ x) / x.shape[0]
    return x @ inv_sqrt_psd(C)


def samp_anti_whiten(rng):
    h = rng.standard_normal((N_TOTAL // 2, WIDTH)).astype(np.float32)
    x = np.concatenate([h, -h], axis=0)
    return whiten(x)


SCHEMES = {
    "no_rewhiten": 0,
    "rewhiten_8":  8,
    "rewhiten_4":  4,
    "rewhiten_2":  2,
    "rewhiten_1":  1,
}


if __name__ == "__main__":
    var_acc  = {k: [] for k in SCHEMES}
    bias2_acc = {k: [] for k in SCHEMES}

    t0 = time.time()
    for seed in range(SEEDS):
        weights = make_mlp(seed)
        gt = ground_truth_final(weights, seed=50_000 + seed)
        for name, every in SCHEMES.items():
            preds = np.empty((TRIALS, WIDTH), dtype=np.float64)
            for t in range(TRIALS):
                rng = np.random.default_rng(seed * 9973 + t * 7 + 1)
                x = samp_anti_whiten(rng)
                preds[t] = forward_final(x, weights, rewhiten_every=every)
            var_acc[name].append(float(preds.var(axis=0, ddof=1).mean()))
            bias2_acc[name].append(float(((preds.mean(axis=0) - gt) ** 2).mean()))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)")

    print(f"\nN_TOTAL={N_TOTAL}  trials={TRIALS}  seeds={SEEDS}")
    base_mse = np.mean(var_acc["no_rewhiten"]) + np.mean(bias2_acc["no_rewhiten"])
    print(f"{'scheme':>16}  {'variance':>11}  {'bias^2':>11}  {'MSE':>11}  {'vs_baseline':>11}")
    for name in SCHEMES:
        v  = np.mean(var_acc[name])
        b2 = np.mean(bias2_acc[name])
        mse = v + b2
        print(f"{name:>16}  {v:>11.3e}  {b2:>11.3e}  {mse:>11.3e}  {base_mse/mse:>10.2f}x")

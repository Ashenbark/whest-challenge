"""Does multi-axis antithetic scale? Test K rotation axes (2K antithetic copies).

double_anti (K=2: {I,P} x {+,-}) beat anti+ZCA by 1.065x. Here we sweep K: the
batch is {+-Q_j x} for K orthogonal Q_j (Q_0=I), nq = N/(2K) base points each,
total N samples, then ZCA-whiten. Measures final-layer MSE vs 1M-sample GT at
FIXED total sample count, to find the best number of axes.
"""

import math
import time

import numpy as np

WIDTH, DEPTH = 256, 32
N_HALF = 2963
N = 2 * N_HALF
TRIALS = 32
SEEDS = 4
N_GT = 1_000_000
GT_CHUNK = 100_000
KS = [1, 2, 3, 4, 6, 8]


def make_mlp(seed):
    rng = np.random.default_rng(seed)
    s = math.sqrt(2.0 / WIDTH)
    return [(rng.standard_normal((WIDTH, WIDTH)) * s).astype(np.float32) for _ in range(DEPTH)]


def inv_sqrt_psd(C):
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-12)
    return ((vecs * (1.0 / np.sqrt(vals))) @ vecs.T).astype(np.float32)


def gt_final(weights, seed):
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
    return s / N_GT


def forward_mean(X, weights):
    x = X
    for w in weights:
        x = np.maximum(x @ w, 0.0)
    return x.mean(axis=0)


def multiaxis(rng, K):
    nq = N // (2 * K)
    base = rng.standard_normal((nq, WIDTH)).astype(np.float32)
    parts = []
    for j in range(K):
        if j == 0:
            Q = np.eye(WIDTH, dtype=np.float32)
        else:
            A = rng.standard_normal((WIDTH, WIDTH)).astype(np.float32)
            Q, _ = np.linalg.qr(A)
        Bj = base @ Q.T
        parts.append(Bj)
        parts.append(-Bj)
    x = np.concatenate(parts, axis=0)
    return x @ inv_sqrt_psd((x.T @ x) / x.shape[0])


if __name__ == "__main__":
    var_acc = {K: [] for K in KS}
    bias2_acc = {K: [] for K in KS}
    t0 = time.time()
    for seed in range(SEEDS):
        weights = make_mlp(seed)
        gt = gt_final(weights, seed=60_000 + seed)
        preds = {K: np.empty((TRIALS, WIDTH)) for K in KS}
        for t in range(TRIALS):
            for K in KS:
                rng = np.random.default_rng(seed * 9973 + t * 7 + 1 + K * 131)
                preds[K][t] = forward_mean(multiaxis(rng, K), weights)
        for K in KS:
            var_acc[K].append(float(preds[K].var(axis=0, ddof=1).mean()))
            bias2_acc[K].append(float(((preds[K].mean(axis=0) - gt) ** 2).mean()))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)")

    base = np.mean(var_acc[1]) + np.mean(bias2_acc[1])
    print(f"\nN={N} trials={TRIALS} seeds={SEEDS}  (K=1 is current anti+ZCA)")
    print(f"{'K_axes':>7}  {'variance':>11}  {'bias^2':>11}  {'MSE':>11}  {'vs_K1':>8}")
    for K in KS:
        v = np.mean(var_acc[K]); b2 = np.mean(bias2_acc[K]); mse = v + b2
        print(f"{K:>7}  {v:>11.3e}  {b2:>11.3e}  {mse:>11.3e}  {base/mse:>7.3f}x")

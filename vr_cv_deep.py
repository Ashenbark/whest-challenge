"""Ceiling test: can a control variate from a DEEPER layer stack on whitening?

cv_layer1 gave only 1.07x on top of anti+whiten despite per-sample R^2=0.46,
because whitening already removes the batch-mean fluctuation a CV exploits.
This tests whether a deeper layer (higher raw correlation with layer-32) can
break that overlap. We use ORACLE expectations E[a_l] (from a large separate MC)
to measure the ACHIEVABLE ceiling — if even the oracle CV can't beat 1.3x on top
of whitening, control variates are conclusively dead here.

Controls tested: a_l for l in {1, 2, 4, 8, 16} (single layer each), and the
stack {1,2,4,8} together. Reports realized variance reduction vs direct, all
on top of anti+whiten sampling.
"""

import math
import time

import numpy as np

WIDTH, DEPTH = 256, 32
N_TOTAL = 5768
TRIALS = 16
SEEDS = 4
N_GT = 1_000_000
GT_CHUNK = 100_000
RIDGE = 1e-3
CAPTURE = [1, 2, 4, 8, 16]      # 1-indexed layer numbers to capture


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
    """Return dict{layer -> activation} for captured layers + final activation."""
    x = X
    caps = {}
    cset = set(capture)
    for li, w in enumerate(weights):
        x = np.maximum(x @ w, 0.0)
        if (li + 1) in cset:
            caps[li + 1] = x.copy()
    return caps, x


def oracle_means(weights, seed, capture):
    """Large-MC E[a_l] for captured layers + final, as float64."""
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
    Ea = {l: (sums[l] / N_GT) for l in capture}
    return Ea, (sfin / N_GT)


def cv_estimate(caps, a32, Ea, layers):
    """Multivariate CV using the given layers' activations stacked as controls."""
    cols = [caps[l] for l in layers]
    A = np.concatenate(cols, axis=1).astype(np.float64)        # (n, 256*k)
    Ea_stack = np.concatenate([Ea[l] for l in layers]).astype(np.float64)
    n, p = A.shape
    Amean = A.mean(axis=0)
    a32_mean = a32.mean(axis=0)
    Ac = A - Amean
    Bc = (a32 - a32_mean).astype(np.float64)
    C_cc = (Ac.T @ Ac) / n
    C_ct = (Ac.T @ Bc) / n
    C_cc.flat[:: p + 1] += RIDGE
    beta = np.linalg.solve(C_cc, C_ct)
    dev = (Amean - Ea_stack)
    return a32_mean - dev @ beta


CONTROLS = {
    "direct": None,
    "cv_l1":  [1],
    "cv_l2":  [2],
    "cv_l4":  [4],
    "cv_l8":  [8],
    "cv_l16": [16],
    "cv_1248": [1, 2, 4, 8],
}


if __name__ == "__main__":
    var_acc = {k: [] for k in CONTROLS}
    bias2_acc = {k: [] for k in CONTROLS}

    t0 = time.time()
    for seed in range(SEEDS):
        weights = make_mlp(seed)
        Ea, gt = oracle_means(weights, seed=50_000 + seed, capture=CAPTURE)
        preds = {k: np.empty((TRIALS, WIDTH)) for k in CONTROLS}
        for t in range(TRIALS):
            rng = np.random.default_rng(seed * 9973 + t * 7 + 1)
            X = samp_anti_whiten(rng)
            caps, a32 = forward_capture(X, weights, CAPTURE)
            for name, layers in CONTROLS.items():
                if layers is None:
                    preds[name][t] = a32.mean(axis=0)
                else:
                    preds[name][t] = cv_estimate(caps, a32, Ea, layers)
        for name in CONTROLS:
            var_acc[name].append(float(preds[name].var(axis=0, ddof=1).mean()))
            bias2_acc[name].append(float(((preds[name].mean(axis=0) - gt) ** 2).mean()))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)")

    print(f"\nN_TOTAL={N_TOTAL}  trials={TRIALS}  seeds={SEEDS}  (oracle E[a_l])")
    base_mse = np.mean(var_acc["direct"]) + np.mean(bias2_acc["direct"])
    print(f"{'control':>10}  {'variance':>11}  {'bias^2':>11}  {'MSE':>11}  {'vs_direct':>10}")
    for name in CONTROLS:
        v = np.mean(var_acc[name])
        b2 = np.mean(bias2_acc[name])
        mse = v + b2
        print(f"{name:>10}  {v:>11.3e}  {b2:>11.3e}  {mse:>11.3e}  {base_mse/mse:>9.2f}x")

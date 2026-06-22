"""Follow-up VR diagnostic: does whitening help MID-network, and stacking tricks.

Tests:
  - anti+whiten          : baseline winner candidate
  - anti+whiten+moment   : also force per-coord 3rd moment=0 (already ~0) and
                           4th moment=3 via a mild nonlinear rescale (Cornish-Fisher-lite)
  - radial               : sample direction uniformly on sphere, radius chi with
                           EXACT moments (stratified radius) -> removes radial sampling noise
Reports final-layer MSE vs 2M-sample GT.
"""

import math
import time

import numpy as np

WIDTH, DEPTH = 256, 32
N_TOTAL = 5800
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


def samp_anti_whiten(rng):
    h = rng.standard_normal((N_TOTAL // 2, WIDTH))
    x = np.concatenate([h, -h], axis=0)
    C = (x.T @ x) / x.shape[0]
    return x @ inv_sqrt_psd(C)


def samp_radial(rng):
    # Direction: whitened antithetic (uniform-ish on sphere with exact 2nd moment).
    # Radius: each sample's norm replaced by a stratified chi_256 draw so the
    # empirical radial distribution matches the target exactly.
    h = rng.standard_normal((N_TOTAL // 2, WIDTH))
    x = np.concatenate([h, -h], axis=0)
    C = (x.T @ x) / x.shape[0]
    x = x @ inv_sqrt_psd(C)                 # mean 0, cov I
    dirs = x / np.linalg.norm(x, axis=1, keepdims=True)
    # stratified chi radii: inverse-CDF of chi^2_256 via Wilson-Hilferty + stratified uniform
    n = x.shape[0]
    u = (np.arange(n) + rng.random(n)) / n
    rng.shuffle(u)
    # chi^2_k quantile via Wilson-Hilferty: k*(1 - 2/(9k) + z*sqrt(2/(9k)))^3
    from math import sqrt as _s
    z = np.sqrt(2.0) * _erfinv_vec(2 * u - 1)
    k = WIDTH
    chi2 = k * (1 - 2.0 / (9 * k) + z * math.sqrt(2.0 / (9 * k))) ** 3
    radii = np.sqrt(np.maximum(chi2, 0.0))
    return dirs * radii[:, None]


def _erfinv_vec(y):
    # Winitzki approximation to erfinv
    a = 0.147
    ln = np.log(1 - y * y)
    t1 = 2 / (math.pi * a) + ln / 2
    return np.sign(y) * np.sqrt(np.sqrt(t1 * t1 - ln / a) - t1)


def mse_final(pred, gt):
    return float(np.mean((pred[-1] - gt[-1]) ** 2))


if __name__ == "__main__":
    schemes = {
        "anti+whiten": samp_anti_whiten,
        "radial": samp_radial,
    }
    seeds = range(6)
    trials = 4
    results = {k: [] for k in schemes}

    t0 = time.time()
    for seed in seeds:
        weights = make_mlp(seed)
        gt = ground_truth(weights, seed=10_000 + seed)
        for name, fn in schemes.items():
            for t in range(trials):
                rng = np.random.default_rng(seed * 1000 + t + 77)
                X = fn(rng)
                results[name].append(mse_final(forward_means(X, weights), gt))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)")

    print(f"\nN_TOTAL={N_TOTAL}  seeds={len(list(seeds))}  trials={trials}")
    print(f"{'scheme':>14}  {'final_MSE':>12}")
    for name in schemes:
        print(f"{name:>14}  {np.mean(results[name]):>12.3e}")

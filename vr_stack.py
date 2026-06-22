"""Does anything stack on top of anti+whiten? (variance-based, fast, no GT)

Schemes:
  anti+whiten         : mean 0 + cov I exactly (current best)
  +radial             : also replace each sample norm with a stratified chi_256
                        draw (removes radial sampling noise while keeping cov I-ish)
  +reortho            : re-whiten AFTER radial to restore cov I exactly
"""

import math
import time

import numpy as np

WIDTH, DEPTH = 256, 32
N_TOTAL = 5642          # matches whitened estimator's n_total at budget floor
TRIALS = 16
SEEDS = 4


def make_mlp(seed):
    rng = np.random.default_rng(seed)
    s = math.sqrt(2.0 / WIDTH)
    return [(rng.standard_normal((WIDTH, WIDTH)) * s).astype(np.float32) for _ in range(DEPTH)]


def forward_final(X, weights):
    x = X
    for w in weights:
        x = np.maximum(x @ w, 0.0)
    return x.mean(axis=0)


def inv_sqrt_psd(C):
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-12)
    return ((vecs * (1.0 / np.sqrt(vals))) @ vecs.T).astype(np.float32)


def whiten(x):
    C = (x.T @ x) / x.shape[0]
    return x @ inv_sqrt_psd(C)


def _erfinv(y):
    a = 0.147
    ln = np.log(np.maximum(1 - y * y, 1e-300))
    t1 = 2 / (math.pi * a) + ln / 2
    return np.sign(y) * np.sqrt(np.sqrt(t1 * t1 - ln / a) - t1)


def stratified_chi(n, k, rng):
    u = (np.arange(n) + rng.random(n)) / n
    rng.shuffle(u)
    z = math.sqrt(2.0) * _erfinv(2 * u - 1)
    chi2 = k * (1 - 2.0 / (9 * k) + z * math.sqrt(2.0 / (9 * k))) ** 3
    return np.sqrt(np.maximum(chi2, 0.0)).astype(np.float32)


def samp_anti_whiten(rng):
    h = rng.standard_normal((N_TOTAL // 2, WIDTH)).astype(np.float32)
    x = np.concatenate([h, -h], axis=0)
    return whiten(x)


def samp_radial(rng):
    x = samp_anti_whiten(rng)
    dirs = x / np.linalg.norm(x, axis=1, keepdims=True)
    r = stratified_chi(x.shape[0], WIDTH, rng)
    return dirs * r[:, None]


def samp_radial_reortho(rng):
    return whiten(samp_radial(rng))


SCHEMES = {
    "anti+whiten": samp_anti_whiten,
    "+radial": samp_radial,
    "+radial+reortho": samp_radial_reortho,
}


if __name__ == "__main__":
    var_acc = {k: [] for k in SCHEMES}
    t0 = time.time()
    for seed in range(SEEDS):
        weights = make_mlp(seed)
        for name, fn in SCHEMES.items():
            preds = np.empty((TRIALS, WIDTH))
            for t in range(TRIALS):
                rng = np.random.default_rng(seed * 9973 + t * 7 + 1)
                preds[t] = forward_final(fn(rng), weights)
            var_acc[name].append(float(preds.var(axis=0, ddof=1).mean()))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)")

    print(f"\nN_TOTAL={N_TOTAL}  trials={TRIALS}  seeds={SEEDS}")
    base = np.mean(var_acc["anti+whiten"])
    print(f"{'scheme':>18}  {'variance':>11}  {'vs anti+whiten':>14}")
    for name in SCHEMES:
        v = np.mean(var_acc[name])
        print(f"{name:>18}  {v:>11.3e}  {base/v:>13.2f}x")

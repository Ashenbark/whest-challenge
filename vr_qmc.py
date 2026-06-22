"""VR diagnostic: does scrambled Sobol QMC help on top of anti+whiten?

Tests whether low-discrepancy sequences give additional variance reduction
beyond anti+whiten (our current 2.10x winner). For d=256, QMC stratification
weakens but may still help with higher-order moment matching.

Also tests orthogonal Monte Carlo (OMC): samples are rows of random orthogonal
matrices (blocks of 256), which gives stronger structural constraints than
anti+whiten.
"""

import math
import time

import numpy as np
from scipy.stats import qmc, norm as spnorm

WIDTH, DEPTH = 256, 32
N_TOTAL = 5642
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


# ---- baseline ----

def samp_anti_whiten(rng):
    h = rng.standard_normal((N_TOTAL // 2, WIDTH)).astype(np.float32)
    x = np.concatenate([h, -h], axis=0)
    return whiten(x)


# ---- Sobol QMC variants ----

def samp_sobol_only(rng):
    """Scrambled Sobol -> normal, no whitening."""
    seed_int = int(rng.integers(2**30))
    sampler = qmc.Sobol(d=WIDTH, scramble=True, seed=seed_int)
    n_pow2 = 1 << math.ceil(math.log2(N_TOTAL))
    u = sampler.random(n_pow2)[:N_TOTAL]
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return spnorm.ppf(u).astype(np.float32)


def samp_sobol_whiten(rng):
    """Scrambled Sobol -> normal -> whiten (mean=0 + cov=I)."""
    seed_int = int(rng.integers(2**30))
    sampler = qmc.Sobol(d=WIDTH, scramble=True, seed=seed_int)
    n_pow2 = 1 << math.ceil(math.log2(N_TOTAL))
    u = sampler.random(n_pow2)[:N_TOTAL]
    u = np.clip(u, 1e-6, 1 - 1e-6)
    x = spnorm.ppf(u).astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    return whiten(x)


def samp_sobol_anti_whiten(rng):
    """Scrambled Sobol -> normal for half, antithetic pair, then whiten."""
    seed_int = int(rng.integers(2**30))
    sampler = qmc.Sobol(d=WIDTH, scramble=True, seed=seed_int)
    n_half = N_TOTAL // 2
    n_pow2 = 1 << math.ceil(math.log2(n_half))
    u = sampler.random(n_pow2)[:n_half]
    u = np.clip(u, 1e-6, 1 - 1e-6)
    h = spnorm.ppf(u).astype(np.float32)
    x = np.concatenate([h, -h], axis=0)
    return whiten(x)


# ---- Orthogonal Monte Carlo ----

def samp_orth_blocks(rng):
    """Orthogonal blocks of WIDTH samples each, scaled by chi_WIDTH.

    Within each block, samples are rows of a random orthogonal matrix × chi norm.
    Blocks are independent. Exact cov=I and mean=0 enforced by whiten at the end.
    """
    n_blocks = N_TOTAL // WIDTH
    remainder = N_TOTAL - n_blocks * WIDTH

    parts = []
    for _ in range(n_blocks):
        G = rng.standard_normal((WIDTH, WIDTH)).astype(np.float32)
        Q, _ = np.linalg.qr(G)           # orthonormal rows (after transpose)
        # scale each row to have norm sqrt(WIDTH) → E[‖row‖²] = WIDTH
        Q = Q * math.sqrt(WIDTH)          # rows ⊥, each norm = sqrt(WIDTH)
        parts.append(Q)

    if remainder > 0:
        G = rng.standard_normal((WIDTH, WIDTH)).astype(np.float32)
        Q, _ = np.linalg.qr(G)
        Q = Q[:remainder] * math.sqrt(WIDTH)
        parts.append(Q)

    x = np.concatenate(parts, axis=0)    # (N_TOTAL, WIDTH)
    return whiten(x)                     # re-enforce cov=I exactly


SCHEMES = {
    "anti+whiten": samp_anti_whiten,
    "sobol_only": samp_sobol_only,
    "sobol+whiten": samp_sobol_whiten,
    "sobol+anti+whiten": samp_sobol_anti_whiten,
    "orth_blocks+whiten": samp_orth_blocks,
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
    print(f"{'scheme':>22}  {'variance':>11}  {'vs anti+whiten':>14}")
    for name in SCHEMES:
        v = np.mean(var_acc[name])
        print(f"{name:>22}  {v:>11.3e}  {base/v:>13.2f}x")

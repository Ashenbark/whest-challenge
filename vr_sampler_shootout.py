"""Base-sampler shootout: find the best variance-reduction constant.

All variants draw the SAME number of samples (fixed n) and we measure final-layer
MSE (variance + bias^2) vs a 1M-sample GT. Goal: beat anti+ZCA-whiten (2.10x).

Variants:
  plain            iid N(0,I)
  anti             antithetic (x,-x)
  zca              ZCA-whiten only
  anti_zca         antithetic + ZCA-whiten           (CURRENT)
  anti_pca         antithetic + PCA-whiten (no rotate-back)
  zca_then_anti    whiten the half then antithetic
  anti_zca_radial  + radial (||x||) stratification on the half-batch
  anti_zca_sobol   Sobol base points -> normal -> antithetic + whiten
  double_anti      (x,-x,Px,-Px) for a random orthogonal P, then whiten

Also prints a low-rank diagnostic: the effective rank of Cov(a_32 across samples)
and how much of the per-sample variance lives in the top-k directions (tells us
whether an active-subspace quadrature could beat the moment ceiling).
"""

import math
import time

import numpy as np
from scipy.stats import norm
from scipy.stats import qmc

WIDTH, DEPTH = 256, 32
N_HALF = 2963
N = 2 * N_HALF
TRIALS = 32
SEEDS = 4
N_GT = 1_000_000
GT_CHUNK = 100_000


def make_mlp(seed):
    rng = np.random.default_rng(seed)
    s = math.sqrt(2.0 / WIDTH)
    return [(rng.standard_normal((WIDTH, WIDTH)) * s).astype(np.float32) for _ in range(DEPTH)]


def inv_sqrt_psd(C):
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-12)
    return ((vecs * (1.0 / np.sqrt(vals))) @ vecs.T).astype(np.float32)


def pca_whiten_mat(C):
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-12)
    return (vecs * (1.0 / np.sqrt(vals))).astype(np.float32)   # no rotate-back


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


def forward_samples(X, weights):
    x = X
    for w in weights:
        x = np.maximum(x @ w, 0.0)
    return x


# ---- samplers: each returns an (N, WIDTH) batch ----
def s_plain(rng):
    return rng.standard_normal((N, WIDTH)).astype(np.float32)

def s_anti(rng):
    h = rng.standard_normal((N_HALF, WIDTH)).astype(np.float32)
    return np.concatenate([h, -h], axis=0)

def s_zca(rng):
    x = rng.standard_normal((N, WIDTH)).astype(np.float32)
    x = x - x.mean(0)
    return x @ inv_sqrt_psd((x.T @ x) / x.shape[0])

def s_anti_zca(rng):
    x = s_anti(rng)
    return x @ inv_sqrt_psd((x.T @ x) / x.shape[0])

def s_anti_pca(rng):
    x = s_anti(rng)
    return x @ pca_whiten_mat((x.T @ x) / x.shape[0])

def s_zca_then_anti(rng):
    h = rng.standard_normal((N_HALF, WIDTH)).astype(np.float32)
    h = h @ inv_sqrt_psd((h.T @ h) / h.shape[0])
    return np.concatenate([h, -h], axis=0)

def s_anti_zca_radial(rng):
    # stratified chi radius for the half-batch, random directions, then antithetic+whiten
    d = rng.standard_normal((N_HALF, WIDTH)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    u = (np.arange(N_HALF) + rng.random(N_HALF)) / N_HALF
    r = np.sqrt(2.0) * np.sqrt(_gammaincinv_chi2(WIDTH, u)).astype(np.float32)  # ||x||
    h = d * r[:, None]
    x = np.concatenate([h, -h], axis=0)
    return x @ inv_sqrt_psd((x.T @ x) / x.shape[0])

def _gammaincinv_chi2(k, p):
    from scipy.special import gammaincinv
    return gammaincinv(k / 2.0, p)  # returns (||x||^2)/2 quantile

def s_anti_zca_sobol(rng):
    seed = int(rng.integers(1 << 30))
    eng = qmc.Sobol(d=WIDTH, scramble=True, seed=seed)
    m = int(math.ceil(math.log2(max(2, N_HALF))))
    pts = eng.random_base2(m)[:N_HALF]
    pts = np.clip(pts, 1e-6, 1 - 1e-6)
    h = norm.ppf(pts).astype(np.float32)
    x = np.concatenate([h, -h], axis=0)
    return x @ inv_sqrt_psd((x.T @ x) / x.shape[0])

def s_double_anti(rng):
    nq = N_HALF // 2
    h = rng.standard_normal((nq, WIDTH)).astype(np.float32)
    A = rng.standard_normal((WIDTH, WIDTH)).astype(np.float32)
    P, _ = np.linalg.qr(A)
    Ph = h @ P.T
    x = np.concatenate([h, -h, Ph, -Ph], axis=0)
    return x @ inv_sqrt_psd((x.T @ x) / x.shape[0])


SAMPLERS = {
    "plain": s_plain, "anti": s_anti, "zca": s_zca,
    "anti_zca": s_anti_zca, "anti_pca": s_anti_pca,
    "zca_then_anti": s_zca_then_anti, "anti_zca_radial": s_anti_zca_radial,
    "anti_zca_sobol": s_anti_zca_sobol, "double_anti": s_double_anti,
}


if __name__ == "__main__":
    var_acc = {k: [] for k in SAMPLERS}
    bias2_acc = {k: [] for k in SAMPLERS}
    rank_info = []
    t0 = time.time()
    for seed in range(SEEDS):
        weights = make_mlp(seed)
        gt = gt_final(weights, seed=60_000 + seed)

        # low-rank diagnostic: per-sample a_32 variance spectrum
        rng = np.random.default_rng(seed * 17 + 1)
        A = forward_samples(s_anti_zca(rng), weights)
        Ac = A - A.mean(0)
        # eigenspectrum of sample covariance of a_32
        s = np.linalg.svd(Ac, compute_uv=False)
        ev = (s**2) / Ac.shape[0]
        frac_top = lambda k: ev[:k].sum() / ev.sum()
        eff_rank = (ev.sum()**2) / (ev**2).sum()
        rank_info.append((eff_rank, frac_top(8), frac_top(32), frac_top(64)))

        preds = {k: np.empty((TRIALS, WIDTH)) for k in SAMPLERS}
        for t in range(TRIALS):
            rng = np.random.default_rng(seed * 9973 + t * 7 + 1)
            for name, fn in SAMPLERS.items():
                preds[name][t] = forward_mean(fn(np.random.default_rng(
                    seed * 9973 + t * 7 + 1 + hash(name) % 1000)), weights)
        for name in SAMPLERS:
            var_acc[name].append(float(preds[name].var(axis=0, ddof=1).mean()))
            bias2_acc[name].append(float(((preds[name].mean(axis=0) - gt) ** 2).mean()))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)  "
              f"eff_rank={rank_info[-1][0]:.0f}  top32_var={rank_info[-1][2]:.2f}")

    base = np.mean(var_acc["anti_zca"]) + np.mean(bias2_acc["anti_zca"])
    plain = np.mean(var_acc["plain"]) + np.mean(bias2_acc["plain"])
    print(f"\nN={N} (n_half={N_HALF})  trials={TRIALS}  seeds={SEEDS}")
    er = np.mean([r[0] for r in rank_info])
    print(f"a_32 effective rank ~{er:.0f}/256   "
          f"top8={np.mean([r[1] for r in rank_info]):.2f} "
          f"top32={np.mean([r[2] for r in rank_info]):.2f} "
          f"top64={np.mean([r[3] for r in rank_info]):.2f}")
    print(f"{'sampler':>18}  {'variance':>11}  {'bias^2':>11}  {'MSE':>11}  "
          f"{'vs_plain':>9}  {'vs_anti_zca':>11}")
    rows = [(k, np.mean(var_acc[k]), np.mean(bias2_acc[k]),
             np.mean(var_acc[k]) + np.mean(bias2_acc[k])) for k in SAMPLERS]
    for k, v, b2, mse in sorted(rows, key=lambda r: r[3]):
        print(f"{k:>18}  {v:>11.3e}  {b2:>11.3e}  {mse:>11.3e}  "
              f"{plain/mse:>8.2f}x  {base/mse:>10.3f}x")

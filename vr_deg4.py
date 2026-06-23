"""Degree-4 control-variate ceiling test (the last untested MC lever).

Anti+whiten = exact cubature for all degree-<=2 polynomials of the input
(antithetic kills odd terms, whitening forces mean=0 & cov=I exactly). So the
residual MC variance is ENTIRELY in degree->=4 even components of G(h).
The remaining question: how much of it can a tractable degree-4 control capture?

Setup: capture u = whitened first-layer pre-activation (~N(0,I) exactly in batch),
and final activation a_32. Build degree-4 Hermite controls:
  - diagonal:     He4(u_j) = u_j^4 - 6 u_j^2 + 3            (256 features)
  - random proj:  He4(u . r_m) for random unit r_m          (M features, capture cross)
All have E=0 under N(0,I). Fit beta on TRAIN trials, apply to HELD-OUT trials
(honest out-of-sample variance reduction; ridge-regularized). If ceiling >= ~1.4x,
degree-4 control reaches top-5; if < ~1.15x, it is not the lever.
"""

import math
import time

import numpy as np

WIDTH, DEPTH = 256, 32
N_TOTAL = 5768
TRIALS_TRAIN = 40
TRIALS_TEST = 40
SEEDS = 3
N_GT = 1_000_000
GT_CHUNK = 100_000
M_PROJ = 256          # random degree-4 projection directions
RIDGE = 1e-2


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


def forward_capture_u(X, weights):
    """Return (u, a_32): u = whitened layer-1 pre-activation (batch cov=I exact)."""
    h = X @ weights[0]                       # layer-1 pre-activation (n, 256)
    Ch = (h.T @ h) / h.shape[0]
    u = h @ inv_sqrt_psd(Ch)                 # batch-whitened -> exact N(0,I) moments<=2
    x = np.maximum(h, 0.0)
    for w in weights[1:]:
        x = np.maximum(x @ w, 0.0)
    return u, x


def deg4_features(u, R):
    """Degree-4 Hermite controls, all with E=0 under N(0,I). u:(n,256) R:(256,M)."""
    u2 = u * u
    diag = u2 * u2 - 6.0 * u2 + 3.0          # He4(u_j)  (n,256)
    p = u @ R                                # (n, M) projections ~N(0,1)
    p2 = p * p
    proj = p2 * p2 - 6.0 * p2 + 3.0          # He4(u.r_m) (n,M)
    return np.concatenate([diag, proj], axis=1).astype(np.float64)


def oracle_final(weights, seed):
    rng = np.random.default_rng(seed)
    sfin = np.zeros(WIDTH, dtype=np.float64)
    done = 0
    while done < N_GT:
        m = min(GT_CHUNK, N_GT - done)
        x = rng.standard_normal((m, WIDTH)).astype(np.float32)
        for w in weights:
            x = np.maximum(x @ w, 0.0)
        sfin += x.sum(axis=0)
        done += m
    return sfin / N_GT


if __name__ == "__main__":
    var_direct, var_cv = [], []
    bias2_direct, bias2_cv = [], []
    r2_acc = []
    t0 = time.time()
    for seed in range(SEEDS):
        weights = make_mlp(seed)
        gt = oracle_final(weights, seed=50_000 + seed)
        rngR = np.random.default_rng(7777 + seed)
        R = rngR.standard_normal((WIDTH, M_PROJ)).astype(np.float32)
        R /= np.linalg.norm(R, axis=0, keepdims=True)

        # --- estimate beta on TRAIN trials: regress per-sample a_32 on deg-4 features ---
        Cff = np.zeros((WIDTH + M_PROJ, WIDTH + M_PROJ))
        Cft = np.zeros((WIDTH + M_PROJ, WIDTH))
        resid_num = 0.0
        resid_den = 0.0
        for t in range(TRIALS_TRAIN):
            rng = np.random.default_rng(seed * 9973 + t * 7 + 1)
            X = samp_anti_whiten(rng)
            u, a32 = forward_capture_u(X, weights)
            F = deg4_features(u, R)              # (n, P) mean ~0 per batch
            Fc = F - F.mean(axis=0)
            Bc = (a32 - a32.mean(axis=0)).astype(np.float64)
            Cff += (Fc.T @ Fc) / Fc.shape[0]
            Cft += (Fc.T @ Bc) / Fc.shape[0]
            resid_num += Bc.var(0).sum()
            resid_den += Bc.var(0).sum()
        Cff /= TRIALS_TRAIN
        Cft /= TRIALS_TRAIN
        Cff.flat[:: (WIDTH + M_PROJ) + 1] += RIDGE
        beta = np.linalg.solve(Cff, Cft)        # (P, 256)

        # R^2 of the fit (in-sample, last batch)
        pred_resid = Bc - Fc @ beta
        r2 = 1.0 - pred_resid.var(0).mean() / Bc.var(0).mean()
        r2_acc.append(float(r2))

        # --- evaluate on TEST trials (held-out) ---
        d_preds, cv_preds = [], []
        for t in range(TRIALS_TEST):
            rng = np.random.default_rng(seed * 9973 + (1000 + t) * 7 + 1)
            X = samp_anti_whiten(rng)
            u, a32 = forward_capture_u(X, weights)
            F = deg4_features(u, R)
            a32_mean = a32.mean(axis=0)
            d_preds.append(a32_mean)
            # control variate: subtract beta^T (mean F - E[F]); E[F]=0
            cv_preds.append(a32_mean - F.mean(axis=0) @ beta)
        d_preds = np.array(d_preds)
        cv_preds = np.array(cv_preds)
        var_direct.append(float(d_preds.var(axis=0, ddof=1).mean()))
        var_cv.append(float(cv_preds.var(axis=0, ddof=1).mean()))
        bias2_direct.append(float(((d_preds.mean(0) - gt) ** 2).mean()))
        bias2_cv.append(float(((cv_preds.mean(0) - gt) ** 2).mean()))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)  R2={r2:.3f}  "
              f"var: {var_direct[-1]:.3e}->{var_cv[-1]:.3e}  "
              f"({var_direct[-1]/var_cv[-1]:.2f}x)")

    vd = np.mean(var_direct); vc = np.mean(var_cv)
    bd = np.mean(bias2_direct); bc = np.mean(bias2_cv)
    print(f"\nseeds={SEEDS}  M_proj={M_PROJ}  ridge={RIDGE}  mean R2={np.mean(r2_acc):.3f}")
    print(f"{'method':>10}  {'variance':>11}  {'bias^2':>11}  {'MSE':>11}  {'vs_direct':>10}")
    print(f"{'direct':>10}  {vd:>11.3e}  {bd:>11.3e}  {vd+bd:>11.3e}  {1.0:>9.2f}x")
    print(f"{'deg4_cv':>10}  {vc:>11.3e}  {bc:>11.3e}  {vc+bc:>11.3e}  {(vd+bd)/(vc+bc):>9.2f}x")
    print(f"\nProjected adjusted (0.1x floor): direct={0.1*(vd+bd):.3e}  "
          f"deg4_cv={0.1*(vc+bc):.3e}  (top-5=2.79e-7, best=2.25e-7)")

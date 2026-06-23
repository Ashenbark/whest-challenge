"""Active-subspace test: do the dominant a_32 output modes depend on FEW input dirs?

Shootout found Cov(a_32) has effective rank ~2-3 (deep rank collapse). If the top
output modes phi_k(x) = v_k^T a_32(x) each depend on only a few INPUT directions,
then Gauss-Hermite quadrature in those directions (exact for high-degree polys)
+ MC in the complement could slash the estimator variance far past the 2.1x
moment-cubature ceiling.

Method (all via cheap backprop VJPs, ~1 forward each):
  1. forward an anti+whiten batch -> a_32 ; top-m output PCA dirs v_1..v_m.
  2. for each v_k, compute grad phi_k(x_s) = VJP(v_k) for all samples.
  3. active subspace = eigsystem of E[grad phi_k grad phi_k^T]; report effective
     rank and the fraction of Var(phi_k) captured by the top-r input directions
     (via a quadratic surrogate fit on the projected coords).
If a few input dirs explain most of Var(phi_k), active-subspace quadrature wins.
"""

import math
import time

import numpy as np

WIDTH, DEPTH = 256, 32
N_HALF = 2963
M_MODES = 4


def make_mlp(seed):
    rng = np.random.default_rng(seed)
    s = math.sqrt(2.0 / WIDTH)
    return [(rng.standard_normal((WIDTH, WIDTH)) * s).astype(np.float32) for _ in range(DEPTH)]


def inv_sqrt_psd(C):
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-12)
    return ((vecs * (1.0 / np.sqrt(vals))) @ vecs.T).astype(np.float32)


def samp_anti_whiten(rng, n_half):
    h = rng.standard_normal((n_half, WIDTH)).astype(np.float32)
    x = np.concatenate([h, -h], axis=0)
    return x @ inv_sqrt_psd((x.T @ x) / x.shape[0])


def forward_with_masks(X, weights):
    x = X
    masks = []
    for w in weights:
        z = x @ w
        m = (z > 0).astype(np.float32)
        masks.append(m)
        x = z * m
    return x, masks


def vjp(v_row, masks, weights):
    """Vector-Jacobian product: grad of (v . a_32) wrt input x, for all samples.

    v_row: (256,) output covector. Returns (n, 256) input-space gradients.
    """
    n = masks[0].shape[0]
    g = np.broadcast_to(v_row.astype(np.float32), (n, WIDTH)).copy()
    for li in range(DEPTH - 1, -1, -1):
        g = g * masks[li]              # through ReLU
        g = g @ weights[li].T          # through linear
    return g                            # (n, 256) = grad_x (v . a_32)


if __name__ == "__main__":
    SEEDS = 4
    t0 = time.time()
    for seed in range(SEEDS):
        weights = make_mlp(seed)
        rng = np.random.default_rng(seed * 31 + 7)
        X = samp_anti_whiten(rng, N_HALF)
        a32, masks = forward_with_masks(X, weights)
        Ac = a32 - a32.mean(0)
        # top output modes
        U, s, Vt = np.linalg.svd(Ac, full_matrices=False)
        ev = (s**2) / Ac.shape[0]
        print(f"\nseed {seed}: a_32 output-cov top eigvals frac: "
              f"{(ev[:4]/ev.sum()).round(3)}  (eff_rank "
              f"{(ev.sum()**2)/(ev**2).sum():.1f})")
        for k in range(M_MODES):
            v_k = Vt[k]                          # (256,) output direction
            phi = Ac @ v_k                        # (n,) mode score, var = ev[k]
            grad = vjp(v_k, masks, weights)       # (n,256) input gradients
            # active subspace: eigsystem of mean grad grad^T
            Cg = (grad.T @ grad) / grad.shape[0]
            gval, gvec = np.linalg.eigh(Cg)
            gval = gval[::-1]; gvec = gvec[:, ::-1]
            eff_as = (gval.sum()**2) / (gval**2).sum()
            # how much of Var(phi) does a quadratic surrogate on top-r input dirs explain?
            def surro_r2(r):
                P = gvec[:, :r]                   # (256, r)
                z = X @ P                          # (n, r) projected inputs
                feats = [np.ones(len(z)), *z.T, *(z[:, i] * z[:, j]
                          for i in range(r) for j in range(i, r))]
                F = np.stack(feats, axis=1)
                beta, *_ = np.linalg.lstsq(F, phi, rcond=None)
                resid = phi - F @ beta
                return 1.0 - resid.var() / phi.var()
            r2_2 = surro_r2(2); r2_4 = surro_r2(4); r2_8 = surro_r2(8)
            print(f"   mode {k} (var frac {ev[k]/ev.sum():.2f}): "
                  f"active-subspace eff_rank={eff_as:.1f}  "
                  f"quad-surrogate R^2: r2={r2_2:.2f} r4={r2_4:.2f} r8={r2_8:.2f}")
        print(f"   ({time.time()-t0:.0f}s)")
    print("\nIf eff_rank is small AND R^2 high at low r => active-subspace quadrature viable.")

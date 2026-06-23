"""Arc-cosine kernel herding: rotated herding vs anti+ZCA.

Key insight from pilot: naive herding selects the same fixed points every trial
(deterministic given the fixed candidate pool) → var≈0 but large bias. Fix:

ROTATED KERNEL HERDING:
  1. Build X_herd (N_HALF×n) ONCE via kernel herding over a large random pool.
  2. Each trial: draw random Haar orthogonal Q, use Q@X_herd as the half-batch.
  3. Form (Q@X_herd, -Q@X_herd) → antithetic; optionally ZCA-whiten.

Correctness: N(0,I) is rotationally invariant, so any rotation of an unbiased
estimator is also unbiased. The kernel-MMD quality of Q@X_herd equals X_herd.
The trial-to-trial variance comes from Q varying — the network's fixed weights
break rotational symmetry, giving non-trivial cross-trial variation.

This is analogous to scrambled QMC: a fixed low-discrepancy point set is
randomly scrambled to get unbiasedness while preserving the coverage structure.
"""

import math, time
import numpy as np

WIDTH, DEPTH = 256, 32
N_HALF = 3000          # same as current estimator
N = 2 * N_HALF
N_CAND = 10000         # herding candidate pool (more → better selection)
N_REF  = 10000         # reference set for kernel mean embedding
TRIALS = 24            # trials per seed
SEEDS  = 4
N_GT   = 1_000_000
GT_CHUNK = 100_000


def make_mlp(seed):
    rng = np.random.default_rng(seed)
    s = math.sqrt(2.0 / WIDTH)
    return [(rng.standard_normal((WIDTH, WIDTH)) * s).astype(np.float32)
            for _ in range(DEPTH)]


def inv_sqrt_psd(C):
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-12)
    return ((vecs * (1.0 / np.sqrt(vals))) @ vecs.T).astype(np.float32)


def deep_arccos_kernel(X, Y, depth):
    """Depth-L arc-cosine kernel matrix. Diagonal invariant: k^l(x,x) = ||x||^2."""
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    K = X @ Y.T
    xx = (X ** 2).sum(1, keepdims=True)   # (m,1) — invariant across depth
    yy = (Y ** 2).sum(1, keepdims=True)   # (k,1)
    denom = np.sqrt(np.maximum(xx * yy.T, 1e-30))  # (m,k)
    for _ in range(depth):
        ct = np.clip(K / denom, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = np.arccos(ct)
        K = denom * (np.sin(theta) + (np.pi - theta) * ct) / np.pi
    return K


def random_haar(n, rng):
    """Sample a Haar-uniform orthogonal matrix via QR of a random Gaussian."""
    A = rng.standard_normal((n, n)).astype(np.float64)
    Q, R = np.linalg.qr(A)
    # Fix sign so R diagonal is positive (ensures Haar uniformity)
    Q *= np.sign(np.diag(R))
    return Q.astype(np.float32)


def build_herded_set(rng_seed):
    """Run kernel herding to select N_HALF points from a large pool."""
    rng = np.random.default_rng(rng_seed)
    print("  Drawing candidate pool...", flush=True)
    X_cand = rng.standard_normal((N_CAND, WIDTH)).astype(np.float32)
    Z_ref  = rng.standard_normal((N_REF,  WIDTH)).astype(np.float32)

    print("  Computing kernel mean embedding (N_CAND×N_REF)...", flush=True)
    K_cr = deep_arccos_kernel(X_cand, Z_ref, DEPTH)  # (N_CAND, N_REF)
    mu   = K_cr.mean(1)                               # (N_CAND,)

    print("  Computing within-candidate kernel (N_CAND×N_CAND)...", flush=True)
    K_cc = deep_arccos_kernel(X_cand, X_cand, DEPTH)  # (N_CAND, N_CAND)

    print("  Greedy herding...", flush=True)
    selected = []
    k_acc = np.zeros(N_CAND)
    for t in range(N_HALF):
        score = mu - (k_acc / max(t, 1))
        score[selected] = -1e30
        idx = int(np.argmax(score))
        selected.append(idx)
        k_acc += K_cc[:, idx]

    X_herd = X_cand[selected]   # (N_HALF, WIDTH)
    print(f"  Herding done. Selected {N_HALF} points from {N_CAND} candidates.", flush=True)
    return X_herd


def gt_final(weights, seed):
    rng = np.random.default_rng(seed)
    s = np.zeros(WIDTH, dtype=np.float64)
    done = 0
    while done < N_GT:
        m = min(GT_CHUNK, N_GT - done)
        x = rng.standard_normal((m, WIDTH)).astype(np.float32)
        for w in weights: x = np.maximum(x @ w, 0.0)
        s += x.sum(0); done += m
    return s / N_GT


def forward_mean(X, weights):
    x = X.astype(np.float32)
    for w in weights: x = np.maximum(x @ w, 0.0)
    return x.mean(0)


if __name__ == "__main__":
    var_acc  = {"anti_zca": [], "rherd_anti_zca": [], "rherd_anti": []}
    bias2_acc = {k: [] for k in var_acc}
    t0 = time.time()

    for seed in range(SEEDS):
        weights = make_mlp(seed)
        gt = gt_final(weights, seed=60_000 + seed)
        print(f"\n=== seed {seed}  ({time.time()-t0:.0f}s) ===", flush=True)

        # Build herded set once per seed (network-independent, fixed for all trials)
        X_herd = build_herded_set(rng_seed=seed * 7919 + 3)
        print(f"  kernel structures done ({time.time()-t0:.0f}s)", flush=True)

        preds = {k: np.empty((TRIALS, WIDTH)) for k in var_acc}
        for t in range(TRIALS):
            trial_rng = np.random.default_rng(seed * 9973 + t * 7 + 1)

            # Current baseline: anti+ZCA
            h = trial_rng.standard_normal((N_HALF, WIDTH)).astype(np.float32)
            x = np.concatenate([h, -h], 0)
            xw = x @ inv_sqrt_psd((x.T @ x) / x.shape[0])
            preds["anti_zca"][t] = forward_mean(xw, weights)

            # Rotated herding + anti + ZCA
            Q = random_haar(WIDTH, trial_rng)
            Qh = X_herd @ Q.T                         # (N_HALF, WIDTH) — rotate
            x_rh = np.concatenate([Qh, -Qh], 0)
            # ZCA-whiten the rotated herded batch
            x_rhw = x_rh @ inv_sqrt_psd((x_rh.T @ x_rh) / x_rh.shape[0])
            preds["rherd_anti_zca"][t] = forward_mean(x_rhw.astype(np.float32), weights)

            # Rotated herding + anti (no ZCA, to isolate herding contribution)
            preds["rherd_anti"][t] = forward_mean(x_rh.astype(np.float32), weights)

            if t % 6 == 5:
                print(f"  trial {t+1}/{TRIALS} done ({time.time()-t0:.0f}s)", flush=True)

        for k in var_acc:
            var_acc[k].append(float(preds[k].var(axis=0, ddof=1).mean()))
            bias2_acc[k].append(float(((preds[k].mean(0) - gt) ** 2).mean()))

        base_mse = var_acc["anti_zca"][-1] + bias2_acc["anti_zca"][-1]
        print(f"  seed {seed} results:", flush=True)
        for k in var_acc:
            mse = var_acc[k][-1] + bias2_acc[k][-1]
            print(f"    {k}: mse={mse:.3e}  vs_anti_zca={base_mse/mse:.3f}x", flush=True)

    base_mse = np.mean(var_acc["anti_zca"]) + np.mean(bias2_acc["anti_zca"])
    print(f"\n{'='*70}")
    print(f"N={N}  N_CAND={N_CAND}  N_REF={N_REF}  TRIALS={TRIALS}  SEEDS={SEEDS}")
    print(f"{'method':>20}  {'variance':>11}  {'bias^2':>11}  {'MSE':>11}  {'vs_anti_zca':>12}")
    rows = [(k, np.mean(var_acc[k]), np.mean(bias2_acc[k]),
             np.mean(var_acc[k]) + np.mean(bias2_acc[k])) for k in var_acc]
    for k, v, b2, mse in sorted(rows, key=lambda r: r[3]):
        print(f"{k:>20}  {v:>11.3e}  {b2:>11.3e}  {mse:>11.3e}  {base_mse/mse:>11.3f}x")

"""Reverse-engineering the leaders: can a budget-split MLMC deep CV reach ~5x?

The VR factor the leaders need (MSE 2.25e-6 vs our ~3.7e-6 => ~1.6x on top of
our 2.1x => ~3.4-5x total) eerily matches the deep-CV ORACLE ceiling (L20=4.99x).
So maybe they realize the deep control variate by spending part of the FLOP
budget estimating E[a_L] with an INDEPENDENT shallower MC (L layers, not 32).

This tests the full economics honestly:
  - main estimate: N samples, full 32-layer forward, anti+whiten.
  - control expectation: n' INDEPENDENT samples, L-layer forward (cost L/32 each).
  - CV: m = mean_N(a_32) - beta^T (mean_N(a_L) - Ehat_{n'}[a_L]).
  Ehat is independent of the main batch => the CV stays UNBIASED; its imperfect
  accuracy injects variance beta^2 Var(Ehat), not bias.
Budget is split between N and n' under a fixed floor. We sweep L and the split,
and report the best achievable MSE vs plain anti+whiten at the same total budget.
If the best is still >= direct, deep CV is provably net-negative (leaders aren't
doing this); if some split beats direct by ~1.5x, we've found the path.
"""

import math
import time

import numpy as np

WIDTH, DEPTH = 256, 32
N_GT = 1_000_000
GT_CHUNK = 100_000
SEEDS = 3
TRIALS = 24
CAPTURE = [8, 12, 16, 20]
RIDGE = 1e-2
# total per-MLP sample-layer budget (matches ~floor: 32*N_direct with N_direct~6000)
BUDGET_SL = 32 * 6000      # "sample-layers" available
SPLITS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]   # fraction of budget to E[a_L]


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
    C = (x.T @ x) / x.shape[0]
    return x @ inv_sqrt_psd(C)


def forward_capture(X, weights, capture):
    x = X
    caps = {}
    cset = set(capture)
    for li, w in enumerate(weights):
        x = np.maximum(x @ w, 0.0)
        if (li + 1) in cset:
            caps[li + 1] = x.copy()
    return caps, x


def oracle_means(weights, seed, capture):
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
    return {l: sums[l] / N_GT for l in capture}, sfin / N_GT


def Ehat_shallow(weights, L, n_half, seed):
    """Independent anti+whiten MC of E[a_L] using only the first L layers."""
    rng = np.random.default_rng(seed)
    X = samp_anti_whiten(rng, n_half)
    x = X
    for w in weights[:L]:
        x = np.maximum(x @ w, 0.0)
    return x.mean(axis=0)


if __name__ == "__main__":
    # For each split and L, compute realized MSE. N and n' set by budget.
    results = {}   # (L, split) -> list of mse over seeds
    direct_mse = []
    t0 = time.time()
    for seed in range(SEEDS):
        weights = make_mlp(seed)
        Ea_or, gt = oracle_means(weights, seed=50_000 + seed, capture=CAPTURE)

        # Precompute beta per L from a large pilot (in practice would cost FLOPs;
        # we grant it free to give the CV its BEST case -- a ceiling, not a method).
        rng_p = np.random.default_rng(seed * 13 + 1)
        Xp = samp_anti_whiten(rng_p, 4000)
        caps_p, a32_p = forward_capture(Xp, weights, CAPTURE)
        betas = {}
        for L in CAPTURE:
            Ac = (caps_p[L] - caps_p[L].mean(0)).astype(np.float64)
            Bc = (a32_p - a32_p.mean(0)).astype(np.float64)
            Ccc = (Ac.T @ Ac) / Ac.shape[0]
            Cct = (Ac.T @ Bc) / Ac.shape[0]
            Ccc.flat[:: WIDTH + 1] += RIDGE
            betas[L] = np.linalg.solve(Ccc, Cct)

        # direct: all budget to 32-layer main estimate
        N_direct_half = BUDGET_SL // (32 * 2)
        # trials
        for split in SPLITS:
            for L in CAPTURE:
                key = (L, split)
                results.setdefault(key, [])
            # nothing here; computed below
        # run TRIALS, each draws a main batch and (per split,L) a fresh Ehat batch
        preds_direct = np.empty((TRIALS, WIDTH))
        preds = {(L, split): np.empty((TRIALS, WIDTH)) for L in CAPTURE for split in SPLITS}
        for t in range(TRIALS):
            rng_m = np.random.default_rng(seed * 9973 + t * 7 + 1)
            # direct uses full budget
            Xd = samp_anti_whiten(rng_m, N_direct_half)
            _, a32d = forward_capture(Xd, weights, [])
            preds_direct[t] = a32d.mean(axis=0)
            for split in SPLITS:
                if split == 0.0:
                    for L in CAPTURE:
                        preds[(L, split)][t] = preds_direct[t]
                    continue
                budget_main = int(BUDGET_SL * (1 - split))
                budget_ctrl = int(BUDGET_SL * split)
                N_main_half = budget_main // (32 * 2)
                rng_mm = np.random.default_rng(seed * 7919 + t * 11 + 3)
                Xm = samp_anti_whiten(rng_mm, N_main_half)
                caps_m, a32m = forward_capture(Xm, weights, CAPTURE)
                a32m_mean = a32m.mean(axis=0)
                for L in CAPTURE:
                    n_ctrl_half = budget_ctrl // (L * 2)
                    Eh = Ehat_shallow(weights, L, n_ctrl_half,
                                      seed=seed * 31 + t * 101 + L)
                    dev = caps_m[L].mean(axis=0) - Eh
                    preds[(L, split)][t] = a32m_mean - dev @ betas[L]
        dm = float(preds_direct.var(0, ddof=1).mean()
                   + ((preds_direct.mean(0) - gt) ** 2).mean())
        direct_mse.append(dm)
        for split in SPLITS:
            for L in CAPTURE:
                p = preds[(L, split)]
                mse = float(p.var(0, ddof=1).mean() + ((p.mean(0) - gt) ** 2).mean())
                results[(L, split)].append(mse)
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)  direct={dm:.3e}")

    base = float(np.mean(direct_mse))
    print(f"\nseeds={SEEDS} trials={TRIALS} budget_SL={BUDGET_SL} (N_direct_half={BUDGET_SL//64})")
    print(f"direct anti+whiten MSE = {base:.3e}  (this is the 2.1x baseline)")
    print(f"{'L':>4} {'split':>6} {'MSE':>11} {'vs_direct':>10}")
    best = (None, base)
    for L in CAPTURE:
        for split in SPLITS:
            mse = float(np.mean(results[(L, split)]))
            flag = ""
            if mse < best[1]:
                best = ((L, split), mse)
            if split > 0:
                print(f"{L:>4} {split:>6.1f} {mse:>11.3e} {base/mse:>9.2f}x")
    print(f"\nBEST: L={best[0]}  MSE={best[1]:.3e}  ({base/best[1]:.2f}x vs direct)")
    print("If BEST <= ~1.0x, deep-CV MLMC is net-negative => leaders are NOT doing this.")

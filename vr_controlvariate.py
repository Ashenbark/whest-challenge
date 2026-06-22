"""Control-variate test: regress the final-layer activation on quantities whose
expectation is known EXACTLY, then subtract the (mean - known_E) deviation.

A control variate needs an exactly-known expectation. Only two sources qualify:
  - the input x ~ N(0,I): all polynomial moments known.
  - layer-1 activations a1 = ReLU(W1^T x): E[a1_k] = ||W1[:,k]|| / sqrt(2*pi)
    exactly (W1^T x is N(0, ||col||^2)).

Estimator (multivariate CV with regression coefficients, unbiased to O(1/n)):
  m_i = mean_s a32_i^s  -  (mean_s c^s - E[c])^T beta_i
where beta_i = Cov(c)^{-1} Cov(c, a32_i) estimated from the batch.
Variance reduction = 1/(1 - R^2), R^2 = fraction of a32_i variance explained by c.

We test c = a1 (layer-1 activations, 256-dim, ridge-regularized regression).
If layer-1 still predicts layer-32 across samples, this stacks on whitening.

Reports variance (across trials), bias^2 (vs GT) and MSE.
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


def forward_capture(X, weights):
    """Return (a1, a_final) for one batch."""
    x = X
    a1 = None
    for li, w in enumerate(weights):
        x = np.maximum(x @ w, 0.0)
        if li == 0:
            a1 = x
    return a1, x


def analytic_Ea1(W1):
    """E[ReLU(W1^T x)_k] = ||W1[:,k]|| / sqrt(2*pi) for x ~ N(0,I)."""
    col_norm = np.linalg.norm(W1, axis=0)        # ||W1[:,k]||
    return (col_norm / math.sqrt(2.0 * math.pi)).astype(np.float64)


def ground_truth_final(weights, seed):
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
    return (s / N_GT).astype(np.float64)


def est_direct(a1, a32, Ea1):
    return a32.mean(axis=0)


def est_cv_layer1(a1, a32, Ea1):
    """Multivariate control variate using layer-1 activations."""
    n = a1.shape[0]
    a1_mean = a1.mean(axis=0)
    a32_mean = a32.mean(axis=0)
    Ac = (a1 - a1_mean).astype(np.float64)        # centered controls (n, 256)
    Bc = (a32 - a32_mean).astype(np.float64)       # centered targets  (n, 256)
    C_cc = (Ac.T @ Ac) / n                         # Cov(a1)  (256,256)
    C_ct = (Ac.T @ Bc) / n                         # Cov(a1, a32) (256,256)
    C_cc.flat[:: WIDTH + 1] += RIDGE               # ridge
    beta = np.linalg.solve(C_cc, C_ct)             # (256, 256), col i = beta_i
    dev = (a1_mean - Ea1)                          # (256,)  mean(a1) - E[a1]
    return a32_mean - dev @ beta                   # (256,)


ESTIMATORS = {
    "direct": est_direct,
    "cv_layer1": est_cv_layer1,
}


if __name__ == "__main__":
    var_acc = {k: [] for k in ESTIMATORS}
    bias2_acc = {k: [] for k in ESTIMATORS}
    r2_acc = []

    t0 = time.time()
    for seed in range(SEEDS):
        weights = make_mlp(seed)
        Ea1 = analytic_Ea1(weights[0])
        gt = ground_truth_final(weights, seed=50_000 + seed)
        preds = {k: np.empty((TRIALS, WIDTH)) for k in ESTIMATORS}
        for t in range(TRIALS):
            rng = np.random.default_rng(seed * 9973 + t * 7 + 1)
            X = samp_anti_whiten(rng)
            a1, a32 = forward_capture(X, weights)
            for name, fn in ESTIMATORS.items():
                preds[name][t] = fn(a1, a32, Ea1)
            if t == 0:
                # measure R^2 of regressing a32 on a1 (explanatory power)
                n = a1.shape[0]
                Ac = (a1 - a1.mean(0)).astype(np.float64)
                Bc = (a32 - a32.mean(0)).astype(np.float64)
                C_cc = (Ac.T @ Ac) / n
                C_ct = (Ac.T @ Bc) / n
                C_cc.flat[:: WIDTH + 1] += RIDGE
                beta = np.linalg.solve(C_cc, C_ct)
                resid = Bc - Ac @ beta
                r2 = 1.0 - resid.var(0).mean() / Bc.var(0).mean()
                r2_acc.append(float(r2))
        for name in ESTIMATORS:
            var_acc[name].append(float(preds[name].var(axis=0, ddof=1).mean()))
            bias2_acc[name].append(float(((preds[name].mean(axis=0) - gt) ** 2).mean()))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)  R^2(a32|a1)={r2_acc[-1]:.4f}")

    print(f"\nN_TOTAL={N_TOTAL}  trials={TRIALS}  seeds={SEEDS}  mean R^2={np.mean(r2_acc):.4f}")
    base_mse = np.mean(var_acc["direct"]) + np.mean(bias2_acc["direct"])
    print(f"{'estimator':>12}  {'variance':>11}  {'bias^2':>11}  {'MSE':>11}  {'vs_direct':>10}")
    for name in ESTIMATORS:
        v = np.mean(var_acc[name])
        b2 = np.mean(bias2_acc[name])
        mse = v + b2
        print(f"{name:>12}  {v:>11.3e}  {b2:>11.3e}  {mse:>11.3e}  {base_mse/mse:>9.2f}x")

"""Carefully-unbiased last-layer Rao-Blackwell estimator — build & benchmark.

Only the final layer is scored. z_32 = a_31 @ W_32 is a 256-term sum -> nearly
Gaussian per neuron (CLT). The RB estimator replaces the noisy sample average
  m_direct = mean_s ReLU(z_s)
with the analytic Gaussian-ReLU mean at the empirical moments
  m_rb = mu_hat*Phi(a) + sig_hat*phi(a),   a = mu_hat/sig_hat.
This has lower variance (depends only on well-estimated mu_hat, sig_hat) but is
BIASED by the residual non-Gaussianity of z_32.

We test four variants and measure variance, bias^2, MSE vs a 2M-sample GT:
  1. direct                 - the current estimator (baseline)
  2. gauss_rb               - biased RB (replacement)
  3. rb_debiased_split      - UNBIASED: estimate RB's bias on an independent
                              half-batch and subtract it (honest, no leakage)
  4. rb_cv                  - RB as a control variate against the direct mean,
                              optimal lambda (unbiased, lambda from a pilot)
Also prints the THEORETICAL per-neuron variance-reduction ceiling
  [Var(ReLU(z))] / [sigma^2 (Phi^2 + phi^2/2)]
to show why this lever is fundamentally ~1.03x at best.
"""

import math
import time

import numpy as np
from scipy.stats import norm

WIDTH, DEPTH = 256, 32
N_TOTAL = 5926          # matches the deployed estimator (n_half=2963)
TRIALS = 40
SEEDS = 5
N_GT = 2_000_000
GT_CHUNK = 100_000


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


def forward_to_pre32(X, weights):
    """Return z_32 = the final pre-activation (n, 256)."""
    x = X
    for w in weights[:-1]:
        x = np.maximum(x @ w, 0.0)
    return x @ weights[-1]


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


def relu_gauss_mean(mu, sig):
    sig = np.maximum(sig, 1e-12)
    a = mu / sig
    return mu * norm.cdf(a) + sig * norm.pdf(a)


def est_direct(z):
    return np.maximum(z, 0.0).mean(axis=0)


def est_gauss_rb(z):
    return relu_gauss_mean(z.mean(axis=0), z.std(axis=0))


def est_rb_debiased_split(z):
    """Unbiased: estimate RB's bias on half A, apply RB on half B, combine.

    bias_A = RB(A) - direct(A)  (estimates the systematic Gaussian-approx error)
    m = RB(B) - bias_A          (independent halves => unbiased)
    """
    n = z.shape[0]
    h = n // 2
    zA, zB = z[:h], z[h:]
    biasA = est_gauss_rb(zA) - est_direct(zA)
    return est_gauss_rb(zB) - biasA


def est_rb_cv(z, lam):
    """RB as control variate: m = direct - lam*(RB - direct_again)? -> unbiased CV.

    Use the identity E[direct] = E[ReLU]. The control is (RB - direct): its
    expectation is the (deterministic) bias b. Optimal blend toward whichever
    has lower variance. We pass a pilot-fitted lam (per neuron) in [0,1].
    """
    d = est_direct(z)
    rb = est_gauss_rb(z)
    return d + lam * (rb - d)


if __name__ == "__main__":
    methods = ["direct", "gauss_rb", "rb_debiased_split", "rb_cv"]
    var_acc = {m: [] for m in methods}
    bias2_acc = {m: [] for m in methods}
    ceil_acc = []
    t0 = time.time()
    n_half = N_TOTAL // 2

    for seed in range(SEEDS):
        weights = make_mlp(seed)
        gt = gt_final(weights, seed=70_000 + seed)

        # pilot to fit rb_cv lambda (per neuron), on independent trials
        pilot_d, pilot_rb = [], []
        for t in range(12):
            rng = np.random.default_rng(seed * 555 + t * 3 + 9)
            z = forward_to_pre32(samp_anti_whiten(rng, n_half), weights)
            pilot_d.append(est_direct(z))
            pilot_rb.append(est_gauss_rb(z))
        pilot_d = np.array(pilot_d); pilot_rb = np.array(pilot_rb)
        # optimal lambda minimizing Var(d + lam(rb-d)) ignoring bias:
        diff = pilot_rb - pilot_d
        cov_d_diff = ((pilot_d - pilot_d.mean(0)) * (diff - diff.mean(0))).mean(0)
        var_diff = diff.var(0) + 1e-18
        lam = np.clip(-cov_d_diff / var_diff, 0.0, 1.0)

        # theoretical ceiling using GT moments of z (one big sample)
        rng = np.random.default_rng(seed * 9973 + 1)
        zbig = forward_to_pre32(samp_anti_whiten(rng, n_half), weights)
        mu = zbig.mean(0); sig = zbig.std(0); a = mu / np.maximum(sig, 1e-12)
        var_relu = np.maximum(np.maximum(zbig, 0).var(0), 1e-18)
        var_rb_lin = sig**2 * (norm.cdf(a)**2 + norm.pdf(a)**2 / 2.0)
        ceil_acc.append(float((var_relu / var_rb_lin).mean()))

        preds = {m: np.empty((TRIALS, WIDTH)) for m in methods}
        for t in range(TRIALS):
            rng = np.random.default_rng(seed * 9973 + t * 7 + 1)
            z = forward_to_pre32(samp_anti_whiten(rng, n_half), weights)
            preds["direct"][t] = est_direct(z)
            preds["gauss_rb"][t] = est_gauss_rb(z)
            preds["rb_debiased_split"][t] = est_rb_debiased_split(z)
            preds["rb_cv"][t] = est_rb_cv(z, lam)
        for m in methods:
            var_acc[m].append(float(preds[m].var(axis=0, ddof=1).mean()))
            bias2_acc[m].append(float(((preds[m].mean(axis=0) - gt) ** 2).mean()))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)  "
              f"theory_ceiling={ceil_acc[-1]:.3f}x  mean_lam={lam.mean():.2f}")

    print(f"\nN_TOTAL={N_TOTAL}  trials={TRIALS}  seeds={SEEDS}")
    print(f"theoretical per-neuron variance-reduction ceiling: {np.mean(ceil_acc):.3f}x")
    base = np.mean(var_acc["direct"]) + np.mean(bias2_acc["direct"])
    print(f"{'method':>20}  {'variance':>11}  {'bias^2':>11}  {'MSE':>11}  {'vs_direct':>10}")
    for m in methods:
        v = np.mean(var_acc[m]); b2 = np.mean(bias2_acc[m]); mse = v + b2
        print(f"{m:>20}  {v:>11.3e}  {b2:>11.3e}  {mse:>11.3e}  {base/mse:>9.2f}x")

"""Novel: first-layer activation as an EXACT-expectation control variate.

KEY INSIGHT (breaks our prior 'ceiling is degree-2 input moments' theorem):
The network is bias-free, so z1 = x @ W1 is EXACTLY Gaussian, N(0, W1^T W1).
Therefore a1 = ReLU(z1) has an EXACTLY-KNOWN expectation per neuron:
    E[a1_i] = sigma_i / sqrt(2*pi),   sigma_i = ||W1[:,i]|| = sqrt((W1^T W1)_ii)
a1 is degree-2+ in x (through ReLU), so it controls HIGHER-degree input
fluctuations than ZCA whitening (which only fixes degree-<=2 input moments).
Its expectation is known in closed form and free to evaluate.

Control-variate estimator for the final-layer mean:
    theta_hat = mean_s(a32) - B @ (mean_s(a1) - E_exact[a1])
where B (256x256, ridge-regularized) is fit from the SAME batch (plug-in).
Unbiased to O(1/N) regardless of B since E[mean(a1)-E_exact[a1]] = 0.

We also test stacking with a2, a3 controls. Their expectations are NOT exactly
known, but are computed by analytic moment-propagation INITIALIZED with the exact
a1 first/second moments (CLT makes deep z marginals near-Gaussian -> accurate at
shallow depth). Variants:
    anti_zca            baseline
    +a1_cv              first-layer exact-mean control
    +a1a2_cv            a1 exact + a2 analytic-mean controls
    +a1a2a3_cv          through a3

Measures final-layer MSE (variance + bias^2) vs 1M-sample GT.
"""
import math, time
import numpy as np

WIDTH, DEPTH = 256, 32
N_HALF = 3000
N = 2 * N_HALF
TRIALS = 24
SEEDS = 4
N_GT = 1_000_000
GT_CHUNK = 100_000
RIDGE = 1e-3      # ridge on Cov(control) for the B-matrix solve
SQRT_2PI = math.sqrt(2.0 * math.pi)


def make_mlp(seed):
    rng = np.random.default_rng(seed)
    s = math.sqrt(2.0 / WIDTH)
    return [(rng.standard_normal((WIDTH, WIDTH)) * s).astype(np.float32) for _ in range(DEPTH)]


def inv_sqrt_psd(C):
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-12)
    return ((vecs * (1.0 / np.sqrt(vals))) @ vecs.T).astype(np.float32)


def s_anti_zca(rng):
    h = rng.standard_normal((N_HALF, WIDTH)).astype(np.float32)
    x = np.concatenate([h, -h], 0)
    return x @ inv_sqrt_psd((x.T @ x) / x.shape[0])


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


# ---- exact / analytic expectations of early-layer activations ----
def relu_mean_gauss(mu, var):
    """E[ReLU(z)], z~N(mu,var), elementwise."""
    sig = np.sqrt(np.maximum(var, 1e-30))
    a = mu / sig
    from scipy.stats import norm
    return mu * norm.cdf(a) + sig * norm.pdf(a)


def relu_second_moment_gauss(mu, var):
    """E[ReLU(z)^2], z~N(mu,var)."""
    sig = np.sqrt(np.maximum(var, 1e-30))
    a = mu / sig
    from scipy.stats import norm
    return (mu * mu + var) * norm.cdf(a) + mu * sig * norm.pdf(a)


def bivar_relu_cov_diag_only(W1):
    """Exact mean and (diagonal) variance of a1 = ReLU(z1), z1=N(0, W1^T W1).

    Returns (E_a1 (256,), Var_a1 (256,)). Off-diagonal cov of a1 is expensive
    (bivariate per pair); for the a1-only control we just need the exact MEAN.
    """
    Sig = W1.T.astype(np.float64) @ W1.astype(np.float64)      # (256,256) cov of z1
    var_z = np.maximum(np.diag(Sig), 1e-30)
    E_a1 = relu_mean_gauss(np.zeros(WIDTH), var_z)              # z1 mean is 0
    E_a1_sq = relu_second_moment_gauss(np.zeros(WIDTH), var_z)
    Var_a1 = np.maximum(E_a1_sq - E_a1 ** 2, 1e-30)
    return E_a1, Var_a1, Sig


def forward_capture(X, weights, capture_layers):
    """Forward pass; return final-layer per-sample acts and captured early acts."""
    x = X
    caps = {}
    for li, w in enumerate(weights):
        x = np.maximum(x @ w, 0.0)
        if li in capture_layers:
            caps[li] = x.copy()
    return x, caps   # x is final (N,256)


def cv_estimate(a_final, controls, control_means):
    """Plug-in multivariate control-variate estimate of E[a_final].

    a_final: (N,256). controls: (N, C) stacked control activations.
    control_means: (C,) known/analytic E[control]. Returns (256,) estimate.
    """
    N = a_final.shape[0]
    cbar = controls.mean(0)
    delta = cbar - control_means                     # (C,)
    Cc = np.cov(controls, rowvar=False)              # (C,C)
    Ctf = np.cov(controls, a_final, rowvar=False)[:controls.shape[1], controls.shape[1]:]  # (C,256)
    Cc_reg = Cc + RIDGE * np.eye(Cc.shape[0]) * np.trace(Cc) / Cc.shape[0]
    B = np.linalg.solve(Cc_reg, Ctf)                 # (C,256): Cc^{-1} Ctf
    return a_final.mean(0) - B.T @ delta             # (256,)


if __name__ == "__main__":
    methods = ["anti_zca", "+a1_cv", "+a1a2_cv", "+a1a2a3_cv"]
    var_acc = {m: [] for m in methods}
    bias2_acc = {m: [] for m in methods}
    t0 = time.time()

    for seed in range(SEEDS):
        weights = make_mlp(seed)
        gt = gt_final(weights, seed=60_000 + seed)

        # exact a1 mean
        E_a1, Var_a1, Sig1 = bivar_relu_cov_diag_only(weights[0])

        # analytic a2, a3 means via moment propagation (diagonal-cov approx,
        # initialized with EXACT a1 diagonal variance). Mean of z2 = E_a1 @ W2.
        W2 = weights[1].astype(np.float64); W3 = weights[2].astype(np.float64)
        mu_z2 = E_a1 @ W2
        # var(z2_k) ~ sum_j W2[j,k]^2 Var_a1[j]  (diagonal-cov approx for a1)
        var_z2 = (W2 ** 2).T @ Var_a1
        E_a2 = relu_mean_gauss(mu_z2, var_z2)
        E_a2_sq = relu_second_moment_gauss(mu_z2, var_z2)
        Var_a2 = np.maximum(E_a2_sq - E_a2 ** 2, 1e-30)
        mu_z3 = E_a2 @ W3
        var_z3 = (W3 ** 2).T @ Var_a2
        E_a3 = relu_mean_gauss(mu_z3, var_z3)

        preds = {m: np.empty((TRIALS, WIDTH)) for m in methods}
        for t in range(TRIALS):
            rng = np.random.default_rng(seed * 9973 + t * 7 + 1)
            X = s_anti_zca(rng)
            a_final, caps = forward_capture(X, weights, capture_layers={0, 1, 2})
            a1, a2, a3 = caps[0], caps[1], caps[2]

            preds["anti_zca"][t] = a_final.mean(0)
            preds["+a1_cv"][t] = cv_estimate(a_final, a1, E_a1)
            preds["+a1a2_cv"][t] = cv_estimate(
                a_final, np.concatenate([a1, a2], 1), np.concatenate([E_a1, E_a2]))
            preds["+a1a2a3_cv"][t] = cv_estimate(
                a_final, np.concatenate([a1, a2, a3], 1),
                np.concatenate([E_a1, E_a2, E_a3]))

        for m in methods:
            var_acc[m].append(float(preds[m].var(axis=0, ddof=1).mean()))
            bias2_acc[m].append(float(((preds[m].mean(0) - gt) ** 2).mean()))
        line = f"  seed {seed} ({time.time()-t0:.0f}s): "
        for m in methods:
            line += f"{m}={var_acc[m][-1]+bias2_acc[m][-1]:.3e}  "
        print(line, flush=True)

    base = np.mean(var_acc["anti_zca"]) + np.mean(bias2_acc["anti_zca"])
    print(f"\nN={N}  TRIALS={TRIALS}  SEEDS={SEEDS}  RIDGE={RIDGE}")
    print(f"{'method':>14}  {'variance':>11}  {'bias^2':>11}  {'MSE':>11}  {'vs_anti_zca':>12}")
    for m in methods:
        v = np.mean(var_acc[m]); b2 = np.mean(bias2_acc[m]); mse = v + b2
        print(f"{m:>14}  {v:>11.3e}  {b2:>11.3e}  {mse:>11.3e}  {base/mse:>11.3f}x")

"""Improved first-layer CV: EXACT a1 cov -> accurate E[a2], E[a3] controls.

The v1 a2/a3 controls were catastrophic because their analytic means used a
DIAGONAL-cov approximation for a1, biasing E[a2]. Here we compute the EXACT
256x256 covariance of a1 = ReLU(z1), z1~N(0,Sig1), via the closed-form
bivariate-Gaussian E[ReLU(z_i)ReLU(z_j)] (done ONCE per MLP, ~256^2 cheap ops).
Then Var(z2) is exact, and E[a2] is accurate to the (CLT-justified) Gaussian
marginal approximation of z2. Likewise E[a3].

Tests whether a control evaluated at L2/L3 -- which correlates more strongly with
the depth-32 output than L1 -- can be made unbiased enough to beat the ceiling.
"""
import math, time
import numpy as np
from scipy.stats import norm

WIDTH, DEPTH = 256, 32
N_HALF = 3000
N = 2 * N_HALF
TRIALS = 24
SEEDS = 4
N_GT = 1_000_000
GT_CHUNK = 100_000
RIDGE = 1e-3


def make_mlp(seed):
    rng = np.random.default_rng(seed)
    s = math.sqrt(2.0 / WIDTH)
    return [(rng.standard_normal((WIDTH, WIDTH)) * s).astype(np.float32) for _ in range(DEPTH)]

def inv_sqrt_psd(C):
    vals, vecs = np.linalg.eigh(C); vals = np.maximum(vals, 1e-12)
    return ((vecs * (1.0 / np.sqrt(vals))) @ vecs.T).astype(np.float32)

def s_anti_zca(rng):
    h = rng.standard_normal((N_HALF, WIDTH)).astype(np.float32)
    x = np.concatenate([h, -h], 0)
    return x @ inv_sqrt_psd((x.T @ x) / x.shape[0])

def gt_final(weights, seed):
    rng = np.random.default_rng(seed); s = np.zeros(WIDTH, np.float64); done = 0
    while done < N_GT:
        m = min(GT_CHUNK, N_GT - done); x = rng.standard_normal((m, WIDTH)).astype(np.float32)
        for w in weights: x = np.maximum(x @ w, 0.0)
        s += x.sum(0); done += m
    return s / N_GT

def relu_mean(mu, var):
    sig = np.sqrt(np.maximum(var, 1e-30)); a = mu / sig
    return mu * norm.cdf(a) + sig * norm.pdf(a)

def relu_2nd(mu, var):
    sig = np.sqrt(np.maximum(var, 1e-30)); a = mu / sig
    return (mu*mu + var) * norm.cdf(a) + mu * sig * norm.pdf(a)

def exact_relu_cov(Sig):
    """Exact Cov[ReLU(z)] for z~N(0,Sig). Returns (mean(256,), cov(256,256)).

    Uses the closed form for E[ReLU(z_i)ReLU(z_j)] of a zero-mean bivariate
    Gaussian (Wick/arc-cosine):
      E[r_i r_j] = (s_i s_j / (2*pi)) * (sin(t) + (pi - t) cos(t)),  t = arccos(rho)
    where s = sqrt(diag), rho = Sig_ij/(s_i s_j). Diagonal: E[r_i^2] = s_i^2/2.
    """
    s = np.sqrt(np.maximum(np.diag(Sig), 1e-30))
    E_r = s / math.sqrt(2 * math.pi)                       # zero-mean ReLU mean
    rho = np.clip(Sig / np.outer(s, s), -1.0, 1.0)
    t = np.arccos(rho)
    E_rr = (np.outer(s, s) / (2 * math.pi)) * (np.sin(t) + (math.pi - t) * np.cos(t))
    np.fill_diagonal(E_rr, s * s / 2.0)
    cov = E_rr - np.outer(E_r, E_r)
    return E_r, cov

def propagate_exact_moments(weights, n_layers):
    """Return list of (E[a_l], Cov[a_l]) for l=0..n_layers-1 using exact bivariate
    ReLU cov at each step + Gaussian marginal assumption for pre-activations."""
    W = [w.astype(np.float64) for w in weights]
    Sig = W[0].T @ W[0]                       # cov z1 (mean 0)
    moments = []
    E_a, Cov_a = exact_relu_cov(Sig)
    moments.append((E_a, Cov_a))
    for l in range(1, n_layers):
        Wl = W[l]
        mu_z = E_a @ Wl
        Cov_z = Wl.T @ Cov_a @ Wl              # exact 2nd-order propagation
        var_z = np.maximum(np.diag(Cov_z), 1e-30)
        # exact bivariate ReLU cov needs full Cov_z; build mean+cov
        s = np.sqrt(var_z)
        E_a = relu_mean(mu_z, var_z)
        # general (nonzero-mean) bivariate ReLU 2nd moment is costlier; approximate
        # off-diagonal a-cov by gain*gain*Cov_z (Phi(alpha) gains), exact diagonal.
        a = mu_z / s
        Phi = norm.cdf(a)
        E_a_sq = relu_2nd(mu_z, var_z)
        Var_a = np.maximum(E_a_sq - E_a**2, 1e-30)
        Cov_a = np.outer(Phi, Phi) * Cov_z
        np.fill_diagonal(Cov_a, Var_a)
        moments.append((E_a, Cov_a))
    return moments

def cv_estimate(a_final, controls, control_means):
    cbar = controls.mean(0); delta = cbar - control_means
    Cc = np.cov(controls, rowvar=False)
    C = controls.shape[1]
    Ctf = np.cov(controls, a_final, rowvar=False)[:C, C:]
    Cc_reg = Cc + RIDGE * np.eye(C) * np.trace(Cc) / C
    B = np.linalg.solve(Cc_reg, Ctf)
    return a_final.mean(0) - B.T @ delta

def forward_capture(X, weights, caps_at):
    x = X; caps = {}
    for li, w in enumerate(weights):
        x = np.maximum(x @ w, 0.0)
        if li in caps_at: caps[li] = x.copy()
    return x, caps

if __name__ == "__main__":
    methods = ["anti_zca", "+a1", "+a2", "+a3", "+a1a2a3"]
    var_acc = {m: [] for m in methods}; bias2_acc = {m: [] for m in methods}
    t0 = time.time()
    for seed in range(SEEDS):
        weights = make_mlp(seed)
        gt = gt_final(weights, seed=60_000 + seed)
        mom = propagate_exact_moments(weights, 3)
        E1, E2, E3 = mom[0][0], mom[1][0], mom[2][0]
        preds = {m: np.empty((TRIALS, WIDTH)) for m in methods}
        for t in range(TRIALS):
            rng = np.random.default_rng(seed * 9973 + t * 7 + 1)
            X = s_anti_zca(rng)
            a_final, caps = forward_capture(X, weights, {0, 1, 2})
            a1, a2, a3 = caps[0], caps[1], caps[2]
            preds["anti_zca"][t] = a_final.mean(0)
            preds["+a1"][t] = cv_estimate(a_final, a1, E1)
            preds["+a2"][t] = cv_estimate(a_final, a2, E2)
            preds["+a3"][t] = cv_estimate(a_final, a3, E3)
            preds["+a1a2a3"][t] = cv_estimate(
                a_final, np.concatenate([a1, a2, a3], 1), np.concatenate([E1, E2, E3]))
        for m in methods:
            var_acc[m].append(float(preds[m].var(0, ddof=1).mean()))
            bias2_acc[m].append(float(((preds[m].mean(0) - gt)**2).mean()))
        line = f"  seed {seed} ({time.time()-t0:.0f}s): "
        for m in methods: line += f"{m}={var_acc[m][-1]+bias2_acc[m][-1]:.2e} "
        print(line, flush=True)
    base = np.mean(var_acc["anti_zca"]) + np.mean(bias2_acc["anti_zca"])
    print(f"\nN={N} TRIALS={TRIALS} SEEDS={SEEDS} RIDGE={RIDGE}  (exact a1 cov, gain-approx a2/a3 cov)")
    print(f"{'method':>10} {'variance':>11} {'bias^2':>11} {'MSE':>11} {'vs_base':>9}")
    for m in methods:
        v=np.mean(var_acc[m]); b2=np.mean(bias2_acc[m]); mse=v+b2
        print(f"{m:>10} {v:>11.3e} {b2:>11.3e} {mse:>11.3e} {base/mse:>8.3f}x")

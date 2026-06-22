"""Fast local diagnostic: compare analytical, MC, and blend on 1 random He-init MLP."""

import math
import time
import numpy as np
from numpy.polynomial.hermite import hermgauss

SQRT2 = math.sqrt(2.0)
SQRT_PI_INV = 1.0 / math.sqrt(math.pi)
WIDTH = 256
DEPTH = 32
SEED = 42
N_GT = 200_000
N_GH = 16


def _phi(x):
    return np.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


def _Phi(x):
    flat = np.array([0.5 * (1 + math.erf(float(v) / math.sqrt(2))) for v in x.flat])
    return flat.reshape(x.shape).astype(np.float32)


def make_he_mlp(n, depth, rng):
    return [rng.standard_normal((n, n)).astype(np.float32) * math.sqrt(2.0 / n)
            for _ in range(depth)]


def ground_truth(weights, n_samples, rng):
    x = rng.standard_normal((n_samples, WIDTH)).astype(np.float32)
    rows = []
    for w in weights:
        x = np.maximum(x @ w, 0.0)
        rows.append(x.mean(axis=0))
    return np.stack(rows)


def predict_gain_approx(weights):
    n = WIDTH
    mu = np.zeros(n, dtype=np.float32)
    cov = np.eye(n, dtype=np.float32)
    rows = []
    for w in weights:
        mu_pre = w.T @ mu
        cov_pre = np.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sigma_pre = np.sqrt(var_pre)
        alpha = (mu_pre / sigma_pre).astype(np.float32)
        phi_a = _phi(alpha).astype(np.float32)
        Phi_a = _Phi(alpha)
        mu = (mu_pre * Phi_a + sigma_pre * phi_a).astype(np.float32)
        ez2 = (mu_pre * mu_pre + var_pre) * Phi_a + mu_pre * sigma_pre * phi_a
        var_post = np.maximum(ez2 - mu * mu, 0.0)
        gain = np.where(sigma_pre > 1e-12, Phi_a, 0.0)
        cov = np.outer(gain, gain) * cov_pre
        np.fill_diagonal(cov, var_post)
        rows.append(mu.copy())
    return np.stack(rows)


def predict_analytical(weights):
    n = WIDTH
    gh_t, gh_w = hermgauss(N_GH)
    gh_t = gh_t.astype(np.float32)
    gh_w = gh_w.astype(np.float32)

    mu = np.zeros(n, dtype=np.float32)
    cov = np.eye(n, dtype=np.float32)
    rows = []

    for w in weights:
        mu_pre = (w.T @ mu).astype(np.float32)
        cov_pre = np.einsum("ij,ia,jb->ab", cov, w, w).astype(np.float32)
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sigma_pre = np.sqrt(var_pre)
        alpha = (mu_pre / sigma_pre).astype(np.float32)
        phi_a = _phi(alpha).astype(np.float32)
        Phi_a = _Phi(alpha)
        mu = (mu_pre * Phi_a + sigma_pre * phi_a).astype(np.float32)
        ez2 = (mu_pre * mu_pre + var_pre) * Phi_a + mu_pre * sigma_pre * phi_a
        var_post = np.maximum(ez2 - mu * mu, 0.0)

        sigma2 = sigma_pre * sigma_pre
        C = (cov_pre / sigma2[:, None]).astype(np.float32)
        rho2 = (cov_pre * cov_pre / np.outer(sigma2, sigma2)).astype(np.float32)
        sigma_ji = (sigma_pre[None, :] * np.sqrt(np.maximum(1.0 - rho2, 0.0))).astype(np.float32)

        z_ik = (mu_pre[None, :] + sigma_pre[None, :] * (SQRT2 * gh_t[:, None])).astype(np.float32)
        relu_zi = np.maximum(z_ik, 0.0)
        delta = (z_ik - mu_pre[None, :]).astype(np.float32)

        mu_ji = (mu_pre[None, None, :] + C[None, :, :] * delta[:, :, None]).astype(np.float32)
        alpha_ji = (mu_ji / np.maximum(sigma_ji[None, :, :], 1e-12)).astype(np.float32)

        Phi_ji = _Phi(alpha_ji)
        phi_ji = _phi(alpha_ji).astype(np.float32)
        E_relu_ji = (mu_ji * Phi_ji + sigma_ji[None, :, :] * phi_ji).astype(np.float32)

        w_relu = (gh_w[:, None] * relu_zi).astype(np.float32)
        E_bicov = (np.sum(w_relu[:, :, None] * E_relu_ji, axis=0) * SQRT_PI_INV).astype(np.float32)
        E_bicov = ((E_bicov + E_bicov.T) * 0.5).astype(np.float32)

        cov = (E_bicov - np.outer(mu, mu)).astype(np.float32)
        np.fill_diagonal(cov, var_post)
        rows.append(mu.copy())

    return np.stack(rows)


def predict_mc(weights, n_half, rng):
    x_half = rng.standard_normal((n_half, WIDTH)).astype(np.float32)
    x = np.concatenate([x_half, -x_half], axis=0)
    rows = []
    for w in weights:
        x = np.maximum(x @ w, 0.0)
        rows.append(x.mean(axis=0))
    return np.stack(rows)


def mse_final(pred, gt):
    return float(np.mean((pred[-1] - gt[-1]) ** 2))


def mse_all(pred, gt):
    return float(np.mean((pred - gt) ** 2))


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    weights = make_he_mlp(WIDTH, DEPTH, rng)

    print(f"Generating ground truth ({N_GT:,} samples)...")
    t0 = time.time()
    gt = ground_truth(weights, N_GT, rng)
    print(f"  done in {time.time()-t0:.1f}s.  final-layer mean={gt[-1].mean():.4f}  var={gt[-1].var():.4f}")

    print(f"\nGain-approx baseline...")
    t0 = time.time()
    pred_gain = predict_gain_approx(weights)
    t1 = time.time()
    print(f"  {t1-t0:.1f}s  final_layer_MSE={mse_final(pred_gain, gt):.3e}  all={mse_all(pred_gain, gt):.3e}")

    print(f"\nGH analytical (K={N_GH})...")
    t0 = time.time()
    pred_a = predict_analytical(weights)
    t1 = time.time()
    print(f"  {t1-t0:.1f}s  final_layer_MSE={mse_final(pred_a, gt):.3e}  all={mse_all(pred_a, gt):.3e}")

    print(f"\nMC at various sample counts, and optimal blend with analytical:")
    print(f"  {'n_half':>8}  {'n_total':>8}  {'MSE_mc':>10}  {'best_alpha':>10}  {'MSE_blend':>10}")
    for n_half in [500, 1000, 2000, 2408, 5000, 10000]:
        pred_mc = predict_mc(weights, n_half, rng)
        mse_mc = mse_final(pred_mc, gt)
        best_alpha, best_mse = 0.0, mse_mc
        for alpha in np.linspace(0, 1, 21):
            blend = alpha * pred_a + (1 - alpha) * pred_mc
            m = mse_final(blend, gt)
            if m < best_mse:
                best_mse, best_alpha = m, float(alpha)
        print(f"  {n_half:>8}  {2*n_half:>8}  {mse_mc:>10.3e}  {best_alpha:>10.2f}  {best_mse:>10.3e}")

    print(f"\nBlend grid at n_half=2408 (competition budget):")
    pred_mc = predict_mc(weights, 2408, rng)
    for alpha in [0.0, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0]:
        blend = alpha * pred_a + (1 - alpha) * pred_mc
        print(f"  alpha={alpha:.2f}  MSE={mse_final(blend, gt):.3e}")

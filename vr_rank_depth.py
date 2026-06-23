"""Effective rank of the RAW activation batch a_L vs depth (not mean-subtracted).

If a_L (N x n) collapses to low rank R by some depth, then a_L @ W costs ~N*R*n
instead of N*n^2 -> cheaper samples. ReLU destroys exact low rank, but if z_{L+1}
is near-low-rank we can propagate the latent score distribution cheaply.
"""
import math, numpy as np
WIDTH, DEPTH = 256, 32
N = 6000

def make_mlp(seed):
    rng = np.random.default_rng(seed)
    s = math.sqrt(2.0/WIDTH)
    return [(rng.standard_normal((WIDTH,WIDTH))*s).astype(np.float32) for _ in range(DEPTH)]

def eff_rank(M):
    # mean-INCLUSIVE: rank of raw activation matrix
    s = np.linalg.svd(M, compute_uv=False)
    ev = s**2
    return (ev.sum()**2)/(ev**2).sum(), ev

for seed in range(3):
    W = make_mlp(seed)
    rng = np.random.default_rng(1000+seed)
    x = rng.standard_normal((N,WIDTH)).astype(np.float32)
    print(f"\n=== seed {seed} ===  layer: effrank(raw) | effrank(centered) | top-singval-frac")
    for li,w in enumerate(W):
        x = np.maximum(x@w,0.0)
        er_raw, ev = eff_rank(x)
        xc = x - x.mean(0)
        er_c, evc = eff_rank(xc)
        if li in (0,1,3,7,11,15,19,23,27,31):
            frac1 = ev[0]/ev.sum()
            print(f"  L{li+1:2d}: raw={er_raw:6.1f}  centered={er_c:6.2f}  top_raw_frac={frac1:.3f}")

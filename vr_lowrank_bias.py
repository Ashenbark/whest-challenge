"""Feasibility: does low-rank deep propagation inject acceptable bias?

Forward a fixed anti+ZCA batch exactly -> per-batch 'truth' final mean.
Then re-propagate layers > L0 keeping only rank-R structure after each ReLU
(truncated SVD re-compression). Measure bias injected into final-layer mean,
vs the MC variance floor (~2e-6 target MSE).
"""
import math, numpy as np
WIDTH, DEPTH = 256, 32
N_HALF = 3000
N = 2*N_HALF

def make_mlp(seed):
    rng = np.random.default_rng(seed)
    s = math.sqrt(2.0/WIDTH)
    return [(rng.standard_normal((WIDTH,WIDTH))*s).astype(np.float32) for _ in range(DEPTH)]

def inv_sqrt_psd(C):
    v,Q = np.linalg.eigh(C); v=np.maximum(v,1e-12)
    return ((Q*(1.0/np.sqrt(v)))@Q.T).astype(np.float32)

def anti_zca(rng):
    h = rng.standard_normal((N_HALF,WIDTH)).astype(np.float32)
    x = np.concatenate([h,-h],0)
    return x@inv_sqrt_psd((x.T@x)/x.shape[0])

def exact_final(x,W):
    for w in W: x = np.maximum(x@w,0.0)
    return x.mean(0)

def lowrank_final(x0,W,L0,R):
    """Exact for layers<=L0, then rank-R truncated propagation."""
    x = x0
    for li,w in enumerate(W):
        x = np.maximum(x@w,0.0)
        if li>=L0:
            # re-compress to mean + rank-R fluctuation
            mu = x.mean(0)
            xc = x-mu
            U,s,Vt = np.linalg.svd(xc,full_matrices=False)
            xc_r = (U[:,:R]*s[:R])@Vt[:R]
            x = mu + xc_r
    return x.mean(0)

for seed in range(3):
    W = make_mlp(seed)
    rng = np.random.default_rng(7*seed+3)
    x0 = anti_zca(rng)
    truth = exact_final(x0,W)
    print(f"\nseed {seed}: ||truth||^2/n mean = {(truth**2).mean():.4e}")
    for L0 in (8,12,16):
        for R in (4,8,16,32):
            est = lowrank_final(x0,W,L0,R)
            bias2 = ((est-truth)**2).mean()
            print(f"  L0={L0:2d} R={R:2d}: injected bias^2 = {bias2:.3e}")

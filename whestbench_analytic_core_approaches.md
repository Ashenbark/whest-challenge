# Beyond the control variate: routes to a more accurate analytic core for WhestBench

**Context.** WhestBench (the ARC White-Box Estimation Challenge) asks for the per-neuron
expected post-ReLU activation of a randomly-initialised ReLU MLP (width 256, depth 8,
He-initialised weights, standard-normal input), scored as MSE on the final layer against
a large Monte-Carlo reference. ARC's own companion method ("Estimating the expected output
of wide random MLPs more efficiently than sampling", arXiv:2605.05179) is a form of
**cumulant propagation**: start from a Gaussian approximation to the activation
distributions and track the lowest-order deviations. It beats sampling at large width but,
in ARC's words, *"breaks down as the depth grows"* — and L=8 is squarely in that regime.

Your current estimator already implements a respectable cumulant-propagation core (Wick
covariance + Edgeworth mean + a κ₃ diagram) and bolts a Monte-Carlo control variate on top.
Your own analysis is correct: the affine CV explains only ρ²≈0.21 of the final-layer
variance, the FLOP wall floors CV MSE near 6e-6, and a leaderboard MSE near 1e-8 cannot come
from the CV lever. It must come from the **analytic core**. This note is a menu of
mathematical routes to make that core dramatically more accurate, ordered roughly by
expected payoff-per-effort, with the depth problem kept front and centre.

---

## 0. The one reframing that tells you where to spend effort

Means and covariances propagate **exactly** through the linear layers. If layer ℓ−1's
post-activations have *true* mean `μ` and *true* covariance `Σ`, then the pre-activations
`z^(ℓ) = Wᵀ h^(ℓ−1)` have

```
E[z^(ℓ)]   = Wᵀ μ          (exact)
Cov[z^(ℓ)] = Wᵀ Σ W        (exact)
```

No approximation is committed in the matmul. **Every bit of error in the whole pipeline is
committed at one step: turning the moments of `z^(ℓ)` into the moments of
`h^(ℓ) = ReLU(z^(ℓ))`.** That step is approximate because `z^(ℓ)` is not Gaussian (for ℓ≥2),
yet we feed it a Gaussian (or lightly cumulant-corrected) closure.

Two consequences drive everything below:

1. **Per-layer accuracy = closure accuracy.** Make `moments(z) → E[ReLU(z)]` and
   `→ Cov[ReLU(zᵢ),ReLU(zⱼ)]` as exact as the available statistics allow.
2. **Depth is a compounding problem, not just a per-layer one.** Layer ℓ's closure consumes
   layer ℓ−1's *approximate* moments. A small bias in `Σ` at layer 2 re-enters the closure
   at layer 3, gets re-amplified, and so on. At L=8 the dominant error is accumulated, not
   local. So you must (a) shrink per-layer closure error and (b) attack the *propagation* of
   error, not only its creation.

Also worth banking: **layer 1 is exact.** `z^(1)=W₁ᵀX` is genuinely Gaussian, so layer-1
means, variances and *all* pairwise covariances are closed-form (Section 1). Spend nothing
approximating it.

---

## 1. Exact bivariate ReLU second moments (kill the ρ-series truncation)

Your covariance step expands `Cov[ReLU(zᵢ),ReLU(zⱼ)]` as a Hermite/Wick series in the
correlation `P=ρ`, truncated at `k=4` (the `1·P + 2·P² + 6·P³ + 24·P⁴` terms). That series is
exact only as `ρ→0`. Across depth, neurons within a layer become **more** correlated, so
`|ρ|` grows and the truncated tail `Σ_{k>4} aₖ² k! ρᵏ` becomes a real, *compounding* bias in
`Σ` — which then poisons every downstream mean.

There is no need to truncate: the bivariate ReLU second moment has a **closed form**.

**Centered case** (means zero, unit variance, correlation ρ — the Cho–Saul order-1
arc-cosine kernel):

```
E[ReLU(x)ReLU(y)] = (1/2π) · ( sin θ + (π − θ) cos θ ),    θ = arccos(ρ)
```

(check: ρ=1 ⇒ 1/2 = E[ReLU²]; ρ=0 ⇒ 1/2π = E[ReLU]²). Scale by `σᵢσⱼ` for general
variances. This single substitution makes the *entire* covariance recursion exact whenever
pre-activation means are negligible.

**Non-centered case** (your neurons have α=μ/σ that is O(1), so you need this). The general
`E[max(X,0)max(Y,0)]` for bivariate normal `(μₓ,μᵧ,σₓ,σᵧ,ρ)` is closed-form in the bivariate
normal CDF `Φ₂`, the univariate `Φ/φ`, and **Owen's T** function. Implement via
`scipy.special.owens_t` and `scipy.stats.multivariate_normal.cdf` (both cheap), or tabulate.

**Why it helps with depth specifically.** The error you remove is `O(ρ⁵)` per pair and grows
with correlation; correlations grow with depth; the error feeds the variance, which feeds
`α`, which feeds the mean. Removing it stops one of the main depth-amplified leaks.
**Cost:** O(n²) special-function calls per layer — comparable to what you already do.
**Risk:** low; this is a strict accuracy upgrade of an existing step.

---

## 2. Higher-order cumulant propagation, done with the *exact* mean correction

Your mean closure currently keeps the Gaussian term plus the κ₃ (skewness) term. The natural
— and the ARC-blessed — next move is to carry the **fourth cumulant** and add the matching
mean correction. The key facts:

The pre-activation cumulants are exact contractions of the previous layer's *joint* activation
cumulants:

```
κ₃(zᵢ)  = Σ_{jkl}   Wᵢⱼ Wᵢₖ Wᵢₗ · κ₃(hⱼ,hₖ,hₗ)
κ₄(zᵢ)  = Σ_{jklm}  Wᵢⱼ Wᵢₖ Wᵢₗ Wᵢₘ · κ₄(hⱼ,hₖ,hₗ,hₘ)
```

and because the *inputs* to a ReLU layer are jointly Gaussian (their distribution is fixed by
pairwise correlations alone), the joint rectified-Gaussian cumulants `κ₃(hⱼ,hₖ,hₗ)`,
`κ₄(…)` are **functions of pairwise correlations only** and have closed forms. So κ₃, κ₄ of
the pre-activations are, in principle, *exact*. Your existing `k3_curr` line is a cheap
leading "diagram" of the κ₃ contraction; you can add a κ₄ diagram in the same spirit.

**The exact Edgeworth mean correction (verified).** Write `z = μ + σξ`, `α = μ/σ`, and let the
standardized `ξ` have skewness `γ₁` and excess kurtosis `γ₂`. The Gram–Charlier/Edgeworth
density correction `(γ₁/3!)He₃ + (γ₂/4!)He₄` contributes to `E[ReLU(z)] = σ·E[(α+ξ)₊]` through
the integrals `Iₖ(α) = ∫_{−α}^∞ (α+ξ)Heₖ(ξ)φ(ξ)dξ`. These collapse to a clean closed form

```
Iₖ(α) = φ(α) · [ α·He_{k−1}(−α) + Heₖ(−α) + k·He_{k−2}(−α) ]
```

which gives (I verified both numerically to 6+ digits):

```
I₃(α) = −α·φ(α)            →  skew term:     σ · (γ₁/6)  · (−α·φ(α))
I₄(α) = (α²−1)·φ(α)        →  kurtosis term: σ · (γ₂/24) · (α²−1)·φ(α)
```

The κ₃ term matches your code's `c3 = σ·(−1/6)·α·φ(α)` exactly — confirmation your core is
correct so far. **The new, droppable line is the kurtosis term:**

```
mu_h += σ_pre · (1/24) · (α² − 1) · φ(α) · γ₂        # γ₂ = standardized κ₄ of z
```

So the full corrected mean is

```
E[ReLU(z)] ≈ σ[φ(α)+αΦ(α)]  −  σ(γ₁/6)α φ(α)  +  σ(γ₂/24)(α²−1)φ(α)  + …
```

**Why it helps with depth.** The Gaussian closure's leading error is exactly the skew/kurtosis
you're now subtracting; it is the term ARC describes as "the lowest-order deviations from
Gaussian", and getting κ₄ right is what their depth-sensitive accuracy hinges on.
**Cost:** the mean correction is free (one extra term per neuron). The real cost is the κ₄
*contraction*; keep it cheap with a diagram/low-rank approximation, or only over the
dominant correlated blocks. **Risk:** Edgeworth is an *asymptotic* series — it can overshoot
and even produce negative densities when |γ| is not small. Cap the correction (you already
clamp `k3t∈[−2,2]`) or, better, switch to the saddlepoint closure below, which does not blow up.

---

## 3. Saddlepoint (tilted-Gaussian) closure instead of Edgeworth

Edgeworth/Gram–Charlier truncation is the wrong tool past a point: the series is asymptotic,
oscillates, and degrades precisely in the tails and at moderate skew — and its errors
*compound* over 8 layers. The **saddlepoint approximation** uses the same cumulant inputs
(κ₁..κ₄, or a fitted CGF) but is uniformly more accurate, respects positivity, and is exact
for Gaussians, so it degrades gracefully.

Procedure per neuron: approximate the cumulant generating function `K(t)` of `z` from its
cumulants (e.g. a quartic CGF, or a fitted exponential-family tilt), then either
(a) saddlepoint-reconstruct the density `f̂(x)` and integrate `∫₀^∞ x f̂(x) dx` by 1-D
quadrature, or (b) use a Lugannani–Rice tail expression for `P(z>x)` and
`E[ReLU(z)] = ∫₀^∞ P(z>x) dx`. Both are O(n·Q) with a small quadrature count Q.

**Why it helps with depth.** It removes the "series blows up" failure mode that is the most
likely cause of ARC's depth degradation, and it keeps the mean estimate sane even where γ
is no longer small (which is exactly what happens as activations get squeezed through many
ReLUs). **Cost:** modest (root-find + small quadrature per neuron). **Risk:** implementation
care around the saddlepoint root near α≈0; fall back to the Edgeworth/Gaussian form there.

---

## 4. Gauss–Hermite quadrature against a reconstructed marginal

A robust sibling of Approach 3 that avoids series truncation entirely. Once you have a
marginal model for `z^(ℓ)ᵢ` (from cumulants via saddlepoint, or a moment-matched
skew-normal / generalized-Gaussian / two-piece-normal), compute

```
E[ReLU(zᵢ)] = ∫ max(z,0) f̂ᵢ(z) dz
```

by a fixed Gauss–Hermite rule (15–30 nodes is plenty for a 1-D smooth integrand). This is
exact up to the accuracy of `f̂ᵢ`, never produces the oscillatory artefacts of a truncated
expansion, and is trivially vectorised across neurons. A **skew-normal** fit (match κ₁,κ₂,κ₃)
is a particularly cheap, positivity-respecting closure that already captures the dominant
non-Gaussianity; a **two-piece (split) normal** is even cheaper and surprisingly good for
rectified-then-summed variables. **Cost:** O(n·Q). **Risk:** low; the main question is which
parametric family best matches summed-rectified-Gaussian shape — worth a quick empirical
bake-off on layers 2–4 where you can compare to a heavy MC reference offline.

---

## 5. Finite-width (NLO) perturbative correction

At infinite width the Gaussian closure is *exact* (this is the NNGP / arc-cosine-kernel limit,
and the per-layer recursion in Approach 1 becomes the exact map). All finite-width error is a
power series in `1/n`. The **leading 1/n correction** to the activation statistics can be
written down analytically — this is the "effective theory of deep learning" program
(Roberts–Yaida–Hanin; Yaida 2019). The four-point (connected) function at `O(1/n)` propagates
by its own linear recursion alongside the two-point function, and its contribution to the mean
is a closed contraction.

**Why it helps with depth.** The 1/n four-point correction is the object whose recursion has a
**depth-dependent coefficient** — it is the mathematically precise version of "breaks down as
depth grows". Tracking it explicitly (rather than hoping it stays negligible) is the most
principled attack on the exact failure ARC flags. At n=256 the correction is O(1/256)≈0.004
relative *per layer*, but the recursion can make the depth-8 accumulation an order of
magnitude larger — i.e. right in the range between your current core and the 1e-4 RMS you
need. **Cost:** you must carry one more recursive object (the connected 4-pt function, or a
compressed/diagonal proxy). **Risk:** the bookkeeping is the hard part; start with the
diagonal (per-neuron κ₄) version, which overlaps with Approach 2, before attempting the full
tensor.

---

## 6. Attack the *compounding*, not just the per-layer error

Because depth-8 error is accumulated, three structural moves pay off independently of which
closure you use:

- **Weight the budget toward the last layers.** Only the final row is scored. Early-layer
  moments matter only insofar as they feed the final closure, and their influence is
  attenuated by the intervening linear maps. Compute layers 1–5 with a cheap closure and pour
  your expensive closure (saddlepoint, κ₄, exact bivariate) into layers 6–8. This reallocates
  FLOPs to where final-layer MSE is most sensitive.
- **Self-consistent (fixed-point) closure per layer.** Instead of a single forward pass,
  iterate the `(μ,Σ) → ReLU moments → (μ,Σ)` map to a fixed point within a layer when the
  variance estimate is what drives `α`. This removes the "stale variance" bias that a single
  pass leaves and that accumulates over depth.
- **Propagate an error/uncertainty estimate.** Carry a cheap surrogate for the closure error
  itself (e.g. the size of the κ₄ term you dropped) so you know *which* neurons' final
  predictions are least trustworthy — then target them (Approach 7, or residual MC).

---

## 7. Adaptive per-neuron effort by kink geometry

Not all neurons are equally hard. The closure error is governed by how much probability mass
sits near the ReLU kink:

- **|α| large positive** → neuron is nearly linear → `E[ReLU]≈μ`, closure nearly exact.
- **|α| large negative** → neuron nearly dead → `E[ReLU]≈σφ(α)`, closure nearly exact.
- **|α| ≲ 2 (kink-straddling)** → all the non-Gaussian action lives here.

Spend κ₄ terms, saddlepoint, or a few targeted MC samples **only** on the middle band. This is
the single cheapest way to convert FLOPs into final-layer accuracy, and it composes with every
other approach. It also tells you where your residual MC control variate should aim its
samples (variance reduction is wasted on already-near-exact neurons).

---

## 8. Don't discard the control variate — feed it the better core

A more accurate analytic core is *also* a better CV surrogate. Two synergies:

- **Higher ρ for free.** Your affine surrogate is frozen at the Gaussian operating point;
  ρ²≈0.21. A surrogate built from the *cumulant-corrected* prediction (or a per-neuron
  saddlepoint linearisation) correlates more strongly with the true ReLU output, lifting ρ²
  and lowering the CV floor below 6e-6 at no extra sample cost.
- **Analytic core as importance/stratification proposal.** Use the analytic marginal to
  stratify or importance-sample the kink-straddling neurons, concentrating samples where the
  closure is least certain (ties to Approach 7).

Keep the empirical-Bayes blend — but note that as the core improves, the optimal shrinkage
moves toward the analytic estimate, and the blend should track that automatically through your
`τ²/(τ²+v)` weighting.

---

## Recommended order of attack

1. **Exact bivariate ReLU covariance (Approach 1).** Strict upgrade, modest cost, removes a
   depth-amplified bias. Do this first.
2. **κ₄ mean term (Approach 2), with the verified `σ(γ₂/24)(α²−1)φ(α)` line.** Free given a κ₄
   estimate; start with a cheap diagonal/diagram κ₄ before any full contraction.
3. **Swap Edgeworth → saddlepoint or GH-quadrature closure (Approaches 3–4).** Removes the
   series-blowup failure mode; this is the most likely single cause of depth degradation.
4. **Reallocate budget to the last layers + self-consistent variance (Approach 6).** Cheap,
   high-leverage given only the final layer is scored.
5. **Finite-width 1/n correction (Approach 5).** The principled long game; attempt after the
   above, beginning with its diagonal form.
6. **Re-tune the CV surrogate off the improved core (Approach 8)** to mop up residual bias.

A useful internal check throughout: run the depth-2,3,4 sub-networks against a heavy offline MC
reference and watch *where* the per-layer diagnostic error first departs from machine-ish
precision — that layer's closure is your binding constraint, and the diagnostic rows ARC
reports exist precisely to expose it.

---

## References / leads

- ARC, *Estimating the expected output of wide random MLPs more efficiently than sampling*,
  arXiv:2605.05179 (2026); code `alignment-research-center/mlp_cumulant_propagation`. The
  method is cumulant propagation; depth scaling is discussed in their Appendix D, ablations in
  Section 6.4. **Read these two sections first** — they describe the exact degradation you are
  fighting.
- ARC, *Formalizing the presumption of independence* (2022), Appendix D — original cumulant
  propagation.
- Cho & Saul, *Kernel methods for deep learning* (2009) — arc-cosine kernels (exact bivariate
  ReLU moments, Approach 1).
- Roberts, Yaida & Hanin, *The Principles of Deep Learning Theory* (2022); Yaida (2019) —
  finite-width 1/n effective theory (Approach 5).
- Daniels, *Saddlepoint approximations in statistics* (1954); Lugannani & Rice (1980) —
  saddlepoint closure (Approach 3).
- Owen's T function (`scipy.special.owens_t`) — non-centered bivariate normal orthant moments
  (Approach 1, general case).

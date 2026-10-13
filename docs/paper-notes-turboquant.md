# TurboQuant (arXiv:2504.19874 v1) — extracted facts for the talk

Zandieh (Google Research), Daliri (NYU), Hadian (Google DeepMind), Mirrokni (Google Research).
"[my check]" = derived by us, not in the paper.

## Problem (§1.1)
- $Q:\mathbb R^d\to\{0,1\}^B$, $B=b\cdot d$. Randomized allowed; worst-case input; expectation over $Q$'s randomness only.
- $D_{\tt mse}=\mathbb E_Q\|x-Q^{-1}(Q(x))\|_2^2$ (1);  $D_{\tt prod}=\mathbb E_Q|\langle y,x\rangle-\langle y,Q^{-1}(Q(x))\rangle|^2$ (2).
- Unbiased: $\mathbb E_Q\langle y,Q^{-1}(Q(x))\rangle=\langle y,x\rangle$.
- Assume $\|x\|=1$; otherwise store norm in float (not counted in $b$).

## Rotation
"Generate Π by QR decomposition on a random matrix with i.i.d. Normal entries." Then Πx is uniform on $\mathbb S^{d-1}$.

## Algorithm 1 — TurboQuant_mse
```
Setup: random rotation Π ∈ R^{d×d}; codebook c_1..c_{2^b} ∈ [−1,1] minimizing Eq.4
Quant_mse(x):   y ← Π·x;  idx_j ← argmin_k |y_j − c_k|   (b-bit ints)
DeQuant_mse(idx): ỹ_j ← c_{idx_j};  x̃ ← Πᵀ·ỹ
```
## Algorithm 2 — TurboQuant_prod
```
Setup: TurboQuant_mse with bit-width b−1;  S ∈ R^{d×d}, S_ij ~ N(0,1)
Quant_prod(x):  idx ← Quant_mse(x);  r ← x − DeQuant_mse(idx);  qjl ← sign(S·r);  output (idx, qjl, ‖r‖₂)
DeQuant_prod(idx, qjl, γ):  x̃_mse ← DeQuant_mse(idx);  x̃_qjl ← (√(π/2)/d)·γ·Sᵀ·qjl;  output x̃_mse + x̃_qjl
```
b = 1 → MSE stage has 0 bits, r = x, pure QJL.

## Lemma 1
$x$ uniform on $\mathbb S^{d-1}$ ⇒ $x_j\sim f_X(x)=\frac{\Gamma(d/2)}{\sqrt\pi\,\Gamma((d-1)/2)}(1-x^2)^{(d-3)/2}$ on $[-1,1]$; → $\mathcal N(0,1/d)$ in high d.
Proof: area of $(d-1)$-sphere radius $\sqrt{1-x^2}$ / area of unit sphere × $1/\sqrt{1-x^2}$ (tilt).
Near-independence stated informally (Vershynin). Intro typo says N(1,1/d).
MSE upper bound only needs identical marginals + linearity, not independence.

## Eq. 4 (Lloyd–Max)
$\mathcal C(f_X,b)=\min_{c_1\le\dots\le c_{2^b}}\sum_i\int_{(c_{i-1}+c_i)/2}^{(c_i+c_{i+1})/2}|x-c_i|^2f_X(x)\,dx$. Solved once per b.
Paper centroids: b=1: $\pm\sqrt{2/\pi}/\sqrt d$; b=2: $\pm0.453/\sqrt d,\ \pm1.51/\sqrt d$.
Gaussian Lloyd–Max [my check], units $1/\sqrt d$:

| b | positive centroids | MSE·d | entropy |
|---|---|---|---|
| 1 | 0.7979 | 0.3634 | 1.0 |
| 2 | 0.4528, 1.5104 | 0.1175 | 1.911 |
| 3 | 0.2451, 0.7560, 1.3439, 2.1519 | 0.03455 | 2.825 |
| 4 | 0.1284, 0.3880, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326 | 0.00950 | 3.765 |

Entropy coding could save ~5% at b=4; not used (simplicity/speed).

## Theorems
- Thm 1: $D_{\tt mse}\le\frac{\sqrt3\pi}{2}4^{-b}$ (≈2.72·4^{-b}); b=1..4: 0.36, 0.117, 0.03, 0.009. $D_{\tt mse}=d\cdot\mathcal C(f_X,b)$. Panter–Dite for b>4.
- Thm 2: unbiased; $D_{\tt prod}\le\frac{\sqrt3\pi^2\|y\|^2}{d}4^{-b}$; b=1..4: 1.57/d, 0.56/d, 0.18/d, 0.047/d. Proof: $D_{\tt prod}\le\frac{\pi}{2d}\|y\|^2D_{\tt mse}(b-1)$.
- Thm 3 (lower): any randomized Q, ∃ hard x: $D_{\tt mse}\ge4^{-b}$; ∃ unit y: $D_{\tt prod}\ge\frac1d4^{-b}$.
  - Yao's minimax: best randomized alg on worst-case input = best deterministic alg on hardest input distribution ≥ best on uniform-on-sphere input.
  - Shannon lower bound (Lemma 2): $D(B)\ge\frac{d}{2\pi e}2^{(2/d)(h(x)-B)}$; sphere entropy via Stirling → $D\ge2^{-2B/d}=4^{-b}$ (asymptotic in d).
  - Inner product: pigeonhole over $e_j$ ⇒ some coordinate direction has error ≥ $\frac1d4^{-b}$.
- Gap: 1.45 (b=1), 1.88, 2.21, 2.43, → 2.72.
- Bias of TQ_mse at b=1: codebook $\pm\sqrt{2/(\pi d)}$ ⇒ $Q_{\tt mse}(x)=\mathrm{sign}(\Pi x)$ ⇒ $\mathbb E\langle y,\tilde x\rangle=\frac2\pi\langle y,x\rangle$. Bias shrinks with b (empirically).
  [my check] General: $\mathbb E\langle x,\tilde x\rangle=\mathbb E\|\tilde x\|^2=1-D_{\tt mse}$ (orthogonality of conditional mean). Slopes ≈ 0.636, 0.883, 0.965, 0.990.

## QJL restated (Def 1, Lemma 4)
$Q_{\tt qjl}(x)=\mathrm{sign}(Sx)$, $S\in\mathbb R^{d\times d}$; $Q^{-1}_{\tt qjl}(z)=\frac{\sqrt{\pi/2}}{d}S^\top z$. Unbiased; Var ≤ $\frac{\pi}{2d}\|y\|^2$.

## Experiments (single A100)
**§4.1 validation**: DBpedia OpenAI3 d=1536, 100k points, 1k queries.
- Fig 1: histograms of IP error, b=1..4; prod centered at 0 (range ±0.1), mse shifted positive (bias, largest at b=1 ~0.03–0.05).
- Fig 2: b=2, Avg IP = 0.01, 0.06, 0.10, 0.17: prod stays centered/same spread; mse shifts right with avg IP (~0.02 at 0.17).
- Fig 3a: $D_{\tt prod}$ vs b=1..5 (log y): prod lower at b=1,2; **cross near b≈3; mse lower for b≥3**. Bounds $\frac1d4^{-b}$, $\frac{\sqrt3\pi^2}{d}4^{-b}$.
- Fig 3b: $D_{\tt mse}$ ≈ 0.36, 0.117, 0.035, 0.0095, 0.0027 (b=1..5), between $4^{-b}$ and $2.72\cdot4^{-b}$.

**§4.2 Needle-in-a-haystack** (Llama-3.1-8B-Instruct, 4k→104k tokens, compression ratio 0.25):
SnapKV 0.858, PyramidKV 0.895, KIVI 0.981, PolarQuant 0.995, Full-Precision 0.997, TurboQuant 0.997. Bit-width not stated.

**§4.3 LongBench-E** (KV size = bits per channel):

| Model / Method | KV | SingleQA | MultiQA | Summ. | Few shot | Synthetic | Code | Avg |
|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B Full | 16 | 45.29 | 45.16 | 26.55 | 68.38 | 59.54 | 46.28 | 50.06 |
| KIVI | 3 | 43.38 | 37.99 | 27.16 | 68.38 | 59.50 | 44.68 | 48.50 |
| KIVI | 5 | 45.04 | 45.70 | 26.47 | 68.57 | 59.55 | 46.41 | 50.16 |
| PolarQuant | 3.9 | 45.18 | 44.48 | 26.23 | 68.25 | 60.07 | 45.24 | 49.78 |
| TurboQuant | 2.5 | 44.16 | 44.96 | 24.80 | 68.01 | 59.65 | 45.76 | 49.44 |
| TurboQuant | 3.5 | 45.01 | 45.31 | 26.00 | 68.63 | 59.95 | 46.17 | 50.06 |
| Ministral-7B Full | 16 | 47.53 | 49.06 | 26.09 | 66.83 | 53.50 | 47.90 | 49.89 |
| TurboQuant | 2.5 | 48.38 | 49.22 | 24.91 | 66.69 | 53.17 | 46.83 | 49.62 |

Outliers: "splitting channels into outlier and non-outlier sets, and applying two independent instances of TurboQuant to each, allocating higher bit precision to outliers". Paper: "in our 2.5-bit setup, 32 outlier channels are quantized at 3 bits, while the remaining 96 channels use 2 bits, leading to an effective bit precision of (32×3+96×2)/128=2.5." **Paper arithmetic error: (32·3+96·2)/128 = 288/128 = 2.25, not 2.5.** An exact 2.5 would be e.g. 64@3 + 64@2 (ours). Unknown which part (channel counts or the label) is wrong; possibly norms overhead (our guess). 3.5-bit: "a different ratio" — split not given (e.g. 64@4 + 64@3 would give 3.5 — our example only).
Not stated: how outliers are chosen; K vs V treatment; mse vs prod variant; whether norm overhead counts.

**§4.4 Nearest neighbor**: DBpedia OpenAI3 d=1536, 3072 (100k pts, 1k queries), GloVe d=200 (10k queries). Recall@1@k.
Table 2 — quantization time (s), 4-bit:

| Approach | d=200 | d=1536 | d=3072 |
|---|---|---|---|
| Product Quantization | 37.04 | 239.75 | 494.42 |
| RabitQ | 597.25 | 2267.59 | 3957.19 |
| TurboQuant | 0.0007 | 0.0013 | 0.0021 |

Fig 5 recall@1@1 (read off plot):

| | TQ 4b | PQ 4b | RabitQ 4b | TQ 2b | PQ 2b | RabitQ 2b |
|---|---|---|---|---|---|---|
| GloVe d=200 | 0.86 | 0.82 | 0.80 | 0.55 | 0.49 | 0.49 |
| d=1536 | 0.970 | 0.963 | 0.928 | 0.896 | 0.876 | 0.854 |
| d=3072 | 0.972 | 0.962 | 0.940 | 0.903 | 0.888 | 0.866 |
All → 1.0 by k≈4–8 on OpenAI3; GloVe 2-bit reaches ~0.98–0.99 at k=64.

## Accelerator / online claims
- Data-oblivious, online, "apply instantly without needing data-specific tuning or calibrations".
- PQ: k-means codebooks, heavy preprocessing; RabitQ: not vectorizable on GPU.
- Per-vector work: one rotation, per-coordinate nearest centroid lookup; prod adds projection + sign.

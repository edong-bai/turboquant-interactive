# Speaker notes

_Generated from `src/slides/*.js` by `npm run notes`. Press **N** in the deck to see these live._

## 1. TurboQuant, interactively

*`#title` · Why*

Welcome. Two papers by roughly the same group: QJL (2024) and TurboQuant (2025).
The goal is not to present them; it's to rediscover them. Each step will be forced by a failure of the step before.
Background animation: random points on a sphere snapping to TurboQuant's 2-bit grid (in a random rotated frame) and back.
Keys: ↓/space next, ↑ back, N notes, F fullscreen, P dims the chrome.

## 2. The KV cache is the bottleneck

*`#kv-cache` · Why · Motivation*

Left: autoregressive decoding, one token per step (the sentence is illustrative). Each step, the newest token produces a query q (blue), a key k (orange) and a value v (teal). k and v are appended as one new row to the cache. Then q is compared with every cached key, the scores go through a softmax (blue bars = attention weights p_i), and the output is the p-weighted average of the value rows.
The yellow sweep is the point of the slide: to produce ONE token, the GPU streams the ENTIRE cache (every K row and every V row, in every layer and head) from HBM. The QJL paper's framing: decoding is memory-bandwidth bound, the work per step is O(n·d) and the whole KV cache is read for every generated token. So the cache size is both a memory-capacity problem and a speed problem.

Right: the cache-size calculator. bytes = 2 (K and V) × layers × KV heads × head dim × tokens × batch × bits/8.
• Llama-3.1-8B (32 layers, 8 KV heads with grouped-query attention, head dim 128): 128 KiB per token at 16 bits. At a 128k context with batch 1 that is about 17 GB, already more than the roughly 16 GB of FP16 weights (8B params × 2 bytes).
• Llama-2-7B has no GQA (32 KV heads), so its cache is 4× bigger per token: 512 KiB/token.
• Drag the context to 1M, or raise the batch: the cache dwarfs the model.
Bits per number: the compression factor is just 16/b. QJL reports "more than fivefold" KV memory reduction at 3 bits (16/3 = 5.3×, i.e. about 81% less). TurboQuant reports quality neutrality at 3.5 bits per channel on LongBench-E for Llama-3.1-8B (16/3.5 ≈ 4.6×) and only marginal degradation at 2.5 bits (6.4×).
Caveat for the honest version: real systems also store a few overhead numbers (scales, norms) and keep some tokens in full precision; the calculator counts only the payload bits. That overhead is exactly what the next slide is about.

## 3. What a quantizer must do

*`#requirements` · Why · The rules*

Four requirements. Every candidate in the rest of the talk gets scored against these.

(a) Online and data-oblivious. Keys and values are produced one token at a time during decoding; you must compress each one the moment it is produced, with no chance to look at the rest of the data. Left half: the same stream as the right, quantized on arrival against a codebook that was fixed in advance. Right half: k-means / product quantization must first collect a dataset, then run Lloyd iterations to fit centroids, and only then can it encode; when the distribution drifts it has to refit. TurboQuant's nearest-neighbor table makes the cost concrete: at 4 bits, quantization takes 37–494 s for PQ and 597–3957 s for RabitQ, versus 0.0007–0.0021 s for TurboQuant (d = 200 to 3072, single A100). The QJL paper also calls QJL data-oblivious: a sketch with no tuning.

(b) Accelerator-friendly. GPUs are fast when thousands of lanes execute the same instruction in lockstep on different data. A per-coordinate rule (rotate, then look up the nearest of a few scalar levels) is exactly that: every lane does the same thing. Search over a learned codebook, tree, or graph is branchy and sequential; lanes diverge and idle. TurboQuant's paper says RabitQ is not vectorizable on GPU, and PQ needs heavy preprocessing.

(c) Zero overhead. Standard KV quantizers work in small groups and store a scale and a zero-point per group in full precision. If both are FP16 and the group has g numbers, the overhead is 32/g bits per number. Drag g: 32 numbers per group costs +1 bit, 16 costs +2. QJL's intro puts it as "1 or 2 additional bits per quantized number". At b = 3, that is a 33–67% tax on the payload. The goal is a scheme whose only side information is one number per whole vector (its norm), which is negligible for d = 128.

(d) Worst-case guarantee near the limit. We want a bound that holds for EVERY input vector (the expectation is only over the quantizer's own randomness), and we want it close to what is information-theoretically possible. Teaser: later we prove no quantizer, randomized or not, can beat 4^(−b) relative MSE on its worst input. Drag the dot ("your quantizer") below the curve: impossible. TurboQuant lands within a factor of about 2.7 of this line for every input.

## 4. Two errors to minimize

*`#metrics` · Why · The scoreboard*

Drag x (orange, stays on the unit circle) and the query y (blue, anywhere).
The quantizer here is a 2-bit-per-coordinate grid (16 points). x̃ (teal) is the nearest grid point.
MSE: the squared length of the violet error vector. Inner-product error: the gap between the projections of x and x̃ onto y.
Values V are averaged, so they need small MSE. Keys K only meet the query through ⟨q,k⟩, so they need small inner-product error.
Turn on "random grid rotation" + "show 200 draws": the quantizer is now randomized, and the expectation is over that randomness, never over x (x is worst-case).
The mean of the teal cloud (white ring) sits inside the circle: the reconstruction is biased toward 0. Keep that in mind; it comes back as the central problem later.

## 5. Normalize, then round

*`#naive` · Naive · First attempt*

The first thing anyone would try. Store the norm ‖x‖ as one float, divide it out so every coordinate lies in [−1, 1], round each coordinate to a uniform b-bit grid, and multiply the norm back in. Cost: b bits per coordinate plus 16 bits for the norm, i.e. b + 16/d bits per coordinate.

Our grid (the gray lines): step Δ = 2/(2^b − 1), round to the nearest multiple of Δ. For b ≥ 2 that gives 2^b − 1 levels including 0 (one code unused, like symmetric INT-b); 1 bit cannot hold a symmetric zero level, so b = 1 is sign × scale. Either way each coordinate is off by at most Δ/2.

Things to show, in order:
1. Gaussian, d = 8, b = 3: works fine. Now slide d up to 128. The orange bars (coordinates of x/‖x‖) shrink like 1/√d, while the grid stays put. Soon almost every coordinate is below half a step and rounds to 0 ("rounded to 0" readout). Error heads to 100%.
2. Toggle min–max scaling (KIVI-style: store max|x_i| instead of the norm and stretch the grid to [−max, max]). Gaussian is fine again at any d: the grid now adapts to the vector.
3. one spike: fine in both modes. The zeros stay exactly 0; the only error is the spike itself, clamped to the top level (1 − Δ/2): 11% at b = 2, 2% at b = 3, 0.4% at b = 4. (At b = 1 the zeros become ±1 and the error explodes to d − 1: sign quantization has no zero.)
4. spread: in fixed mode it is the worst case (every coordinate is tiny, so everything rounds to 0 once 1/√d < Δ/2, i.e. d > (2^b − 1)²); with min–max it is easy (every coordinate equals ±max, so only the same small clamp error).
5. outlier channel + noise with min–max on: one channel at 10 stretches the grid, and the other channels collapse onto the zero level. Hold that thought for the next slide.
The inset is the same game in d = 2: drag x around the unit circle and watch which grid point it snaps to. With min–max on, the grid rescales with x.
Takeaway: each naive variant has inputs it handles well and inputs it destroys. Nothing here is a guarantee about every input.

## 6. Adversarial inputs break it

*`#naive-fails` · Naive · First attempt*

Same naive rounder as the previous slide (step Δ, round to the nearest multiple; 1 bit = sign). Now we let an adversary choose x.

Left, fixed grid after normalizing (solid red). The adversary's favorite input is the spread vector: every coordinate ±1/√d. Once 1/√d is below half a step, every coordinate rounds to 0, x̃ = 0, and the relative error is exactly 100%. That happens for d > (2^b − 1)², so at b = 4 already at d ≈ 226, and for b = 2 at d = 10. For smaller d the adversary still gets close to d·Δ²/4 by putting d − 1 coordinates just under half a step. The curves are the error achieved by these explicit adversarial vectors, computed live. b = 1 (sign) is even worse: a single spike reconstructs as all ±1, so the error is d − 1 (off the top of the chart).
Dashed red: the min–max (KIVI-style) variant. Its adversary is one big coordinate plus many coordinates just under half a step; the error tends to 100% as d grows, again.
The teal band is where we are going: between 4^(−b) (the lower bound for any quantizer, proved later) and about 2.72 · 4^(−b) (TurboQuant's guarantee). Flat in d, and for every input. Switch b to compare.

Right, min–max scaling with one outlier channel (d = 64, synthetic Gaussian channels plus one channel of strength A). Drag A up: the grid stretches to [−A, A], the step grows like A, and every ordinary channel collapses onto the zero level. The whole-vector relative error looks modest because the outlier dominates the norm, but the information in the other 63 channels is gone (third readout). Real keys have exactly this structure: in QJL's Fig. 2 (Llama-2), deeper layers have a few fixed channels with magnitudes around 12–14 while the rest stay near 1. Layer 0 shows no such structure.

The bound: rounding error per coordinate is at most Δ/2, so ‖x − x̃‖² ≤ d·Δ²/4 with Δ = 2 max|x_i| / (2^b − 1). It scales with d · max², and the adversary controls max. No worst-case guarantee relative to ‖x‖² is possible this way.

Takeaway: the naive quantizer has no worst-case guarantee; its error depends on how x sits relative to the axes. Park this failure: we'll come back to it when TurboQuant is introduced. (Next: a different angle on the problem: what do keys actually need?)

## 7. Keys only appear inside $\langle q,k\rangle$, so JL is enough

*`#jl` · QJL · Look again at attention*

Re-read the attention formula. A key k_i is never used as a vector: it only enters through the single number ⟨q, k_i⟩ (softmax only ever sees those scores). Values are different: the output is literally a weighted sum of value vectors, so values must be reconstructed (MSE). Keys need only inner products.

That weaker requirement is exactly what lets us use JL. We may replace k by ANY sketch f(k) as long as ⟨f(q), f(k)⟩ ≈ ⟨q, k⟩. A random Gaussian projection is such a map, and it comes with useful random properties:
- it preserves inner products (and distances), not coordinates: Sk looks nothing like k, and that's fine, because coordinates are exactly what we said we don't need. (If we had to reconstruct k, as for values, JL with m < d would be useless.)
- S is drawn before any key or query exists (data-oblivious), and the guarantee holds for queries we haven't seen yet: online, as the requirements slide demanded;
- the error is random, not systematic, and the target dimension m depends on how many vectors we handle (n) and the accuracy (ε), not on d.

JL lemma (standard form; the QJL paper cites Johnson–Lindenstrauss and Dasgupta–Gupta but doesn't restate it): for n points, ε ∈ (0,1) and m = O(ε⁻² log n), f(x) = Sx/√m keeps every pairwise squared distance within 1 ± ε w.h.p. Polarization, ⟨x,y⟩ = (‖x‖² + ‖y‖² − ‖x−y‖²)/2, turns that into |⟨f(x),f(y)⟩ − ⟨x,y⟩| ≤ ε‖x‖‖y‖.

LEFT, what attention needs: one query and ten keys in d dimensions (synthetic). Orange outlines are the true scores ⟨q, k_i⟩; teal bars are the scores computed entirely in the projected space, ⟨Sq, Sk_i⟩/m. Drag m: at m = 2 the scores are garbage; by m ≈ 64–128 the teal bars sit on the outlines and the top key is preserved, even at d = 1024. Readouts: worst error in units of ‖q‖‖k‖, whether the top-scoring key is the same, and floats stored per key (m instead of d).

RIGHT, the guarantee for every pair at once: 40 synthetic points (five clusters), all 780 pairs. x = true cosine, y = projected inner product / (‖x‖‖y‖); the teal band is ±ε with ε = 2·√(log n/m) (the constant 2 is ours and illustrative; JL only says O(·)). Red pairs fall outside. The toggle shows the classical squared-distance form on log–log axes.
- Switch d between 64, 256, 1024: the spread at a given m doesn't change. m depends on n and ε, not d.
- "re-draw S": different points move, same overall tightness. The guarantee is over the randomness of S for a fixed (worst-case) point set.

Why it works, one pair at a time (header equation): each row s of S gives one sample (sᵀq)(sᵀk), and E[(sᵀq)(sᵀk)] = qᵀE[ssᵀ]k = ⟨q,k⟩ because E[ssᵀ] = I. So ⟨Sq,Sk⟩/m is an average of m independent unbiased samples: unbiased, with variance (⟨q,k⟩² + ‖q‖²‖k‖²)/m. The errors are random, not systematic, so they average out like 1/√m; a union bound over the n² pairs costs the log n in m. (Equivalently, ‖Sx‖²/m ~ ‖x‖²·χ²_m/m concentrates like √(2/m).)

The catch (bottom line on the left): float JL does save memory when m < d (the "floats per key" readout), but only by dropping rows, and the error sd is ~‖q‖‖k‖/√m. With head dim d = 128 there's little room: m = 32 is a 4× saving with an error of ~0.18–0.25‖q‖‖k‖, large for attention logits. And each row still costs 16 bits.
Better: spend the bits on more rows, not more precision per row. At the same 16m bits per key (our arithmetic from the two variance formulas):
• m floats (JL): Var = (⟨q,k⟩² + ‖q‖²‖k‖²)/m, up to 2‖q‖²‖k‖²/m;
• 16m sign bits (QJL, next slide): Var = ((π/2)‖q‖²‖k‖² − ⟨q,k⟩²)/(16m) ≤ π/(32m)·‖q‖²‖k‖².
Error sd ratio ≈ 3.2× (q ⟂ k) to 7.5× (q ∥ k) in favor of signs. Example at 256 bits per key: 16 floats → sd up to ≈ 0.35‖q‖‖k‖; 256 sign bits → ≈ 0.08‖q‖‖k‖.

Transition: so quantize the projection as hard as possible: keep only the sign of each coordinate of Sk, and use many more rows. That's QJL.

## 8. Keep only the sign: more rows beat more precision

*`#qjl` · QJL · QJL*

The JL slide ended with: float rows are expensive (16 bits each), and to save memory we'd have to drop rows, which costs accuracy. QJL's answer: quantize each row as hard as possible, one bit, the sign, and spend the saved bits on MANY more rows. Def. 3.1: H_S(k) = sign(Sk) ∈ {±1}^m, plus ‖k‖ as one float. The query is not quantized:
  Prod_QJL(q,k) = (√(π/2)/m) · ‖k‖ · ⟨Sq, sign(Sk)⟩.

LEFT, the geometry of one sign bit (2-D is exact: only the projection of s onto span(q,k) matters):
- The orange half-plane is the set of directions s with s·k > 0; every random s (dots, half size) is colored by its bit.
- For the newest s, the sample √(π/2)·‖k‖·sign(s·k)·(s·q) drops onto the number line. "+100" a few times: the running average converges to ⟨q,k⟩. One bit per row and still unbiased (the √(π/2) is explained on the next slide).
- Toggle "sign both sides (SimHash)": also taking sign(Sq) estimates the angle (fraction of disagreeing bits ≈ θ/π, the red wedges), and ‖q‖‖k‖cos(π·fraction) is biased (cos of an unbiased estimate; our exact binomial check: −0.073 at θ = 45°, m = 8). The query is in full precision at decode time anyway, so QJL keeps it: asymmetric.

RIGHT, the point of this slide: same bits per key, two ways to spend them.
- Floats (plain JL): B/16 rows of 16-bit floats. Var = (⟨q,k⟩² + ‖q‖²‖k‖²)/(B/16).
- Sign bits (QJL): B − 16 rows of 1 bit, plus 16 bits for ‖k‖. Var = ((π/2)‖q‖²‖k‖² − ⟨q,k⟩²)/(B − 16) (derived from the QJL paper's moment formula).
- Histograms (live simulation, 1500 trials each): teal = QJL up, violet = floats down, same x-axis. Both are centered on the truth (both unbiased); the sign version is several times narrower. Readout: measured sd ratio, ≈ 3–7× depending on the angle between q and k (≈ 3.2× at 90°, more when q is near k). Change "bits per key".
- Bottom-right curve (theory): error sd vs bits per key, log–log. Both fall like 1/√B (parallel lines: same rate, better constant for signs). Reference: storing k itself in FP16 at d = 128 costs 2048 bits and is exact.

Takeaway: a coarse value on many random rows beats a precise value on a few. And it needs no per-block scale or zero point, just one norm per key ("zero overhead" in the QJL title). The catch, for later: error still only falls like 1/√bits; TurboQuant will get 2^(−b).

## 9. Why one bit stays unbiased

*`#qjl-why` · QJL · QJL*

The geometric proof of Lemma 3.2 (QJL is unbiased), in one picture.

Split the query along the key: q = (⟨q,k⟩/‖k‖²)·k + q⊥, where q⊥ is orthogonal to k (blue dashed = parallel part, violet = q⊥). Drag q and k to see the decomposition move.

One row of the estimator is sign(sᵀk)·sᵀq = (⟨q,k⟩/‖k‖²)·|sᵀk| + sign(sᵀk)·sᵀq⊥.
- Second term: for Gaussian s, sᵀq⊥ and sᵀk are jointly Gaussian and uncorrelated (q⊥ ⊥ k), hence independent. The sign of one carries no information about the other, so the term averages to 0. Left histogram: symmetric about 0, running mean → 0.
- First term: sᵀk ~ N(0, ‖k‖²), so |sᵀk| is half-normal with mean √(2/π)·‖k‖. Right histogram with the exact half-normal density (white curve) and its mean.
So E[sign(sᵀk)·sᵀq] = √(2/π)·⟨q,k⟩/‖k‖. Multiplying by √(π/2)·‖k‖ gives exactly ⟨q,k⟩. That is the whole reason for the √(π/2) constant and for storing ‖k‖.

The error. Each row X = √(π/2)‖k‖·sign(sᵀk)·sᵀq has E[X²] = (π/2)‖k‖²·E[(sᵀq)²] = (π/2)‖q‖²‖k‖², because sign² = 1. So Var[X] = (π/2)‖q‖²‖k‖² − ⟨q,k⟩², and averaging m rows divides by m. This variance formula is derived by us from the paper's moment computation; the paper doesn't print it in this form. It is at most (π/(2m))·‖q‖²‖k‖².

Plot (top right): log–log, variance vs m for unit q, k at the angle you set in the sandbox. Dots are a live simulation (300 trials per m); lines are the formulas. Teal = QJL with 1 bit per row; yellow = unquantized JL with a float per row, variance (1 + cos²θ)/m; dashed = the π/(2m) bound. Both have slope −1. Quantizing each row all the way down to 1 bit costs at most a constant: at θ = 90° the ratio is exactly π/2; when q is near k, QJL can even beat float JL because the stored norm ‖k‖ is exact.

Lemma 3.5 (concentration): if m ≥ (4/3)·(1+ε)/ε²·log(2/δ), then |Prod_QJL − ⟨q,k⟩| ≤ ε‖q‖‖k‖ with probability ≥ 1 − δ. (Note: the paper's printed tail bound is missing a minus sign in the exponent; it's a typo, the intended bound is exp(−·).)
Theorem 3.6: if all ‖k_i‖, ‖q‖ ≤ r and m ≥ 2r²ε⁻² log n, all softmax scores are within a (1 ± 3ε) factor w.h.p. So m ≈ ε⁻² log n bits per key, independent of the head dimension d.

The catch, and the cliffhanger: error² ∝ 1/m, and m is the number of bits. Halving the error costs 4× the bits. That's polynomial decay. Information theory allows much better: can the error fall exponentially in the number of bits (like 4^(−b))? That's what TurboQuant does next.

## 10. Why project before taking the sign?

*`#why-project` · QJL · Why randomness helps*

Bridge from QJL to TurboQuant. QJL = project with a random S, then keep signs. A natural question: why not skip the projection and just keep the signs of k itself, i.e. k̂ = (‖k‖/√d)·sign(k)? Same bit budget. Answer: the sign throws away magnitudes, and only after the projection are those magnitudes predictable.

Concrete example (default state): k = (10, 0.1), q = (0, 1). True ⟨q,k⟩ = 0.1.
Raw signs: sign(k) = (+1, +1). Even rescaled to keep the norm (‖k‖/√2 ≈ 7.07 per coordinate), the estimate is ≈ 7.07, off by ~70×. The big channel's 10 and the small channel's 0.1 became indistinguishable. In 2-D there are only 4 representable directions (the diagonals); drag k around and the red arrow just snaps between them.
And it's wrong the same way every time: rotate q (right card 2) — the red curve is a fixed, wrong function of the query. Nothing averages it away.

Project first with a random Gaussian S, then take signs (this is QJL, from the previous slides). Three things change:

1. The information lost by the sign becomes recoverable. Every coordinate of Sk is N(0, ‖k‖²) whatever k looked like, so the average magnitude the sign throws away is known exactly: E|sᵀk| = ‖k‖·√(2/π). That's why the estimator just multiplies by √(π/2) and the stored norm, and why no per-block scales or zero points are needed (card 1: raw k is two spikes; Sk/‖k‖ is always a standard normal). The QJL authors describe this as the JL preconditioning letting them quantize to one bit without normalizing.

2. Systematic error becomes random error that averages out. Raw quantization is deterministic: a given key always has the same error, aligned with the same query directions. With a random S each row is an independent estimate that is correct on average, so errors cancel as rows are added: variance ∝ 1/m, and the guarantee holds for any q and k, not just well-behaved data (card 2: teal = one draw of S, scattered around the true curve; re-draw S).

3. Energy is spread evenly. A key with almost all its weight in one channel becomes, after projection, a vector whose coordinates are all equally informative, so each sign bit carries about the same amount of information instead of most bits encoding near-zero channels (card 3).

Bottom histogram: many independent draws of S (300–1500 depending on m). QJL estimates center on 0.1 (mean readout), with standard deviation ≈ √(((π/2)‖q‖²‖k‖² − ⟨q,k⟩²)/m) ≈ 0.78 at m = 256 — not zero, but random noise around the right answer that shrinks with m. The red line (raw signs, 7.07) never moves.

Transition — the same idea fixes our naive quantizer. Points 1–3 are exactly what the naive "normalize, then round" slides were missing: there, the input's coordinates were whatever the data made them (a spread vector rounding to 0, an outlier channel crushing the rest), so errors were systematic and input-dependent. Randomize first and the coordinates follow a KNOWN distribution whatever the input was, the error becomes random, and energy is spread evenly. Next slide: do this with a random rotation (keeps lengths, so we can still reconstruct) instead of a projection: what if no input could be adversarial?

## 11. What if no input could be adversarial?

*`#rotation` · Rotation · Back to the failure*

Open with the question parked on the naive-fails slide: what if no input could be adversarial? The previous slide gave the answer in QJL's setting: a random projection makes every coordinate N(0, ‖k‖²) whatever k was, so the sign's lost magnitude is known, errors are random, and energy is spread evenly. Here we want the same thing but we must also RECONSTRUCT x (values need MSE), so instead of a Gaussian projection we use a random ROTATION: it preserves lengths and is invertible.

Recall the two failures of "normalize then round": a spread vector (every coordinate ±1/√d) falls between the grid levels and rounds to 0; an outlier channel sets the scale and crushes everything else. Both are adversarial INPUTS. We can't choose the input. But we can choose a random rotation Π, once, and quantize Πx instead of x.

What to do:
• Pick a preset (spike, spread, outlier + noise, Gaussian). The orange bars are the 64 coordinates of the unit vector x; dashed lines are the quantizer's levels; teal ticks are where each coordinate rounds to; violet is the per-coordinate error.
• Press "apply Π": the bars morph into Πx. Whatever you started with, Πx looks the same: 64 small coordinates of size about ±1/√d = ±0.125, Gaussian-looking. Re-roll Π: a different but statistically identical picture.
• Bottom left: the SAME naive quantizer's squared error ‖x − x̃‖² for each preset, before (red) and after (teal) rotation. The teal whisker is the range over 64 independent random Π. After rotation, the error no longer depends on which input you fed in. Worst case = average case.
• Switch the range between "fixed [−1,1]" and "per-vector max". With the fixed range, rotation makes everything uniformly bad (all coordinates are ~0.125, smaller than half a step, so most round to 0). That's not a bug in the argument: rotation only makes the input distribution KNOWN. The next slides find what that distribution is and design the codebook for it.

Why it works (equations): for every fixed unit x, Πx is uniform on the sphere, because a Haar-random rotation maps any fixed point to a uniform point. Rotation preserves lengths, so ‖Πx − ỹ‖ = ‖x − Πᵀỹ‖: quantizing Πx with a fixed grid is exactly the same as quantizing x with a randomly rotated grid (inset, bottom right: left half rotates the input, right half rotates the grid; the violet errors have the same length).

"Randomize the algorithm ⇔ randomize the input": this is Yao-flavored intuition. A randomized algorithm on a worst-case input behaves like a deterministic algorithm on a random input. Yao's minimax principle is used formally later, on the 'bounds' slide, for the lower bound.

Practicalities (paper: "Generate Π by QR decomposition on a random matrix with i.i.d. Normal entries"): Π is drawn once and shared by every vector, so storing it is a one-time cost, and applying it is one d×d matrix–vector product per vector. A consequence of the math (not something the paper spells out): for attention, ⟨q, k⟩ = ⟨Πq, Πk⟩, so we can rotate the query once instead of de-rotating every key.

The presets are synthetic (the outlier channel's size, 12× the noise level, is illustrative). Naive quantizer used here: 2^b − 1 evenly spaced levels on [−s, s] including 0.

## 12. $d=3$: every band holds the same mass

*`#sphere-d3` · Rotation · Where do coordinates land?*

After a random rotation, Πx is a uniform point on the sphere. So the question "what does one coordinate of Πx look like?" is: pick a uniform point on S^{d−1}, look at one coordinate x_j. Start with d = 3, where we can see it.

Left: 1500 uniform random points on the 2-sphere (live sample, drag to orbit). The yellow band is all points whose height (one coordinate, z = x_j) lies in [x − dx/2, x + dx/2]. Middle: histogram of the height of all 1500 points, drawn sideways so "up" means the same thing in every panel. It is flat at 1/2: every band of the same width holds the same fraction of points, whether it is around the equator or near a pole. That is Archimedes' hat-box theorem (the sphere's area between two heights equals the area of the circumscribed cylinder between those heights).

Drag the x slider (or drag in the middle/right panels) from the equator toward a pole. Two things change, and they cancel:
1. The slice shrinks. The band is a ring of radius r = √(1 − x²), so its circumference is 2πr: proportional to r. Near the pole r → 0.
2. The band widens along the surface. Near the pole the sphere is tilted almost flat, so a height step dx covers more surface: arc length ds = dx / r.
Product: 2πr · (1/r) = 2π, divided by the total area 4π gives 1/2, flat.

Right panel, the "Pythagorean correction": the sphere cut through the x_j axis is a circle (up = x_j, sideways = r). The band is the yellow strip; it meets the circle in a short arc. Zoom box: that arc is, to first order, the hypotenuse ds of a small right triangle with legs dx (vertical) and |dr| (horizontal). The tangent is perpendicular to the radius, so the small triangle is similar to the big one (legs r and x, hypotenuse 1), rotated by 90°: dx corresponds to r, |dr| to x, ds to 1. Hence ds/dx = 1/r. Shrink dx with the slider: the measured ratio (chord/dx) converges to 1/r.

General d (the equation): the slice at height x is a (d−2)-sphere of radius r, with area ∝ r^{d−2}; the tilt factor is still 1/r; divide by the total area. That gives TurboQuant's Lemma 1: f_X(x) ∝ (1 − x²)^{(d−3)/2}, with the Gamma-function constant making it integrate to 1. At d = 3 the exponent is 0: flat. For d > 3 the slice factor wins and the mass piles up at the equator, which is where we go next.

## 13. $d=4$: a flipbook of spheres

*`#sphere-d4` · Rotation · Where do coordinates land?*

We can't draw S³ (the unit sphere in R⁴), but we can flip through it like a flipbook, one height at a time.

30,000 uniform random points on S³ (live sample). Choose a height t for the 4th coordinate. Take the thin slab of points with |x₄ − t| < δ and plot them by their first three coordinates (x₁, x₂, x₃). They form an ordinary 2-sphere of radius √(1 − t²) (yellow), sitting inside the dashed unit sphere. At t = 0 the slice is the full unit sphere; as t → ±1 it shrinks to a point. It plays automatically; dragging the slider or the histogram pauses it.

Same two effects as in d = 3:
• slice: the slice is now a 2-sphere, so its area is 4πr², proportional to r² (one more power of r than the circle in d = 3);
• tilt: still 1/r, exactly the same Pythagorean correction as before (it only involves the one coordinate we're slicing along and the radial direction).
Product: 4πr² · (1/r) / |S³| = 4πr / (2π²) = (2/π)·r = (2/π)√(1 − x²). That's the semicircle law (teal) in the histogram; the dashed line is the flat d = 3 density for comparison. The cancellation is gone: the extra r tips the balance toward the equator.

Count readout: the number of points in the slab is about N·2δ·f(t). Watch it drop as the slice sphere shrinks toward the poles.

Every extra dimension adds one more factor of r to the slice. That is the (1 − x²)^{(d−3)/2} of Lemma 1; at large d this is a Gaussian, next slide.

## 14. High dimensions: a Gaussian, nearly independent

*`#concentration` · Rotation · Concentration of measure*

Now crank up the dimension. Left: the density of one coordinate of a uniform point on S^{d−1}. Teal is the exact formula (TurboQuant Lemma 1); orange bars are a live Monte-Carlo histogram (20,000 points per d, sampled when you first visit that d, then cached); the dashed line is the Gaussian N(0, 1/d).

What to drag (d slider is logarithmic, 2 to 1024; or hit "sweep"):
• d = 2: the circle. One coordinate is cos θ: the arcsine density 1/(π√(1 − x²)), piled up at ±1.
• d = 3: flat, the hat-box theorem from two slides ago.
• d = 4: the semicircle from the flipbook.
• d = 16, 128, 1024: a bell, narrower and narrower (standard deviation 1/√d) and indistinguishable from N(0, 1/d). Without rescaling the curve becomes a spike: that IS concentration of measure (almost all of the sphere is near any equator). Toggle "rescale by √d" to zoom in: √d·y converges to N(0, 1).
Readouts: d·Var(y₁) = 1 for every d (since the d squared coordinates sum to 1). Excess kurtosis of y₁ is exactly −6/(d+2) (from E y₁⁴ = 3/(d(d+2))), → 0, the Gaussian value.

Right: near-independence. Scatter of the pair (y₁, y₂) for many random points.
• d = 2: a ring. y₂ is determined by y₁ (up to sign): y₁² + y₂² = 1. Completely dependent.
• d = 3: a filled disk (dense near the rim). Still dependent: |y₁| large forces |y₂| small.
• large d: a round Gaussian blob, i.e., two nearly independent N(0, 1/d).
Readouts: corr(y₁, y₂) is 0 for every d (symmetry), even at d = 2 where they are totally dependent, so plain correlation is the wrong test. corr(y₁², y₂²) catches the dependence. Exact value, derived from E y₁²y₂² = 1/(d(d+2)), E y₁⁴ = 3/(d(d+2)), E y₁² = 1/d: Cov = −2/(d²(d+2)), Var = 2(d−1)/(d²(d+2)), so corr = −1/(d−1). Shown next to the empirical estimate: −1 at d = 2, → 0 as d grows. (The paper states near-independence informally, citing Vershynin. Its MSE upper bound only needs the identical marginals plus linearity of expectation, not independence.)

The takeaway is the whole design: after the random rotation, every coordinate of every input has the same known density, depending only on d. So one fixed scalar codebook (computed once, offline) is right for every coordinate of every vector. No per-block scale or zero point to store, which was the overhead QJL complained about.

Sampling detail: instead of drawing d Gaussians and normalizing, we draw g₁, g₂ and the rest of ‖g‖² as a chi-square with d−2 degrees of freedom. That is exactly the same distribution, just cheaper.

## 15. Lloyd–Max: k-means on a known density

*`#lloyd-max` · TurboQuant · Optimal scalar codebook*

We now know exactly what every rotated coordinate looks like: the density f_X for this d. So the best b-bit scalar quantizer for it is a 1-D k-means problem on a KNOWN density, Eq. 4 in the paper. That's the Lloyd–Max algorithm (Lloyd 1957/1982, Max 1960).

The plot: the x-axis is in units of 1/√d (so the picture looks the same for every large d). Gray curve: the density. Teal: the 2^b centroids (reconstruction levels). Dashed: the bucket boundaries.

Lloyd's two alternating steps, each of which can only lower the cost:
1. Fix the centroids: each x goes to its nearest centroid, so boundaries sit at midpoints between neighbors. Toggle "why midpoints?": the cost of sending x to c_i is the parabola (x − c_i)²; two neighboring parabolas cross exactly halfway between their vertices, and the lower envelope switches there.
2. Fix the buckets: the best single point for a bucket is its conditional mean (center of mass of f over that bucket).

What to do: start from the uniform grid (evenly spaced cells across ±3/√d; that's our choice of "naive" grid, not optimized). Press Step a few times: the centroids slide inward and crowd toward the peak, where the mass is (spacing becomes uneven: tight in the middle, wide in the tails). Press Run to converge. Toggle "show integrand": orange is |x − c_i|² f(x) inside each bucket (in √d units, so its total area IS the readout d·C(f_X, b); for larger b it is drawn vertically exaggerated by a fixed power of 2, shown in the label, so it stays visible). Watch the orange area shrink as you step.

Readouts: d·C is the TurboQuant_mse distortion for a unit vector (Theorem 1: D_mse = d·C(f_X, b)). At large d it approaches 0.3634, 0.1175, 0.03455, 0.00950 for b = 1..4 (Gaussian Lloyd–Max, our check; the paper rounds to 0.36, 0.117, 0.03, 0.009). Compare with the uniform grid with the same number of levels and with 4^{−b}, the information-theoretic lower bound (Theorem 3), and the paper's upper bound (√3·π/2)·4^{−b} ≈ 2.72·4^{−b}. The gap to the lower bound is at most ≈ 2.7×, and only 1.45× at b = 1.

Centroids ×√d: compare with the paper: b = 1 gives ±√(2/π)/√d ≈ ±0.798/√d (the mean of |N(0,1)|); b = 2 gives ±0.453/√d, ±1.51/√d. Small d differs (try d = 3: the density is flat on [−1, 1], so the optimum is a uniform grid on [−1, 1]).

The key systems point: this is solved ONCE per (d, b), offline, from the formula. No data, no calibration, no per-block statistics. In deployment it's a lookup table of 2^b numbers.

## 16. TurboQuant$_{\text{mse}}$

*`#tq-mse` · TurboQuant · Algorithm 1*

Put the last three slides together and you have the whole algorithm. Rotate, then quantize each coordinate independently with the Lloyd–Max codebook for the known coordinate density, then rotate back. No data-dependent tuning, no per-block scale or zero point: the codebook depends only on (d, b) and is computed once.

LEFT, the 2-D picture. The faint tilted axes are the rows of Π, i.e. the rotated frame. Dropping perpendiculars from x onto those axes gives the coordinates y = Πx (dashed orange). Each coordinate snaps to its nearest centroid (teal ticks on the axes; the codebook here is the d = 2 Lloyd–Max codebook, b bits per axis), which picks a point of the rotated product grid: ỹ. Reading that grid point back in the original frame is x̃ = Πᵀỹ (teal). The violet arrow is the error x − x̃.
- Drag x around the circle: the grid point jumps cell to cell.
- "re-roll Π": a different random frame; the same x lands in a different cell. The guarantee is in expectation over Π, for every fixed x (worst case over x).
- b slider: 1 → 3 bits per coordinate; the grid gets finer, the error shrinks about 4× per bit.
- "spin Π": sweeps the frame continuously, showing that what x̃ is depends only on where x falls relative to the random frame.
In 2-D this is a crude picture (two coordinates are far from independent). The point of the earlier slides is that in high d the coordinates of Πx are nearly i.i.d. ~ N(0, 1/d), so a product of scalar codebooks is close to optimal.

RIGHT, the measurement: D_mse vs b on a log scale. Teal dots are a live simulation at d = 128 (640 random unit vectors, one fixed Π, Lloyd–Max codebooks for the exact Beta-like coordinate density). Yellow squares are the paper's measured values (Fig. 3b, d = 1536: 0.36, 0.117, 0.035, 0.0095, 0.0027). The simulation gives ≈ 0.360, 0.116, 0.034, 0.0094, 0.0024 (the exact codebook cost d·C(f_X, b) at d = 128 is 0.361, 0.116, 0.034, 0.0093, 0.0025): the same numbers as the paper's d = 1536, because D_mse depends on d only through tiny finite-d corrections (the coordinate density is ≈ N(0, 1/d) and the cost scales out).
The shaded band is between the lower bound 4^{-b} (any quantizer, Theorem 3, next slides) and the Theorem 1 upper bound (√3π/2)·4^{-b} ≈ 2.72·4^{-b}. Every bit buys a factor of 4: the right exponential rate.

Theorem 1 text quotes 0.36, 0.117, 0.03, 0.009 for b = 1..4. For b ≤ 4 these are the numerically solved Lloyd–Max costs; the closed-form bound for large b comes from Panter–Dite high-resolution analysis: C ≈ (1/12)(∫f^{1/3})³ 4^{-b} = (√3π/2)·4^{-b}/d for a Gaussian.

Note the proof of the upper bound only needs that each coordinate of Πx has the known marginal and linearity of expectation; independence is not needed for MSE.

## 17. MSE-optimal shrinks inner products

*`#mse-bias` · TurboQuant · The catch*

TurboQuant_mse is near-optimal for reconstruction, but it is a biased inner-product estimator: on average it underestimates every inner product by a constant factor.

LEFT: live simulation at d = 128. 640 random unit vectors x, each paired with 6 unit queries y = cosθ·x + sinθ·u (u ⟂ x, θ uniform), so ⟨y,x⟩ spans [−1, 1]. Horizontal: the true ⟨y,x⟩; vertical: ⟨y,x̃_mse⟩. The dashed diagonal is y = x; the red line is the least-squares slope through the origin. Pick b:
- b = 1: slope ≈ 0.64 (2/π = 0.637)
- b = 2: ≈ 0.89
- b = 3: ≈ 0.97
- b = 4: ≈ 0.99
(measured values are shown live; they match 1 − D_mse to the third digit.)

Why exactly 1 − D_mse? Middle-right picture. Each coordinate is reconstructed by its cell's conditional mean (that is the Lloyd–Max centroid condition). Conditional means have orthogonal errors: E[(y_j − c(y_j))·c(y_j)] = 0. Summing over coordinates and rotating back: E⟨x − x̃, x̃⟩ = 0, so E⟨x, x̃⟩ = E‖x̃‖² and then D_mse = 1 − 2E⟨x,x̃⟩ + E‖x̃‖² = 1 − E‖x̃‖². In the picture: x̃ sits on the Thales circle over [0, x] (right angle at x̃), with ‖x̃‖² = 1 − D. Its projection on x, which by rotational symmetry is E[x̃], is (1 − D)·x. Hence E⟨y, x̃⟩ = (1 − D_mse)·⟨y, x⟩ for every y. The shrinkage is not a bug of the codebook; it is what any MSE-optimal (conditional-mean) reconstruction does: when uncertain, shrink toward the mean.

The paper's b = 1 calculation (Section 3.2): the 1-bit codebook is ±√(2/(πd)), so Q(x) = sign(Πx) and x̃ = √(2/(πd))·Πᵀsign(Πx). This is QJL with S replaced by a rotation and no Gaussian projection on the query: E⟨y, x̃⟩ = (2/π)⟨y, x⟩. So 36% of every inner product is lost at 1 bit.

Bottom-right: why it matters for attention. Shrinking every logit by the slope is like raising the softmax temperature by 1/slope: the attention weights flatten. The logits here are illustrative (not from the paper). At b = 1 the top weight drops visibly; at b = 4 the effect is ~1%. Only the expected logits are shown; the quantization noise is on top.

Bias shrinks with b (the paper says so empirically; our 1 − D_mse identity makes it exact). Next slide: remove the bias entirely.

## 18. Spend the last bit on QJL of the residual

*`#tq-prod` · TurboQuant · Algorithm 2*

Fixing the bias: spend b − 1 bits on TurboQuant_mse, then the last bit per coordinate on a 1-bit QJL sketch of what is left over, the residual r = x − x̃_mse. Store idx, the d sign bits sign(S·r), and one float γ = ‖r‖. Dequantize as x̃_mse plus the QJL reconstruction √(π/2)/d · γ · Sᵀ·qjl.

Why it is unbiased: QJL is an unbiased inner-product estimator for any vector, so conditionally on the mse stage, E⟨y, x̃⟩ = ⟨y, x̃_mse⟩ + ⟨y, r⟩ = ⟨y, x⟩. The MSE stage's shrinkage is exactly what r contains, and QJL puts it back on average.
Why it is accurate: QJL's error scales with the length of what it encodes. Its variance is ≤ (π/2d)·‖y‖²·‖r‖², and E‖r‖² = D_mse(b − 1), which is already small. That gives D_prod ≤ (π/2d)·‖y‖²·D_mse(b − 1), and with Theorem 1 at b − 1 bits, Theorem 2: D_prod ≤ (√3π²/d)·‖y‖²·4^{-b}. The paper's numbers: d·D_prod ≈ 1.57, 0.56, 0.18, 0.047 for b = 1..4.
b = 1: the mse stage gets 0 bits, x̃_mse = 0, r = x: TurboQuant_prod is exactly QJL with m = d.

MIDDLE (conceptual, d = 2): the grid is the (b − 1)-bit rotated product grid. Teal x̃_mse is the snapped point; violet r is the short residual. Each faint dot is x̃_mse + QJL(r) for a fresh Gaussian S (m rows). The cloud is noisy, but its mean (white ring) lands on x: unbiased. The spread scales with ‖r‖: drag x to a point near a grid point and the cloud collapses; pick b = 1 and the cloud is as big as x itself (pure QJL). m = more sign bits ⇒ tighter cloud (variance ∝ 1/m). In TurboQuant m = d.
- Drag x. Change b (1–4: grid has b − 1 bits per axis). Re-draw S.

RIGHT, live simulation at d = 128 (same 640 vectors and query pairs as the previous slide; one Π, one d×d Gaussian S): the teal scatter is ⟨y, x̃_prod⟩ vs ⟨y, x⟩, with slope ≈ 1.00 at every b (measured: ≈ 0.997, 0.997, 1.000, 0.999 for b = 1..4), compared with the red TurboQuant_mse cloud at the same total b (slope 0.64, 0.89, 0.97, 0.99). Below: histograms of the inner-product error ⟨y,x⟩ − ⟨y,x̃⟩ for queries with ⟨y,x⟩ = 0.5, like the paper's Fig. 1: mse is shifted right by ≈ 0.5·D_mse(b), prod is centered on 0 but wider. That trade (no bias, more variance) is the subject of two slides from now.

## 19. Within a small constant of any quantizer

*`#bounds` · TurboQuant · How good is it?*

Theorem 3 (lower bound): for ANY randomized quantizer using b bits per coordinate there is a hard unit vector x with D_mse ≥ 4^{-b}, and a hard unit query y with D_prod ≥ (1/d)·4^{-b}. TurboQuant is within a factor √3π/2 ≈ 2.7 of that, at every b.

LEFT, why it costs 4^{-b}: intuition by counting volume. b·d bits = 2^{bd} cells. If each cell had radius ρ, they cover volume ~ 2^{bd}·ρ^d (in units of the sphere's own "volume"), which must be ≳ 1, so ρ ≳ 2^{-b}, and the typical squared error ≳ ρ² = 4^{-b}. The real proof uses the Shannon lower bound D ≥ (d/2πe)·2^{(2/d)(h(x) − B)} with the differential entropy of the uniform distribution on the sphere computed via Stirling; it is asymptotic in d. (Proof detail, not on screen: the bound must hold for randomized quantizers too. The paper uses Yao's minimax principle: the best randomized quantizer's worst-case error is at least the best deterministic quantizer's error on the hardest input distribution, which is at least its error on x uniform on the sphere. So it suffices to bound the uniform-on-sphere case.) For D_prod: the d coordinate directions e_j split the MSE (Σ_j E⟨e_j, x − x̃⟩² = D_mse), so by pigeonhole some axis carries ≥ D_mse/d ≥ (1/d)·4^{-b}.
The plot: TurboQuant_mse (yellow: the paper's Theorem 1 numbers 0.36, 0.117, 0.03, 0.009; teal: our live d = 128 simulation) inside the band between 4^{-b} and 2.72·4^{-b}. Pick b to read the gap: ratio 1.45, 1.88, 2.21, 2.43 → 2.72 (paper); our d = 128 simulation gives 1.44, 1.86, 2.18, 2.40, 2.49 (b = 5). In bits per coordinate (log₄ of the ratio): 0.27, 0.45, 0.57, 0.64 → 0.72. So TurboQuant uses at most ~0.7 extra bits per coordinate compared to an impossible ideal quantizer.

RIGHT, why TurboQuant beats QJL at the same bit budget. QJL with m = b·d sign bits has D_prod ≤ π/(2bd): the error falls only like 1/b (polynomial). TurboQuant_prod with b bits per coordinate has D_prod ≤ (√3π²/d)·4^{-b}: exponential in b. The plot shows d·D_prod. At b = 1 they are the same algorithm (TurboQuant_prod with 1 bit is QJL with m = d). Live simulation at d = 128 (dots; query ⟂ x, which is QJL's worst case): QJL measured ≈ 1.67, 0.82, 0.53, 0.39, 0.32 vs theory π/(2b) = 1.57, 0.79, 0.52, 0.39, 0.31; TurboQuant_prod measured ≈ 1.56, 0.56, 0.17, 0.052, 0.015 (≈ (π/2)·D_mse(b−1); paper Thm 2 text: 1.57, 0.56, 0.18, 0.047). Same bits, 20× less error at b = 5. The reason: QJL spends every bit on 1-bit sketches of the whole vector; TurboQuant spends them on a fine scalar grid and keeps only one sketch for the tiny residual.

Simulation noise: about ±5% on the QJL dots (640 vectors, one query each).

## 20. When to drop the correction

*`#mse-vs-prod` · TurboQuant · Bias vs variance*

Two estimators of ⟨y, x⟩ with the same b bits per coordinate: TurboQuant_mse (biased, small spread) and TurboQuant_prod (unbiased, larger spread). Which has the smaller inner-product error D_prod = E|⟨y,x⟩ − ⟨y,x̃⟩|²? It depends on b and on how large the inner product is.

Paper Fig. 3a (DBpedia/OpenAI3 embeddings, d = 1536): prod has the lower D_prod at b = 1, 2; the two cross near b ≈ 3; mse is lower for b ≥ 3.

TOP-LEFT: live simulation at d = 128 (same 640 vectors as before, 6 queries each), D_prod vs b on a log scale, for unit queries y with a fixed cosine to x (slider). Dashed lines: the paper's bounds for prod, (1/d)·4^{-b} and (√3π²/d)·4^{-b}.
- Slider at 0 (y ⟂ x, like a random query): mse wins at every b. There is no bias to remove, and prod pays variance for nothing.
- Slide to ≈ 0.6: prod wins at b = 1, 2, mse wins from b = 3 on — the paper's Fig. 3a pattern. (Real embedding inner products are smaller than 0.6, but the paper's d = 1536 makes variances 12× smaller than ours, so the crossover happens at smaller inner products there.)
- Slide to 0.9: prod's advantage extends to b = 3.
(Measured winners: cos 0 → mse at all b; 0.3 → prod only at b = 1; 0.6 → prod at b = 1, 2; 0.9 → prod at b = 1–3.)

RIGHT, the decomposition for the selected b: mse error = bias² + variance; prod error = variance only (its bias is zero up to simulation noise).
- mse bias ≈ D_mse(b)·⟨y,x⟩ (the shrinkage from the mse-bias slide), so bias² ∝ ⟨y,x⟩²·16^{-b}: it dies fast with b.
- prod variance ≈ (π/2d)·D_mse(b−1). Since D_mse(b−1) ≈ 4·D_mse(b), that is ≈ 2π·D_mse(b)/d, about 6× the mse quantizer's own spread ≈ D_mse(b)/d. That's the price of spending one of the b bits on 1-bit QJL instead of a finer grid (measured variance ratio at ⟨y,x⟩ = 0: 5.0–6.5×). The yellow ticks are these simple formulas; the bars are measured.
So prod wins only when the mse bias² exceeds ~5× its variance: low b, large inner products.

BOTTOM-LEFT, like the paper's Fig. 2 (b = 2, average inner product 0.01 → 0.17 there): error histograms at b = 2 as ⟨y,x⟩ grows from 0.05 to 0.8. Prod (teal) stays centered at 0 with the same spread; mse (red) is narrower but slides to the right, proportionally to ⟨y,x⟩.

Takeaway: unbiasedness matters most at low bit-width and when the inner products that matter (the large attention logits) are large. At 3+ bits the MSE variant is simply more accurate. Caveat: the paper does not state which variant its KV-cache and nearest-neighbor experiments use.

## 21. Outlier channels get more bits

*`#outliers` · Practice · In the KV cache*

Top-left heatmap: key magnitudes |k| for 48 tokens × 128 channels. It is ILLUSTRATIVE (synthetic), drawn after QJL Fig. 2 (Llama-2 key magnitudes). Toggle the layer: in an early layer (QJL shows layer 0) there is no outlier structure and every magnitude stays below about 2.5. In deep layers (QJL shows layers 15 and 31) a few fixed channels, "approximately four" per the Fig. 2 caption, sit at about 12–14 for every token, and the rest stay around 1.
Why this matters: QJL's and TurboQuant's errors scale with the vector's norm. TurboQuant_mse error is about D_mse·‖k‖², and QJL's inner-product variance is at most π/(2m)·‖q‖²‖k‖². Four channels at ~13 hold most of ‖k‖² (the "outlier energy" readout), so the error they cause spreads over all 128 coordinates after the rotation.
The fix both papers use: split the channels into an outlier set and a non-outlier set, then run two INDEPENDENT quantizer instances, each with its own rotation, codebook and norm. Give the outliers more bits. Effective bits = (n_o·b_hi + (128 − n_o)·b_lo)/128.
TurboQuant's 2.5-bit setting (§4.3): "32 outlier channels are quantized at 3 bits, while the remaining 96 channels use 2 bits", and the paper prints (32×3 + 96×2)/128 = 2.5. CAREFUL: that arithmetic is 288/128 = 2.25, not 2.5 (checked against the arXiv v1 HTML). The slide defaults to this split and shows the honest 2.25. Either the channel counts or the result is a typo (64 @ 3 + 64 @ 2 would give exactly 2.5). Another reconciliation, purely our guess: two fp16 norms per 128-dim vector add 32/128 = 0.25 bits/channel, and 2.25 + 0.25 = 2.5. We don't know which one is right.
For 3.5 bits the paper only says "a different ratio" and doesn't give the split; drag the sliders to explore (e.g. 64 @ 4 + 64 @ 3 = 3.5). Memory vs 16-bit at the nominal widths: 16/2.5 = 6.4× and 16/3.5 ≈ 4.57× (not counting norms).
Bottom-left live simulation: 200 synthetic keys from the selected layer model, d = 128. Bars show the mean relative error ‖k − k̃‖²/‖k‖²:
 • one TurboQuant_mse instance at b_lo or b_hi everywhere (grey, for reference; math.js turboMse with one 128×128 rotation, norm stored);
 • one instance at the SAME average bits (violet): rotate all 128 channels, then give n_o of the rotated coordinates b_hi bits. After a rotation the coordinates are exchangeable, so it doesn't matter which ones;
 • the split (teal): two independent instances of dimension n_o and 128 − n_o, each with its own Haar rotation, Lloyd–Max codebook for that dimension, and norm.
In the deep layer with the paper's split (32 @ 3 + 96 @ 2) the split brings the error down to about the 3-bit single instance, because the bits go where the energy is. In the early layer there's no energy concentration, so the split gains little. The win needs the outlier energy share to be much bigger than n_o/128.
How we pick the outlier set here: the n_o channels with the most energy across these keys, a stand-in for QJL's "identified during the prompt" phase.
What TurboQuant does NOT say: how outlier channels are chosen; whether keys and values are treated the same; whether it uses the mse or the prod variant in the KV experiments; whether the extra per-instance norm (two floats per vector instead of one, about +0.25 bits/channel at fp16 for d = 128) is counted in "2.5 bits". QJL says outliers are identified during the prompt and use two independent QJL quantizers with more bits for outliers.

## 22. Quality-neutral KV cache at 3.5 bits

*`#results-kv` · Practice · Experiments*

Left: TurboQuant Table 1, LongBench-E. Scores per task category for Llama-3.1-8B-Instruct (all six methods) and Ministral-7B (only Full and TurboQuant 2.5 are reported). The number under each bar is the KV cache size in bits per channel (16 = full precision).
Pick a task. Use "Δ vs full" to zoom in honestly: the bars then start at the full-precision score, so the tiny differences are visible without a truncated axis.
Headline: TurboQuant at 3.5 bits averages 50.06, EXACTLY the full-precision average (50.06). At 2.5 bits it loses about 0.6 points on average (49.44). KIVI at 3 bits loses about 1.6 points (48.50); most of that comes from MultiQA (37.99 vs 45.16). PolarQuant at 3.9 bits gets 49.78.
Caveats, from our reading of the paper:
 • KIVI at 5 bits averages 50.16, slightly ABOVE TurboQuant 3.5's 50.06 (= Full). So "best" is about bits, not raw score. KIVI-5 spends 1.5 more bits per channel.
 • The paper states the compression three ways: "a factor exceeding 5×" (§1.3), "at least a factor of 4.5×" (§4.3), and "4× compression" / "more than 4× quantized" (§4.2 and the Fig. 4 caption). For reference, 16/3.5 ≈ 4.57× and 16/2.5 = 6.4×. Norm storage and the outlier split's details aren't fully specified, so treat all these ratios as approximate.
 • The 2.5-bit split as printed, (32×3 + 96×2)/128, actually evaluates to 2.25, not 2.5 (see the previous slide).
 • The 3.5-bit outlier split isn't given.
Right: needle-in-a-haystack (TurboQuant Fig. 4): Llama-3.1-8B-Instruct, 4k to 104k tokens, memory ratio 0.25 (compressed KV is 25% of full). Scores: SnapKV 0.858, PyramidKV 0.895 (token-eviction methods drop context), KIVI 0.981, PolarQuant 0.995, Full precision 0.997, TurboQuant 0.997. The paper does NOT state TurboQuant's bit-width for this experiment. It only says "4× compression" and, in the caption, "more than 4× quantized". A 0.25 memory ratio would be about 4 bits per channel, but that is our inference.
Toggle "QJL (2024)" to show the earlier paper's results. QJL Table 1: LongBench F1 on longchat-7b-v1.5-32k at 3 bits vs KIVI (3 bits), KVQuant (4.3 bits) and FP16. QJL is best on NarrativeQA, Qasper and HotpotQA, and at or near FP16 elsewhere. QJL Table 2: LM-eval accuracy on Llama-2-7B (FP16 / KIVI-3 / QJL-3) and Llama-3-8B (BF16 / QJL-3). Every entry is within about 0.2 points of full precision. KIVI can't run in BF16, so there is no KIVI row for Llama-3.
Takeaway: a data-oblivious, online quantizer with no calibration matches full precision at about 3.5 bits per channel.

## 23. Nearest-neighbor search with zero indexing time

*`#results-nn` · Practice · Experiments*

Same quantizer, different job: compress a database of embeddings, then answer nearest-neighbor queries using inner products with the compressed vectors.
Left: TurboQuant Table 2. Time to quantize (index) the dataset at 4 bits for d = 200 (GloVe), 1536 and 3072 (DBpedia OpenAI3 embeddings, 100k points). Log axis. Pick d: Product Quantization 37.04 s / 239.75 s / 494.42 s; RabitQ 597.25 s / 2267.59 s / 3957.19 s; TurboQuant 0.0007 s / 0.0013 s / 0.0021 s. The TurboQuant-vs-PQ gap is about 5 orders of magnitude (≈ 53,000× at d = 200) and TurboQuant-vs-RabitQ about 6 (≈ 850,000× at d = 200, ≈ 1.9 million× at d = 3072). The log axis spans 8 decades. Why: TurboQuant is data-oblivious. There's no codebook to train (PQ runs k-means per subspace) and no per-dataset tuning, so indexing is one rotation (a matrix multiply) plus a per-coordinate lookup into a precomputed Lloyd–Max codebook. RabitQ isn't vectorizable on GPU, per the paper. Same hardware: a single A100.
Middle: recall@1@1 (is the true nearest neighbor ranked first?) from TurboQuant Fig. 5. These values are READ OFF THE PLOT by us, so they're approximate. Toggle 2 vs 4 bits. TurboQuant is on top in every setting even though it does no training. In the paper, all methods reach 1.0 by k ≈ 4–8 on OpenAI3, and GloVe at 2 bits gets to about 0.98–0.99 by k = 64.
Right: a LIVE, SYNTHETIC demo running in this browser. 2000 unit vectors in d = 128 drawn from 40 Gaussian clusters, plus 100 held-out queries from the same distribution. Each database vector is quantized with math.js turboMse (one random rotation, the Lloyd–Max codebook for d = 128) at 2 and 4 bits. Then recall@1@k = the fraction of queries whose exact top-1 inner-product neighbor is in the top k of the quantized ranking, for k = 1..64 (log axis). The curves climb to 1. 4 bits is near-perfect by small k; 2 bits needs a larger candidate list, the same qualitative picture as Fig. 5. The "index (JS)" readout is the wall-clock time this browser spent quantizing all 2000 vectors at one bit-width (unoptimized JavaScript, single thread, includes the inverse rotation for reconstruction). The "R@1@1" readouts are recall@1@1 at 2 and 4 bits. "New rotation" redraws Π and recomputes.
Caveat: the synthetic data isn't GloVe or OpenAI embeddings, and the absolute recall depends on how tightly clustered the data is. Only the shape of the curves carries over.

## 24. Every step was forced

*`#recap` · Recap · Recap*

The whole talk as one chain. Each box is a decision, and each one is forced by a failure of the box before it. Hover or click any box to show the problem it fixes, its equation, and links back to the slides where we built it. The chain auto-advances until you touch it.
1. Requirements: online (no calibration pass), vectorizable (GPU-friendly), zero overhead (no per-block scale or zero point), worst-case inputs.
2. Naive rounding fails: block constants cost 1–2 extra bits per number (QJL's claim), and outliers stretch the grid.
3. For keys we only need ⟨q,k⟩, a weaker target than reconstruction, so any inner-product-preserving map will do.
4. A JL projection is such a map: unbiased, random error, m independent of d, drawn before any data. But each row costs 16 bits, and saving memory means dropping rows.
5. Keep 1 bit per row and use many more rows (QJL): still unbiased (asymmetric, query unquantized), no overhead, ≈3–7× less error than floats at the same bits. But error falls only like 1/m, polynomial in bits.
6. Why randomness helped: it made the coordinate distribution known, the error random, the energy spread. A random rotation does the same while preserving lengths: worst case becomes average case.
7. Rotated coordinates follow a known Beta-type density, approximately N(0, 1/d) in high dimension, and are nearly independent.
8. So quantize each coordinate with the Lloyd–Max codebook for that density: D_mse ≤ (√3π/2)·4^{-b} ≈ 2.72·4^{-b}. That's exponential in bits.
9. But MSE-optimal reconstructions are biased for inner products: E⟨x, x̃⟩ = 1 − D_mse (our derivation), exactly 2/π at b = 1 (paper). Shrunk logits flatten the softmax.
10. Fix: (b−1)-bit TurboQuant_mse plus 1-bit QJL on the residual. Unbiased, D_prod ≤ (√3π²/d)·4^{-b}.
11. Lower bound (Shannon, extended to randomized quantizers via Yao's principle): no quantizer beats 4^{-b} (MSE) or (1/d)·4^{-b} (inner product). TurboQuant is within about 2.7× (1.45× at b = 1), and decays like 4^{-b} where QJL decays like 1/b.
12. In practice: outlier channels get their own instance with more bits. Paper's split 32 @ 3 + 96 @ 2 (printed as 2.5 bits, actually 2.25); quality-neutral LongBench at 3.5; essentially zero indexing time for nearest-neighbor search.
Color code: grey = setup, red = where an approach breaks, blue = QJL (2024), teal = TurboQuant (2025), yellow = practice.
References at the bottom link to both arXiv papers.

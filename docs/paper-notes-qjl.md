# QJL (arXiv:2406.03482 v2) — extracted facts for the talk

Zandieh (Independent), Daliri (NYU), Han (Adobe/Yale). "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead".

## Motivation
- Existing KV quantizers store constants "(at least a zero point and a scale) in full precision per data block" → "1 or 2 additional bits per quantized number" (asserted, not derived).
- Memory: at 3 bits, "more than fivefold reduction in KV cache memory usage"; "around an 81% reduction" (1 − 3/16).
- Decode is memory-bound: whole KV cache read from HBM per generated token; Eq.2 costs O(nd).
- QJL is data-oblivious: a sketch, no tuning, no grouping.

## Math
- Attention: $o_n=\sum_i \mathrm{Score}(i)\,v_i$, $\mathrm{Score}=\mathrm{softmax}([\langle q_n,k_1\rangle,\dots,\langle q_n,k_n\rangle])$.
- JL lemma not stated formally (cites JL, Dasgupta–Gupta). Supply standard statement ourselves.
- Def 3.1: $S\in\mathbb R^{m\times d}$, $S_{ij}\sim\mathcal N(0,1)$. $\mathcal H_S(k)=\mathrm{sign}(Sk)\in\{\pm1\}^m$.
  $\mathrm{Prod_{QJL}}(q,k)=\frac{\sqrt{\pi/2}}{m}\|k\|_2\langle Sq,\mathcal H_S(k)\rangle$.
- Lemma 3.2: unbiased. Proof: $q^{\perp k}=q-\frac{\langle q,k\rangle}{\|k\|^2}k$; $s^\top q^{\perp}$ and $s^\top k$ independent Gaussians → cross term 0; $\mathbb E|s^\top k|=\sqrt{2/\pi}\|k\|$.
  Per-row identity: $\mathbb E[\mathrm{sign}(s^\top k)\,s^\top q]=\sqrt{2/\pi}\,\langle q,k\rangle/\|k\|$.
- Lemma 3.5: if $m\ge\frac43\frac{1+\varepsilon}{\varepsilon^2}\log\frac2\delta$ then $\Pr[|\mathrm{Prod}-\langle q,k\rangle|>\varepsilon\|q\|\|k\|]\le\delta$. (Paper's exponent is missing a minus sign — typo.)
- Variance (derived, not printed): $\mathrm{Var}=\frac{(\pi/2)\|q\|^2\|k\|^2-\langle q,k\rangle^2}{m}\le\frac{\pi}{2m}\|q\|^2\|k\|^2$.
- Thm 3.6: if $\|k_i\|,\|q\|\le r$ and $m\ge 2r^2\varepsilon^{-2}\log n$ then all softmax scores within $(1\pm3\varepsilon)$ w.h.p. → $m\approx\varepsilon^{-2}\log n$ bits per key, independent of $d$.
- Asymmetric: sign on both sides → estimates the angle (Charikar SimHash); cos of the estimate is biased. So only keys are quantized; query is the unquantized $Sq$.

## Algorithm / practice
- Store per key: $m$ sign bits + norm $\|k\|$. Values: standard token-wise quantization (KIVI-style).
- Outliers: deeper layers have a few fixed channels (Fig 2 caption: "approximately four") with large magnitude across all tokens. Identified during prompt; two independent QJL quantizers (outliers get more bits).
- All experiments orthogonalize rows of $S$ via QR.
- CUDA kernels; single A100; supports GQA and BF16 (KIVI can't run BF16 → no KIVI on Llama-3).

## Experiments
Table 1 (LongBench F1, longchat-7b-v1.5-32k):

| Method | Bits | NarrativeQA | Qasper | MultiQA-en | MultiQA-zh | HotpotQA | 2WikiMultiQA |
|---|---|---|---|---|---|---|---|
| FP16 | 16 | 20.79 | 29.42 | 42.83 | 34.33 | 33.05 | 24.14 |
| KIVI | 3 | 20.96 | 29.01 | 40.93 | 34.75 | 32.79 | 23.01 |
| KVQuant | 4.3 | 20.14 | 28.77 | 44.22 | 34.44 | 34.06 | 23.05 |
| QJL | 3 | 21.83 | 29.44 | 41.52 | 34.42 | 35.62 | 23.60 |

Table 2 (LM-eval accuracy):

| Model | Method | Bits | Lambada | HellaSwag | PIQA | MathQA | MMLU |
|---|---|---|---|---|---|---|---|
| Llama-2-7B | FP16 | 16 | 73.90 | 57.18 | 78.07 | 28.11 | 41.85 |
| | KIVI | 3 | 73.88 | 57.13 | 78.07 | 28.11 | 41.81 |
| | QJL | 3 | 73.88 | 57.14 | 78.07 | 28.17 | 41.78 |
| Llama-3-8B | BF16 | 16 | 75.59 | 60.17 | 79.65 | 40.64 | 62.09 |
| | QJL | 3 | 75.61 | 60.13 | 79.87 | 40.60 | 62.12 |

Fig 2: Llama-2 key magnitudes. Layer 0 no outlier structure (≤ ~2.5). Layers 15 & 31: a few channels at ~12–14, rest low.

Fig 3 (single attention layer wall-clock, log x token length 1k–64k):
(a) encode+quantize ms, Llama2: KVQuant ≈1.85 at 64k; FP16/QJL/KIVI ≈0.4.
(b) decode ms, 128 tokens: KVQuant ≈5.9–6.4 flat; others < 0.6.
(c) total ms Llama3 BF16 vs QJL: nearly identical (~1.75–1.9 at 64k).

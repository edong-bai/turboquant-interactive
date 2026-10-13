# TurboQuant, interactively

A self-hosted, vertically scrolling talk that rediscovers
[QJL](https://arxiv.org/abs/2406.03482) and [TurboQuant](https://arxiv.org/abs/2504.19874)
from first principles: mostly geometry you can drag, with the words in the speaker notes.

## Run it

```bash
npm install        # once
npm run dev        # dev server with hot reload → http://localhost:5173
npm run build      # static site in dist/ — host anywhere (no CDN calls; fonts + KaTeX are bundled)
npm run preview    # serve the built dist/ locally
npm run notes      # regenerate docs/SPEAKER_NOTES.md from the slides
```

### Deploy (GitHub Pages)

`.github/workflows/deploy.yml` builds and publishes `dist/` on every push to `main`.
Enable it once under **Settings → Pages → Source: GitHub Actions**. A private repo can publish a
public Pages site on GitHub Pro / Team / Enterprise (not on GitHub Free).

Keys: **↓ / space / →** next · **↑ / ←** back · **N** speaker notes · **F** fullscreen ·
**P** dim the progress rail · **Home / End**. Every slide has a URL hash (`#qjl`, `#lloyd-max`, …).

Speaker notes for the whole deck: [`docs/SPEAKER_NOTES.md`](docs/SPEAKER_NOTES.md).

## Arc of the talk

| # | Slide | Hash | The point |
|---|---|---|---|
| | **Why** | | |
| 1 | TurboQuant | `#title` | |
| 2 | The KV cache is the bottleneck | `#kv-cache` | Decoding re-reads the whole cache; memory calculator (bits × context × layers) |
| 3 | What a quantizer must do | `#requirements` | Online & data-oblivious, accelerator-friendly, zero per-block overhead, worst-case guarantee |
| 4 | Two errors to minimize | `#metrics` | $D_{\rm mse}$ (values), $D_{\rm prod}$ (keys), unbiasedness |
| | **Naive** | | |
| 5 | Normalize, then round | `#naive` | The obvious algorithm |
| 6 | Adversarial inputs break it | `#naive-fails` | Spread vectors & outlier channels; no worst-case bound (question parked for later) |
| | **QJL** | | |
| 7 | Keys only appear inside ⟨q,k⟩, so JL is enough | `#jl` | Only inner products matter ⇒ any inner-product-preserving map works ⇒ JL: unbiased, random error, m independent of d; but each row costs 16 bits |
| 8 | Keep only the sign: more rows beat more precision | `#qjl` | Same bits: many sign rows beat a few float rows by ≈3–7×; asymmetric (query unquantized) vs SimHash |
| 9 | Why one bit stays unbiased | `#qjl-why` | $\sqrt{\pi/2}$ from the folded Gaussian; variance $\le\pi/2m$; error ∝ 1/√bits |
| | **Rotation** | | |
| 10 | Why project before taking the sign? | `#why-project` | k=(10, 0.1), q=(0, 1): raw signs give 7.07, QJL 0.1 ± noise. Randomizing gives a known distribution, random error, spread energy → same trick next |
| 11 | What if no input could be adversarial? | `#rotation` | Rotate first: worst case → average case |
| 12 | d = 3: every band holds the same mass | `#sphere-d3` | Archimedes + the tilt correction |
| 13 | d = 4: a flipbook of spheres | `#sphere-d4` | Semicircle density |
| 14 | High dimensions: a Gaussian, nearly independent | `#concentration` | → $\mathcal N(0,1/d)$; one fixed codebook, no per-block scale |
| | **TurboQuant** | | |
| 15 | Lloyd–Max: k-means on a known density | `#lloyd-max` | Optimal scalar codebook, solved once offline |
| 16 | TurboQuant_mse | `#tq-mse` | Algorithm 1, $D_{\rm mse}\le 2.72\cdot4^{-b}$ |
| 17 | MSE-optimal shrinks inner products | `#mse-bias` | Bias $1-D_{\rm mse}$ ($2/\pi$ at 1 bit); shrunk logits flatten the softmax |
| 18 | Spend the last bit on QJL of the residual | `#tq-prod` | Algorithm 2, unbiased |
| 19 | Within a small constant of any quantizer | `#bounds` | Lower bound $4^{-b}$, TurboQuant within 2.7×; $4^{-b}$ vs QJL's $1/b$ |
| 20 | When to drop the correction | `#mse-vs-prod` | Bias vs variance; crossover near b ≈ 3 |
| | **Practice** | | |
| 21 | Outlier channels get more bits | `#outliers` | Paper's split 32 @ 3 + 96 @ 2 (printed as 2.5 bits, actually 2.25) |
| 22 | Quality-neutral KV cache at 3.5 bits | `#results-kv` | LongBench, needle-in-a-haystack; QJL's results |
| 23 | Nearest-neighbor search with zero indexing time | `#results-nn` | Recall, indexing time vs PQ / RabitQ |
| | **Recap** | | |
| 24 | Every step was forced | `#recap` | The chain of forced decisions, linked back to each slide |

## Layout

```
index.html               shell (rail, HUD, notes overlay)
src/main.js              deck runtime: mounts slides, visibility-gated animation, keyboard nav
src/styles.css           design tokens + layout classes
src/lib/                 math (rotations, Lloyd–Max, QJL, TurboQuant), canvas, plot, ui, KaTeX, 3-D
src/slides/NN-*.js       one module per slide; deck order is src/slides/index.js
src/slides/_shared-*.js  helpers shared by a few slides
scripts/export-notes.mjs regenerates docs/SPEAKER_NOTES.md
docs/SLIDES.md           how to write a slide
docs/SPEAKER_NOTES.md    all speaker notes, in deck order (generated)
docs/paper-notes-*.md    facts, constants and experiment tables extracted from both papers
```

All numbers shown on slides come from `docs/paper-notes-*.md` or are computed live in the
browser (labeled "live simulation"). Illustrative data is labeled as such.

Note: file numbers in `src/slides/` no longer match deck positions after reordering; the order
in `src/slides/index.js` is authoritative.

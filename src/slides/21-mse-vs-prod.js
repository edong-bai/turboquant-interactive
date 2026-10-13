import { el, eq, slider, segmented, readout, viz, legend, rich } from '../lib/ui.js';
import { createCanvas, line, text, label, color, alpha } from '../lib/canvas.js';
import { Plot } from '../lib/plot.js';
import { histogram, bounds } from '../lib/math.js';
import { sim as getSim, BMAX, D, ipErrors, moments, measuredDmse } from './_shared-tq.js';
import { disc, progressBadge } from './_shared-tq-draw.js';

const HIST_C = [0.05, 0.25, 0.5, 0.8]; // ⟨y, x⟩ for the Fig. 2-like row
const HIST_B = 2;

export default {
  id: 'mse-vs-prod',
  chapter: 'TurboQuant',
  kicker: 'Bias vs variance',
  title: 'When to drop the correction',
  notes: `Two estimators of ⟨y, x⟩ with the same b bits per coordinate: TurboQuant_mse (biased, small spread) and TurboQuant_prod (unbiased, larger spread). Which has the smaller inner-product error D_prod = E|⟨y,x⟩ − ⟨y,x̃⟩|²? It depends on b and on how large the inner product is.

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

Takeaway: unbiasedness matters most at low bit-width and when the inner products that matter (the large attention logits) are large. At 3+ bits the MSE variant is simply more accurate. Caveat: the paper does not state which variant its KV-cache and nearest-neighbor experiments use.`,
  mount(body, ctx) {
    const sim = getSim();
    const state = { c: 0.6, b: 2 };

    const stats = () => {
      const out = [null];
      for (let b = 1; b <= BMAX; b++) out.push({ mse: moments(ipErrors('mse', b, state.c, sim)), prod: moments(ipErrors('prod', b, state.c, sim)) });
      return out;
    };
    let cached = null;
    let cacheKey = '';
    const getStats = () => {
      const key = `${sim.count}:${state.c}`;
      if (key !== cacheKey) {
        cached = stats();
        cacheKey = key;
      }
      return cached;
    };

    // ---------- D_prod vs b ----------
    const vm = viz('', 'live simulation · $d=128$ · $D_{\\rm prod}$ vs $b$');
    const main = createCanvas(vm, (g, w, h) => {
      if (w < 40 || h < 40) return;
      const p = new Plot(g, {
        x: 0,
        y: 0,
        w,
        h,
        xmin: 0.6,
        xmax: BMAX + 0.4,
        ymin: 4e-6,
        ymax: 0.3,
        ylog: true,
        xticks: [1, 2, 3, 4, 5],
        xfmt: (v) => `b=${v}`,
        ylabel: 'D_prod',
        margin: { l: 58, r: 16, t: 32, b: 30 },
      });
      p.frame();
      p.clip(() => {
        g.fillStyle = alpha('yellow', 0.08);
        g.fillRect(p.tx(state.b - 0.35), p.py, p.tx(state.b + 0.35) - p.tx(state.b - 0.35), p.ph);
      });
      p.fn((b) => bounds.prodUpper(b, D), { color: 'teal', width: 1.2, dash: [6, 4], alpha: 0.6 });
      p.fn((b) => bounds.prodLower(b, D), { color: 'blue', width: 1.2, dash: [6, 4], alpha: 0.8 });
      label(g, '(√3π²/d)·4⁻ᵇ', p.tx(4.4), p.ty(bounds.prodUpper(4.4, D)) - 14, { color: 'teal', size: 11 });
      label(g, '(1/d)·4⁻ᵇ', p.tx(1.5), p.ty(bounds.prodLower(1.5, D)) + 14, { color: 'blue', size: 11 });
      if (sim.count > 0) {
        const st = getStats();
        const bs = [1, 2, 3, 4, 5];
        const ym = bs.map((b) => st[b].mse.ms);
        const yp = bs.map((b) => st[b].prod.ms);
        p.line(bs, ym, { color: 'red', width: 2 });
        p.line(bs, yp, { color: 'teal', width: 2 });
        for (let i = 0; i < 5; i++) {
          const mseWins = ym[i] < yp[i];
          disc(p, bs[i], ym[i], { color: 'red', r: mseWins ? 7 : 4.5 });
          disc(p, bs[i], yp[i], { color: 'teal', r: mseWins ? 4.5 : 7 });
          text(g, mseWins ? 'mse' : 'prod', p.tx(bs[i]), p.py + 12, { color: mseWins ? 'red' : 'teal', size: 12, align: 'center', font: 'mono', weight: 600 });
        }
      }
      progressBadge(p, sim.progress);
    });

    // ---------- Fig. 2-like row ----------
    const vh = viz('', `error histograms at $b=${HIST_B}$`);
    const hists = createCanvas(vh, (g, w, h) => {
      if (w < 80 || h < 60 || sim.count === 0) return;
      const sets = HIST_C.map((c) => {
        const em = ipErrors('mse', HIST_B, c, sim);
        const ep = ipErrors('prod', HIST_B, c, sim);
        return { c, em, ep, mm: moments(em), mp: moments(ep) };
      });
      let R = 0;
      for (const s of sets) R = Math.max(R, 3 * Math.sqrt(s.mp.var), Math.abs(s.mm.mean) + 3 * Math.sqrt(s.mm.var));
      R *= 1.05;
      const n = sets.length;
      const pw = w / n;
      sets.forEach((s, k) => {
        const hm = histogram(s.em, -R, R, 31);
        const hp = histogram(s.ep, -R, R, 31);
        const p = new Plot(g, {
          x: k * pw,
          y: 0,
          w: pw,
          h,
          xmin: -R,
          xmax: R,
          ymin: 0,
          ymax: Math.max(...hm, ...hp) * 1.1,
          xticks: [-0.1, 0, 0.1].filter((v) => Math.abs(v) < R),
          yticks: [],
          xfmt: (v) => (v === 0 ? '0' : v.toFixed(1)),
          margin: { l: 10, r: 10, t: 54, b: 22 },
        });
        p.frame({ grid: false });
        p.bars(hp, -R, R, { color: 'teal', alpha: 0.55, gap: 0.5 });
        p.bars(hm, -R, R, { color: 'red', alpha: 0.5, gap: 0.5 });
        p.vline(0, { color: 'ink-2', dash: [3, 3] });
        p.vline(s.mm.mean, { color: 'red', dash: null, width: 2 });
        text(g, `⟨y,x⟩ = ${s.c}`, p.px + p.pw / 2, 42, { color: 'ink-2', size: 12, align: 'center', font: 'mono' });
      });
    });

    // ---------- decomposition bars ----------
    const rMse = readout('mse total');
    const rProd = readout('prod total');
    const rRatio = readout('var ratio prod/mse');
    const vd = viz('', '');
    const decomp = createCanvas(vd, (g, w, h) => {
      if (w < 60 || h < 60 || sim.count === 0) return;
      const st = getStats()[state.b];
      const items = [
        { name: 'mse', bias2: st.mse.mean ** 2, v: st.mse.var, c: 'red' },
        { name: 'prod', bias2: st.prod.mean ** 2, v: st.prod.var, c: 'teal' },
      ];
      // simple formulas: mse ≈ (D(b)·c)² + D(b)(1−c²)/d,  prod ≈ (π/2d)·D(b−1)
      const Db = measuredDmse(state.b, sim);
      const Dprev = state.b > 1 ? measuredDmse(state.b - 1, sim) : 1;
      items[0].pred = (Db * state.c) ** 2 + (Db * (1 - state.c ** 2)) / D;
      items[1].pred = (Math.PI / (2 * D)) * Dprev;
      const top = Math.max(...items.map((i) => Math.max(i.bias2 + i.v, i.pred))) * 1.18;
      const x0 = 70;
      const x1 = w - 16;
      const yTop = 40;
      const bh = Math.max(18, Math.min(64, (h - yTop - 50) / 3.3));
      const X = (v) => x0 + ((x1 - x0) * v) / top;
      text(g, `b = ${state.b},  ⟨y,x⟩ = ${state.c.toFixed(2)}`, 14, 18, { color: 'ink-3', size: 12, font: 'mono' });
      items.forEach((it, k) => {
        const y = yTop + k * (bh * 1.7);
        text(g, it.name, x0 - 10, y + bh / 2, { color: it.c, size: 15, align: 'right', weight: 600 });
        g.fillStyle = color('red');
        g.fillRect(X(0), y, X(it.bias2) - X(0), bh);
        g.fillStyle = alpha('violet', 0.85);
        g.fillRect(X(it.bias2), y, X(it.bias2 + it.v) - X(it.bias2), bh);
        text(g, (it.bias2 + it.v).toExponential(1), Math.max(X(it.bias2 + it.v), X(it.pred)) + 8, y + bh / 2, { color: 'ink', size: 12, font: 'mono' });
        line(g, X(it.pred), y - 5, X(it.pred), y + bh + 5, { color: 'yellow', width: 2.5 });
      });
      const yEnd = yTop + 2 * bh * 1.7 - bh * 0.7;
      line(g, X(0), yTop - 6, X(0), yEnd, { color: 'line-2' });
      const ly = yEnd + 12;
      g.fillStyle = color('red');
      g.fillRect(x0, ly, 12, 10);
      text(g, 'bias²', x0 + 18, ly + 5, { color: 'ink-2', size: 12 });
      g.fillStyle = alpha('violet', 0.85);
      g.fillRect(x0 + 74, ly, 12, 10);
      text(g, 'variance', x0 + 92, ly + 5, { color: 'ink-2', size: 12 });
      g.fillStyle = color('yellow');
      g.fillRect(x0 + 168, ly - 2, 3, 14);
      text(g, 'formula (below)', x0 + 178, ly + 5, { color: 'ink-2', size: 12 });
      rMse.set(st.mse.ms.toExponential(2), 'red');
      rProd.set(st.prod.ms.toExponential(2), 'teal');
      rRatio.set(`×${(st.prod.var / st.mse.var).toFixed(1)}`, 'violet');
    });

    const redrawAll = () => {
      main.request();
      decomp.request();
    };
    const sC = slider({
      label: '$\\langle y,x\\rangle$',
      min: 0,
      max: 0.95,
      step: 0.05,
      value: state.c,
      format: (v) => v.toFixed(2),
      onInput: (v) => {
        state.c = v;
        redrawAll();
      },
    });
    const segB = segmented({
      label: '$b$',
      options: [1, 2, 3, 4, 5],
      value: state.b,
      onChange: (v) => {
        state.b = v;
        redrawAll();
      },
    });

    const grow = (node, flex, min) => Object.assign(node.style, { flex: `${flex} 1 0`, height: 'auto', minHeight: min });
    grow(vm, 1.45, '260px');
    grow(vh, 0.75, '150px');
    grow(vd, 0.8, '170px');

    const left = el(
      'div.col',
      {},
      vm,
      legend([
        { label: 'TurboQuant$_{\\rm mse}$', color: 'red' },
        { label: 'TurboQuant$_{\\rm prod}$', color: 'teal' },
        { label: 'paper bounds for prod', color: 'blue', dash: true },
      ]),
      vh,
    );
    const right = el(
      'div.col',
      {},
      el('div.panel', { style: { gap: '8px', padding: '12px 16px' } }, sC.el, segB.el),
      vd,
      el('div.readouts', {}, rMse.el, rProd.el, rRatio.el),
      el(
        'div.card',
        { 'data-reveal': '1', style: { padding: '8px 14px', gap: '0', fontSize: '0.86em' } },
        eq('\\text{mse: }\\ \\E[\\text{err}]\\approx\\Dmse(b)\\,\\ip{y}{x}'),
        eq('\\Var\\approx\\tfrac{1}{d}\\Dmse(b)\\,(1-\\ip{y}{x}^2)'),
        eq('\\text{prod: }\\ \\Var\\approx\\tfrac{\\pi}{2d}\\Dmse(b{-}1)\\approx 2\\pi\\cdot\\tfrac{\\Dmse(b)}{d}'),
      ),
      rich('p.note', 'Low bits, large $\\langle y,x\\rangle$ → prod. &nbsp;High bits → mse.', { 'data-reveal': '2' }),
    );
    body.append(el('div.stage', {}, left, right));

    let seen = -1;
    ctx.onEnter(() => {
      redrawAll();
      hists.request();
    });
    ctx.loop(() => {
      if (!sim.done) sim.step(5);
      if (sim.count !== seen) {
        seen = sim.count;
        main.redraw();
        decomp.redraw();
        hists.redraw();
      }
    });
  },
};

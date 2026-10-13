import { el, eq, slider, segmented, readout, viz, legend } from '../lib/ui.js';
import { createCanvas, line, text, label, alpha } from '../lib/canvas.js';
import { Plot } from '../lib/plot.js';
import { RNG } from '../lib/math.js';
import { naiveLevels, naiveQuantize, naiveStep, relErr, pct } from './_shared-naive.js';

const TQ_UPPER = (Math.sqrt(3) * Math.PI) / 2; // ≈ 2.72 (TurboQuant Thm 1)
const D_OUT = 64; // dimension of the outlier demo
const J_OUT = 20; // index of the outlier channel

/**
 * Relative error of the naive quantizer on explicit adversarial unit vectors, maximized over a
 * few families. These are achieved values (a lower bound on the true worst case), and they sit
 * just under the bound min(1, d·Δ²/4) for b ≥ 2.
 */
function adversarialWorst(d, b, mode) {
  const D = naiveStep(b, 1);
  const c = (D / 2) * (1 - 1e-9); // just below half a step: rounds to 0
  const cands = [];
  const vec = (f) => Float64Array.from({ length: d }, (_, i) => f(i));
  // spread: every coordinate ±1/√d
  cands.push(vec(() => 1 / Math.sqrt(d)));
  // spike
  cands.push(vec((i) => (i === 0 ? 1 : 0)));
  // one big coordinate plus d−1 tiny ones
  cands.push(vec((i) => (i === 0 ? 1 : 1e-6)));
  if (mode === 'fixed') {
    // d−1 coordinates just under Δ/2 (all round to 0), the rest of the norm in one coordinate
    const k = Math.min(d - 1, Math.floor((1 - 1e-12) / (c * c)));
    const rest = Math.sqrt(Math.max(0, 1 - k * c * c));
    cands.push(vec((i) => (i < k ? c : i === k ? rest : 0)));
  } else {
    // one coordinate sets the scale, all others sit just under half a step
    cands.push(vec((i) => (i === 0 ? 1 : c)));
  }
  let worst = 0;
  for (const x of cands) worst = Math.max(worst, relErr(x, naiveQuantize(x, b, mode).xq));
  return worst;
}

export default {
  id: 'naive-fails',
  chapter: 'Naive',
  kicker: 'First attempt',
  title: 'Adversarial inputs break it',
  notes: `Same naive rounder as the previous slide (step Δ, round to the nearest multiple; 1 bit = sign). Now we let an adversary choose x.

Left, fixed grid after normalizing (solid red). The adversary's favorite input is the spread vector: every coordinate ±1/√d. Once 1/√d is below half a step, every coordinate rounds to 0, x̃ = 0, and the relative error is exactly 100%. That happens for d > (2^b − 1)², so at b = 4 already at d ≈ 226, and for b = 2 at d = 10. For smaller d the adversary still gets close to d·Δ²/4 by putting d − 1 coordinates just under half a step. The curves are the error achieved by these explicit adversarial vectors, computed live. b = 1 (sign) is even worse: a single spike reconstructs as all ±1, so the error is d − 1 (off the top of the chart).
Dashed red: the min–max (KIVI-style) variant. Its adversary is one big coordinate plus many coordinates just under half a step; the error tends to 100% as d grows, again.
The teal band is where we are going: between 4^(−b) (the lower bound for any quantizer, proved later) and about 2.72 · 4^(−b) (TurboQuant's guarantee). Flat in d, and for every input. Switch b to compare.

Right, min–max scaling with one outlier channel (d = 64, synthetic Gaussian channels plus one channel of strength A). Drag A up: the grid stretches to [−A, A], the step grows like A, and every ordinary channel collapses onto the zero level. The whole-vector relative error looks modest because the outlier dominates the norm, but the information in the other 63 channels is gone (third readout). Real keys have exactly this structure: in QJL's Fig. 2 (Llama-2), deeper layers have a few fixed channels with magnitudes around 12–14 while the rest stay near 1. Layer 0 shows no such structure.

The bound: rounding error per coordinate is at most Δ/2, so ‖x − x̃‖² ≤ d·Δ²/4 with Δ = 2 max|x_i| / (2^b − 1). It scales with d · max², and the adversary controls max. No worst-case guarantee relative to ‖x‖² is possible this way.

Takeaway: the naive quantizer has no worst-case guarantee; its error depends on how x sits relative to the axes. Park this failure: we'll come back to it when TurboQuant is introduced. (Next: a different angle on the problem: what do keys actually need?)`,
  mount(body, ctx) {
    const state = { b: 3, A: 10, hoverD: null };
    const rngO = new RNG(64);
    const noise = Float64Array.from({ length: D_OUT }, () => rngO.normal());

    // ---------- (i) worst case vs d ----------
    let curves = null; // {ds, fixed[b][], minmax[b][]}
    const computeCurves = () => {
      if (curves) return;
      const set = new Set();
      for (let i = 0; i <= 160; i++) set.add(Math.round(2 * 512 ** (i / 160)));
      const ds = [...set].sort((a, b) => a - b);
      const fixed = {};
      const minmax = {};
      for (let b = 1; b <= 4; b++) {
        fixed[b] = ds.map((d) => adversarialWorst(d, b, 'fixed'));
        minmax[b] = ds.map((d) => adversarialWorst(d, b, 'minmax'));
      }
      curves = { ds, fixed, minmax };
    };

    const vL = viz('', 'worst case over $x$ · live');
    Object.assign(vL.style, { flex: '1 1 0', height: 'auto', minHeight: '260px' });
    let plotL = null;
    const cvL = createCanvas(vL, (g, w, h) => {
      const P = new Plot(g, {
        x: 0,
        y: 26,
        w,
        h: h - 26,
        xmin: 2,
        xmax: 1024,
        xlog: true,
        ymin: 1e-3,
        ymax: 20,
        ylog: true,
        xticks: [2, 8, 32, 128, 512],
        yticks: [1e-3, 1e-2, 1e-1, 1, 10],
        xfmt: (v) => String(v),
        yfmt: (v) => (v >= 1 ? String(v) : v >= 0.1 ? '0.1' : v >= 0.01 ? '0.01' : '0.001'),
        xlabel: 'dimension d',
        ylabel: 'relative MSE',
        margin: { l: 58, r: 16, t: 8, b: 38 },
      });
      plotL = P;
      P.frame();
      const sel = state.b;
      // target bands
      for (let b = 1; b <= 4; b++) {
        const on = b === sel;
        const lo = 4 ** -b;
        const hi = TQ_UPPER * lo;
        P.clip(() => {
          g.fillStyle = alpha('teal', on ? 0.28 : 0.07);
          g.fillRect(P.px, P.ty(hi), P.pw, P.ty(lo) - P.ty(hi));
        });
        if (on) {
          P.hline(lo, { color: 'teal', dash: null, width: 1.5 });
          P.hline(hi, { color: 'teal', dash: [5, 4], width: 1.5 });
        }
      }
      P.hline(1, { color: 'ink-2', dash: [2, 4], width: 1.2 });
      label(g, '100%: x̃ = 0 would do as well', P.px + 8, P.ty(1) - 11, { color: 'ink-2', size: 12, weight: 500 });
      if (curves) {
        for (let b = 1; b <= 4; b++) {
          if (b === sel) continue;
          P.line(curves.ds, curves.fixed[b], { color: 'red', width: 1.2, alpha: 0.22 });
        }
        P.line(curves.ds, curves.minmax[sel], { color: 'red', width: 2.2, dash: [7, 5], alpha: 0.85 });
        P.line(curves.ds, curves.fixed[sel], { color: 'red', width: 3 });
      }
      const lo = 4 ** -sel;
      label(g, `b = ${sel} target`, P.px + P.pw - 8, P.ty(TQ_UPPER * lo) - 12, { color: 'teal', size: 12.5, align: 'right', weight: 600 });
      if (sel === 1) label(g, '↑ d − 1', P.tx(5), P.py + 14, { color: 'red', size: 12.5, weight: 600 });

      // hover readout
      if (state.hoverD && curves) {
        const d = state.hoverD;
        let k = 0;
        while (k < curves.ds.length - 1 && curves.ds[k] < d) k++;
        const dd = curves.ds[k];
        const xx = P.tx(dd);
        line(g, xx, P.py, xx, P.py + P.ph, { color: 'ink-3', dash: [3, 3] });
        const f = curves.fixed[sel][k];
        const m = curves.minmax[sel][k];
        const tx = xx + (xx > P.px + P.pw * 0.6 ? -10 : 10);
        const al = xx > P.px + P.pw * 0.6 ? 'right' : 'left';
        label(g, `d = ${dd}`, tx, P.py + P.ph - 52, { color: 'ink', size: 12.5, align: al, font: 'mono' });
        label(g, `fixed ${pct(f)}`, tx, P.py + P.ph - 34, { color: 'red', size: 12.5, align: al, font: 'mono' });
        label(g, `min–max ${pct(m)}`, tx, P.py + P.ph - 16, { color: 'red', size: 12.5, align: al, font: 'mono' });
      }
    });
    cvL.canvas.addEventListener('pointermove', (e) => {
      if (!plotL) return;
      const r = cvL.canvas.getBoundingClientRect();
      const px = e.clientX - r.left;
      state.hoverD = px >= plotL.px && px <= plotL.px + plotL.pw ? plotL.ix(px) : null;
      cvL.request();
    });
    cvL.canvas.addEventListener('pointerleave', () => {
      state.hoverD = null;
      cvL.request();
    });

    // ---------- (ii) one outlier channel under min–max scaling ----------
    const rAll = readout('rel. MSE · all 64');
    const rRest = readout('rel. MSE · other 63');
    const rZero = readout('rounded to 0');
    const vR = viz('', `min–max · $d=${D_OUT}$ · synthetic`);
    Object.assign(vR.style, { flex: '1 1 0', height: 'auto', minHeight: '260px' });
    const cvR = createCanvas(vR, (g, w, h) => {
      const { b, A } = state;
      const x = Float64Array.from(noise);
      x[J_OUT] = A;
      const { xq, scale } = naiveQuantize(x, b, 'minmax');
      const L = 46;
      const R = w - 14;
      const T = 34;
      const B = h - 16;
      const ymax = Math.max(scale, 3.2) * 1.08;
      const sy = (v) => T + ((ymax - v) / (2 * ymax)) * (B - T);
      for (const lv of naiveLevels(b, scale)) line(g, L, sy(lv), R, sy(lv), { color: 'ink-3', alpha: lv === 0 ? 0.6 : 0.3, dash: lv === 0 ? null : [4, 5] });
      for (const t of [-Math.round(ymax * 0.9), 0, Math.round(ymax * 0.9)]) text(g, String(t).replace('-', '−'), L - 8, sy(t), { color: 'ink-3', size: 11.5, align: 'right', font: 'mono' });
      const bw = (R - L) / D_OUT;
      const y0 = sy(0);
      let eAll = 0;
      let nAll = 0;
      let eRest = 0;
      let nRest = 0;
      let zeros = 0;
      for (let i = 0; i < D_OUT; i++) {
        const x0 = L + i * bw;
        const ob = bw * 0.8;
        const qb = bw * 0.42;
        g.fillStyle = alpha(i === J_OUT ? 'yellow' : 'accent', i === J_OUT ? 0.7 : 0.6);
        g.fillRect(x0 + (bw - ob) / 2, Math.min(y0, sy(x[i])), ob, Math.abs(sy(x[i]) - y0));
        g.fillStyle = alpha('teal', 0.95);
        g.fillRect(x0 + (bw - qb) / 2, Math.min(y0, sy(xq[i])), qb, Math.max(Math.abs(sy(xq[i]) - y0), xq[i] === 0 ? 0 : 1));
        if (xq[i] === 0) line(g, x0 + (bw - qb) / 2, y0, x0 + (bw + qb) / 2, y0, { color: 'teal', width: 2 });
        const ex = x0 + (bw + ob) / 2 - 1;
        if (Math.abs(x[i] - xq[i]) > 1e-9) line(g, ex, sy(x[i]), ex, sy(xq[i]), { color: 'violet', width: 2 });
        const e = (x[i] - xq[i]) ** 2;
        eAll += e;
        nAll += x[i] ** 2;
        if (i !== J_OUT) {
          eRest += e;
          nRest += x[i] ** 2;
          if (xq[i] === 0) zeros++;
        }
      }
      label(g, `outlier A = ${A}`, L + (J_OUT + 1) * bw + 6, sy(A) + (A > 0.8 * ymax ? 14 : -12), { color: 'yellow', size: 13, weight: 600 });
      const ea = eAll / nAll;
      const er = eRest / nRest;
      rAll.set(pct(ea), ea > 0.5 ? 'red' : ea > 0.1 ? 'yellow' : 'teal');
      rRest.set(pct(er), er > 0.5 ? 'red' : er > 0.1 ? 'yellow' : 'teal');
      rZero.set(`${zeros} / ${D_OUT - 1}`, zeros > 31 ? 'red' : '');
    });

    const segB = segmented({
      label: 'bits $b$',
      options: [1, 2, 3, 4],
      value: state.b,
      onChange: (v) => {
        state.b = v;
        cvL.request();
        cvR.request();
      },
    });
    const sA = slider({
      label: 'outlier $A$',
      min: 1,
      max: 30,
      step: 0.5,
      value: state.A,
      onInput: (v) => {
        state.A = v;
        cvR.request();
      },
    });

    const left = el(
      'div.col',
      { 'data-reveal': '1', style: { minHeight: '0' } },
      vL,
      el(
        'div.row',
        { style: { justifyContent: 'space-between' } },
        segB.el,
        legend([
          { label: 'fixed grid', color: 'red' },
          { label: 'min–max', color: 'red', dash: true },
          { label: '$4^{-b}$ … $2.72\\cdot4^{-b}$', color: 'teal' },
        ]),
      ),
      el('div', { style: { fontSize: '0.95em' } }, eq('\\|x-\\tilde x\\|^2\\le d\\,\\frac{\\Delta^2}{4},\\qquad \\Delta=\\frac{2\\max_i|x_i|}{2^b-1}')),
    );
    const right = el(
      'div.col',
      { 'data-reveal': '2', style: { minHeight: '0' } },
      vR,
      sA.el,
      el('div.readouts', {}, rAll.el, rRest.el, rZero.el),
    );
    body.append(el('div.stage.even', { style: { minHeight: '0' } }, left, right));
    ctx.onEnter(() => {
      computeCurves();
      cvL.request();
      cvR.request();
    });
  },
};

import { el, eq, slider, segmented, toggle, button, readout, viz, legend } from '../lib/ui.js';
import { createCanvas, View2D, line, text, label, circle, arrow, dot, draggable, alpha } from '../lib/canvas.js';
import { RNG } from '../lib/math.js';
import { naiveLevels, naiveQuantize, naiveRound, relErr, pct } from './_shared-naive.js';

const PRESETS = [
  { value: 'gauss', label: 'Gaussian' },
  { value: 'spike', label: 'one spike' },
  { value: 'spread', label: 'spread $\\pm1/\\sqrt d$' },
  { value: 'outlier', label: 'outlier + noise' },
];

function makeVector(kind, d, seed) {
  const rng = new RNG(seed);
  const x = new Float64Array(d);
  if (kind === 'gauss') for (let i = 0; i < d; i++) x[i] = rng.normal();
  else if (kind === 'spike') x[rng.int(d)] = rng.next() < 0.5 ? -1 : 1;
  else if (kind === 'spread') for (let i = 0; i < d; i++) x[i] = (rng.next() < 0.5 ? -1 : 1) / Math.sqrt(d);
  else {
    for (let i = 0; i < d; i++) x[i] = rng.normal();
    x[Math.floor(d * 0.3)] = 10; // one channel ~10× the others (illustrative)
  }
  return x;
}

export default {
  id: 'naive',
  chapter: 'Naive',
  kicker: 'First attempt',
  title: 'Normalize, then round',
  notes: `The first thing anyone would try. Store the norm ‖x‖ as one float, divide it out so every coordinate lies in [−1, 1], round each coordinate to a uniform b-bit grid, and multiply the norm back in. Cost: b bits per coordinate plus 16 bits for the norm, i.e. b + 16/d bits per coordinate.

Our grid (the gray lines): step Δ = 2/(2^b − 1), round to the nearest multiple of Δ. For b ≥ 2 that gives 2^b − 1 levels including 0 (one code unused, like symmetric INT-b); 1 bit cannot hold a symmetric zero level, so b = 1 is sign × scale. Either way each coordinate is off by at most Δ/2.

Things to show, in order:
1. Gaussian, d = 8, b = 3: works fine. Now slide d up to 128. The orange bars (coordinates of x/‖x‖) shrink like 1/√d, while the grid stays put. Soon almost every coordinate is below half a step and rounds to 0 ("rounded to 0" readout). Error heads to 100%.
2. Toggle min–max scaling (KIVI-style: store max|x_i| instead of the norm and stretch the grid to [−max, max]). Gaussian is fine again at any d: the grid now adapts to the vector.
3. one spike: fine in both modes. The zeros stay exactly 0; the only error is the spike itself, clamped to the top level (1 − Δ/2): 11% at b = 2, 2% at b = 3, 0.4% at b = 4. (At b = 1 the zeros become ±1 and the error explodes to d − 1: sign quantization has no zero.)
4. spread: in fixed mode it is the worst case (every coordinate is tiny, so everything rounds to 0 once 1/√d < Δ/2, i.e. d > (2^b − 1)²); with min–max it is easy (every coordinate equals ±max, so only the same small clamp error).
5. outlier channel + noise with min–max on: one channel at 10 stretches the grid, and the other channels collapse onto the zero level. Hold that thought for the next slide.
The inset is the same game in d = 2: drag x around the unit circle and watch which grid point it snaps to. With min–max on, the grid rescales with x.
Takeaway: each naive variant has inputs it handles well and inputs it destroys. Nothing here is a guarantee about every input.`,
  mount(body, ctx) {
    const state = { kind: 'gauss', d: 32, b: 3, minmax: false, seed: 1, a2: 0.5 };
    let x = makeVector(state.kind, state.d, state.seed);
    const regen = () => {
      x = makeVector(state.kind, state.d, state.seed);
      cv.request();
    };

    const rErr = readout('$\\|x-\\tilde x\\|^2/\\|x\\|^2$');
    const rBits = readout('bits/coord $b+16/d$');
    const rZero = readout('rounded to 0');

    const eqFixed = eq('\\tilde x=\\|x\\|\\cdot\\mathrm{round}_b\\!\\big(x/\\|x\\|\\big)');
    const eqMinmax = eq('\\tilde x=m\\cdot\\mathrm{round}_b\\!\\big(x/m\\big),\\ \\ m=\\max_i|x_i|');
    eqMinmax.style.fontSize = '1em';

    const v = viz('', 'live · synthetic vector');
    Object.assign(v.style, { flex: '1 1 0', height: 'auto', minHeight: '320px' });

    const cv = createCanvas(v, (g, w, h) => {
      const { d, b, minmax } = state;
      const mode = minmax ? 'minmax' : 'fixed';
      const { xq, grid, gridQ } = naiveQuantize(x, b, mode);

      // ---- main bar chart in "grid units" (value the rounder sees) ----
      const L = 64;
      const R = w - 18;
      const T = 40;
      const B = h - 34;
      const ymax = 1.12;
      const sy = (val) => T + ((ymax - val) / (2 * ymax)) * (B - T);
      const bw = (R - L) / d;
      // levels
      const levels = naiveLevels(b, 1);
      for (const lv of levels) {
        line(g, L, sy(lv), R, sy(lv), { color: lv === 0 ? 'ink-3' : 'ink-3', alpha: lv === 0 ? 0.55 : 0.32, dash: lv === 0 ? null : [4, 5] });
      }
      // ±1 frame
      line(g, L, sy(1), R, sy(1), { color: 'line-2', width: 1 });
      line(g, L, sy(-1), R, sy(-1), { color: 'line-2', width: 1 });
      text(g, '+1', L - 10, sy(1), { color: 'ink-3', size: 12, align: 'right', font: 'mono' });
      text(g, '0', L - 10, sy(0), { color: 'ink-3', size: 12, align: 'right', font: 'mono' });
      text(g, '−1', L - 10, sy(-1), { color: 'ink-3', size: 12, align: 'right', font: 'mono' });
      g.save();
      g.translate(18, (T + B) / 2);
      g.rotate(-Math.PI / 2);
      text(g, minmax ? 'xᵢ / max|x|' : 'xᵢ / ‖x‖', 0, 0, { color: 'ink-2', size: 13, align: 'center' });
      g.restore();

      for (let i = 0; i < d; i++) {
        const x0 = L + i * bw;
        const ob = Math.max(1, bw * 0.78);
        const qb = Math.max(1, bw * 0.4);
        const y0 = sy(0);
        // original
        g.fillStyle = alpha('accent', 0.55);
        g.fillRect(x0 + (bw - ob) / 2, Math.min(y0, sy(grid[i])), ob, Math.abs(sy(grid[i]) - y0));
        // quantized
        g.fillStyle = alpha('teal', 0.95);
        const qh = Math.abs(sy(gridQ[i]) - y0);
        g.fillRect(x0 + (bw - qb) / 2, Math.min(y0, sy(gridQ[i])), qb, Math.max(qh, gridQ[i] === 0 ? 0 : 1));
        if (gridQ[i] === 0) line(g, x0 + (bw - qb) / 2, y0, x0 + (bw + qb) / 2, y0, { color: 'teal', width: 2 });
        // error
        if (Math.abs(grid[i] - gridQ[i]) > 1e-9) {
          const ex = x0 + (bw + ob) / 2 - Math.max(1, bw * 0.06);
          line(g, ex, sy(grid[i]), ex, sy(gridQ[i]), { color: 'violet', width: Math.max(1.5, Math.min(3.5, bw * 0.16)) });
        }
      }
      text(g, `coordinate i = 1 … ${d}`, (L + R) / 2, h - 14, { color: 'ink-3', size: 12, align: 'center' });

      // ---- readouts ----
      const e = relErr(x, xq);
      rErr.set(pct(e), e > 0.5 ? 'red' : e > 0.1 ? 'yellow' : 'teal');
      rBits.set((b + 16 / d).toFixed(2));
      let zeros = 0;
      let nonzero = 0;
      for (let i = 0; i < d; i++) {
        if (x[i] !== 0) {
          nonzero++;
          if (gridQ[i] === 0) zeros++;
        }
      }
      rZero.set(`${zeros} / ${nonzero}`, zeros > nonzero / 2 ? 'red' : '');
      eqFixed.style.opacity = minmax ? '0.3' : '1';
      eqMinmax.style.opacity = minmax ? '1' : '0.3';
    });

    // ---- 2-D inset: the same rounding in d = 2 ----
    const iv = viz('', '$d=2$ · drag $x$');
    Object.assign(iv.style, { flex: '1 1 0', height: 'auto', minHeight: '170px' });
    let view2 = null;
    const icv = createCanvas(iv, (g, w, h) => {
      const { b, minmax } = state;
      view2 = new View2D({ w, h, xmin: -1.2, xmax: 1.2, ymin: -1.2, ymax: 1.2, pad: 12 });
      const view = view2;
      const p = [Math.cos(state.a2), Math.sin(state.a2)];
      const m2 = minmax ? Math.max(Math.abs(p[0]), Math.abs(p[1])) : 1;
      circle(g, view.sx(0), view.sy(0), view.len(1), { stroke: 'line-2', width: 1.2 });
      const lv2 = naiveLevels(b, m2);
      for (const a of lv2) for (const c of lv2) dot(g, view.sx(a), view.sy(c), 2.4, 'ink-3');
      const pq = [naiveRound(p[0], b, m2), naiveRound(p[1], b, m2)];
      arrow(g, view.sx(0), view.sy(0), view.sx(p[0]), view.sy(p[1]), { color: 'accent', width: 2.5, head: 9 });
      if (Math.hypot(pq[0], pq[1]) > 1e-9) arrow(g, view.sx(0), view.sy(0), view.sx(pq[0]), view.sy(pq[1]), { color: 'teal', width: 2, head: 8 });
      else circle(g, view.sx(0), view.sy(0), 5, { stroke: 'teal', width: 2 });
      line(g, view.sx(p[0]), view.sy(p[1]), view.sx(pq[0]), view.sy(pq[1]), { color: 'violet', width: 2 });
      circle(g, view.sx(p[0]), view.sy(p[1]), 6.5, { fill: 'accent' });
    });
    draggable(icv.canvas, {
      pick(px, py) {
        if (!view2) return null;
        const p = [Math.cos(state.a2), Math.sin(state.a2)];
        return Math.hypot(px - view2.sx(p[0]), py - view2.sy(p[1])) < 20 ? 'p' : null;
      },
      move(_, px, py) {
        state.a2 = Math.atan2(view2.wy(py), view2.wx(px));
        icv.request();
      },
    });
    const redrawAll = () => {
      cv.request();
      icv.request();
    };

    const seg = segmented({
      options: PRESETS,
      value: state.kind,
      onChange: (k) => {
        state.kind = k;
        regen();
      },
    });
    seg.el.querySelectorAll('button').forEach((bt) => (bt.style.padding = '5px 10px'));
    const sD = slider({
      label: '$d$',
      min: 8,
      max: 128,
      step: 4,
      value: state.d,
      onInput: (val) => {
        state.d = val;
        regen();
      },
    });
    const sB = slider({
      label: 'bits $b$',
      min: 1,
      max: 4,
      step: 1,
      value: state.b,
      onInput: (val) => {
        state.b = val;
        redrawAll();
      },
    });
    const tMM = toggle({
      label: 'min–max scaling (store $\\max_i|x_i|$, KIVI-style)',
      onChange: (on) => {
        state.minmax = on;
        redrawAll();
      },
    });
    const reroll = button('re-draw', () => {
      state.seed++;
      regen();
    });

    const panel = el(
      'div.panel',
      { 'data-reveal': '2', style: { minHeight: '0' } },
      el('div', {}, eqFixed, eqMinmax),
      seg.el,
      sD.el,
      sB.el,
      el('div.row', {}, tMM.el),
      el('div.readouts', {}, rErr.el, rBits.el, rZero.el),
      el('div.row', { style: { alignItems: 'stretch', flex: '1 1 0', minHeight: '180px', flexWrap: 'nowrap' } }, iv, el('div.col', { style: { justifyContent: 'flex-end' } }, reroll)),
      legend([
        { label: '$x$', color: 'accent' },
        { label: '$\\tilde x$', color: 'teal' },
        { label: 'error', color: 'violet' },
        { label: 'grid levels', color: 'ink-3', dash: true },
      ]),
    );

    body.append(el('div.stage', { style: { minHeight: '0' } }, v, panel));
    ctx.onEnter(redrawAll);
  },
};

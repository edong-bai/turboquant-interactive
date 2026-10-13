import { el, rich, eq, slider, toggle, button, readout, viz, legend, texify } from '../lib/ui.js';
import { createCanvas, line, label, text, alpha, color } from '../lib/canvas.js';
import { Plot } from '../lib/plot.js';
import * as M from '../lib/math.js';

// Reference centroids in units of 1/√d, large-d limit (positive half).
// b = 1, 2: stated in the paper; b = 3, 4: Gaussian Lloyd–Max (our check, docs/paper-notes-turboquant.md).
const REF = {
  1: { vals: [0.7979], src: 'paper: $\\pm\\sqrt{2/\\pi}\\approx\\pm0.798$' },
  2: { vals: [0.4528, 1.5104], src: 'paper: $\\pm0.453,\\ \\pm1.51$' },
  3: { vals: [0.2451, 0.756, 1.3439, 2.1519], src: 'Gaussian limit (our check)' },
  4: { vals: [0.1284, 0.388, 0.6568, 0.9423, 1.2562, 1.618, 2.069, 2.7326], src: 'Gaussian limit (our check)' },
};

/**
 * Continuous Lloyd–Max on f_X for one d. Cumulative moments F0 = ∫f, F1 = ∫x f, F2 = ∫x² f on a
 * fine table, interpolated within a cell, so bucket integrals are continuous in the boundaries.
 * (A plain gridded Lloyd step stalls once the updates are smaller than one grid cell, which at
 * b = 3, 4 happens visibly short of the optimum.)
 */
const momentCache = new Map();
function momentsFor(d) {
  if (momentCache.has(d)) return momentCache.get(d);
  const s = 1 / Math.sqrt(d);
  const lo = d <= 4 ? -1 : Math.max(-1, -10 * s);
  const hi = -lo;
  const n = 20000;
  const h = (hi - lo) / n;
  const F0 = new Float64Array(n + 1);
  const F1 = new Float64Array(n + 1);
  const F2 = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    const x = lo + (i + 0.5) * h;
    const w = M.sphereCoordPdf(x, d) * h;
    F0[i + 1] = F0[i] + w;
    F1[i + 1] = F1[i] + w * x;
    F2[i + 1] = F2[i] + w * x * x;
  }
  const Z = F0[n];
  for (let i = 0; i <= n; i++) {
    F0[i] /= Z;
    F1[i] /= Z;
    F2[i] /= Z;
  }
  const at = (F, x) => {
    const t = M.clamp((x - lo) / h, 0, n);
    const i = Math.min(n - 1, Math.floor(t));
    return F[i] + (F[i + 1] - F[i]) * (t - i);
  };
  const edges = (c) => [lo, ...M.midpoints(c), hi];
  const m = {
    lo,
    hi,
    step(c) {
      const e = edges(c);
      const out = new Float64Array(c.length);
      for (let i = 0; i < c.length; i++) {
        const w = at(F0, e[i + 1]) - at(F0, e[i]);
        out[i] = w > 1e-15 ? (at(F1, e[i + 1]) - at(F1, e[i])) / w : c[i];
      }
      return out;
    },
    cost(c) {
      const e = edges(c);
      let tot = 0;
      for (let i = 0; i < c.length; i++) {
        const a0 = at(F0, e[i + 1]) - at(F0, e[i]);
        const a1 = at(F1, e[i + 1]) - at(F1, e[i]);
        const a2 = at(F2, e[i + 1]) - at(F2, e[i]);
        tot += a2 - 2 * c[i] * a1 + c[i] * c[i] * a0;
      }
      return Math.max(0, tot);
    },
  };
  momentCache.set(d, m);
  return m;
}

const convCache = new Map();
/** Fully converged Lloyd–Max codebook for (d, b), from the uniform start. */
function convergedFor(d, b) {
  const key = `${d}:${b}`;
  if (convCache.has(key)) return convCache.get(key);
  const m = momentsFor(d);
  let c = uniformInit(d, b);
  for (let it = 0; it < 60000; it++) {
    const n = m.step(c);
    let delta = 0;
    for (let i = 0; i < c.length; i++) delta = Math.max(delta, Math.abs(n[i] - c[i]));
    c = n;
    if (delta * Math.sqrt(d) < 1e-10) break;
  }
  const res = { centroids: c, cost: m.cost(c) };
  convCache.set(key, res);
  return res;
}

/** Uniform grid: 2^b cells of equal width tiling ±3/√d (clipped to ±1), levels at cell centers. */
const uniformInit = (d, b) => {
  const s = 1 / Math.sqrt(d);
  return M.initCodebook(2 ** b, Math.max(-1, -3 * s), Math.min(1, 3 * s));
};

export default {
  id: 'lloyd-max',
  chapter: 'TurboQuant',
  kicker: 'Optimal scalar codebook',
  title: 'Lloyd–Max: k-means on a known density',
  notes: `We now know exactly what every rotated coordinate looks like: the density f_X for this d. So the best b-bit scalar quantizer for it is a 1-D k-means problem on a KNOWN density, Eq. 4 in the paper. That's the Lloyd–Max algorithm (Lloyd 1957/1982, Max 1960).

The plot: the x-axis is in units of 1/√d (so the picture looks the same for every large d). Gray curve: the density. Teal: the 2^b centroids (reconstruction levels). Dashed: the bucket boundaries.

Lloyd's two alternating steps, each of which can only lower the cost:
1. Fix the centroids: each x goes to its nearest centroid, so boundaries sit at midpoints between neighbors. Toggle "why midpoints?": the cost of sending x to c_i is the parabola (x − c_i)²; two neighboring parabolas cross exactly halfway between their vertices, and the lower envelope switches there.
2. Fix the buckets: the best single point for a bucket is its conditional mean (center of mass of f over that bucket).

What to do: start from the uniform grid (evenly spaced cells across ±3/√d; that's our choice of "naive" grid, not optimized). Press Step a few times: the centroids slide inward and crowd toward the peak, where the mass is (spacing becomes uneven: tight in the middle, wide in the tails). Press Run to converge. Toggle "show integrand": orange is |x − c_i|² f(x) inside each bucket (in √d units, so its total area IS the readout d·C(f_X, b); for larger b it is drawn vertically exaggerated by a fixed power of 2, shown in the label, so it stays visible). Watch the orange area shrink as you step.

Readouts: d·C is the TurboQuant_mse distortion for a unit vector (Theorem 1: D_mse = d·C(f_X, b)). At large d it approaches 0.3634, 0.1175, 0.03455, 0.00950 for b = 1..4 (Gaussian Lloyd–Max, our check; the paper rounds to 0.36, 0.117, 0.03, 0.009). Compare with the uniform grid with the same number of levels and with 4^{−b}, the information-theoretic lower bound (Theorem 3), and the paper's upper bound (√3·π/2)·4^{−b} ≈ 2.72·4^{−b}. The gap to the lower bound is at most ≈ 2.7×, and only 1.45× at b = 1.

Centroids ×√d: compare with the paper: b = 1 gives ±√(2/π)/√d ≈ ±0.798/√d (the mean of |N(0,1)|); b = 2 gives ±0.453/√d, ±1.51/√d. Small d differs (try d = 3: the density is flat on [−1, 1], so the optimum is a uniform grid on [−1, 1]).

The key systems point: this is solved ONCE per (d, b), offline, from the formula. No data, no calibration, no per-block statistics. In deployment it's a lookup table of 2^b numbers.`,
  mount(body, ctx) {
    const state = { b: 2, ld: 8, running: false, integrand: true, parabolas: false };
    const D = () => Math.max(3, Math.round(2 ** state.ld));
    let grid = null;
    let c = null; // current centroids (x units)
    let from = null; // tween start
    let tween = 1;
    let iter = 0;
    let history = [];
    let uniformCost = 0;
    let converged = null;

    const shown = () => {
      if (!from || tween >= 1) return c;
      const s = M.smoothstep(tween);
      return c.map((v, i) => from[i] + (v - from[i]) * s);
    };
    const reset = () => {
      const d = D();
      grid = momentsFor(d);
      c = uniformInit(d, state.b);
      from = null;
      tween = 1;
      iter = 0;
      uniformCost = d * grid.cost(c);
      history = [[0, uniformCost]];
      converged = convergedFor(d, state.b);
      update();
    };
    const step = (animate = true) => {
      const next = grid.step(c);
      from = animate ? shown() : null;
      tween = animate ? 0 : 1;
      let delta = 0;
      for (let i = 0; i < c.length; i++) delta = Math.max(delta, Math.abs(next[i] - c[i]));
      c = next;
      iter++;
      // keep the sparkline light: every step early on, then thinned out
      if (iter <= 60 || iter % Math.ceil(iter / 60) === 0) history.push([iter, D() * grid.cost(c)]);
      return delta * Math.sqrt(D());
    };

    // Vertical exaggeration of the integrand, fixed per (d, b) from the uniform start (a power of 2).
    const ampCache = { key: '', amp: 1 };
    const ampFor = (d, pdfU, ymax, lo) => {
      const key = `${d}:${state.b}`;
      if (ampCache.key === key) return ampCache.amp;
      const k = Math.sqrt(d);
      const c0 = Array.from(uniformInit(d, state.b), (v) => v * k);
      const e = [lo, ...Array.from(M.midpoints(c0)), -lo];
      let mx = 0;
      for (let i = 0; i < c0.length; i++)
        for (let j = 0; j <= 60; j++) {
          const u = e[i] + ((e[i + 1] - e[i]) * j) / 60;
          mx = Math.max(mx, (u - c0[i]) ** 2 * pdfU(u));
        }
      const amp = mx > 0 ? Math.max(1, 2 ** Math.floor(Math.log2((0.55 * ymax) / mx))) : 1;
      Object.assign(ampCache, { key, amp });
      return amp;
    };

    // ---------- main plot ----------
    const vMain = viz();
    const cvMain = createCanvas(vMain, (g, w, h) => {
      if (w < 60 || h < 60 || !grid) return;
      const d = D();
      const k = Math.sqrt(d);
      const cc = shown();
      const cu = Array.from(cc, (v) => v * k); // centroids in √d units
      const bu = Array.from(M.midpoints(cc), (v) => v * k);
      const pdfU = (u) => M.sphereCoordPdf(u / k, d) / k;
      const R = Math.min(k * 1.04, 4.2);
      const ymax = Math.max(pdfU(0), 0.2) * 1.3;
      const p = new Plot(g, { x: 0, y: 30, w, h: h - 30, xmin: -R, xmax: R, ymin: 0, ymax, xlabel: 'x · √d', margin: { l: 52, r: 16, t: 8, b: 42 } });
      p.frame();
      const lo = Math.max(-R, -k);
      // integrand per bucket
      if (state.integrand) {
        const edges = [lo, ...bu, -lo];
        const amp = ampFor(d, pdfU, ymax, lo); // fixed per (d, b): vertical exaggeration so b = 3, 4 stay visible
        for (let i = 0; i < cu.length; i++) {
          p.area((u) => amp * (u - cu[i]) ** 2 * pdfU(u), { lo: edges[i], hi: edges[i + 1], samples: 120, color: 'accent', alpha: i % 2 ? 0.42 : 0.62 });
        }
      }
      // density
      p.area(pdfU, { lo, hi: -lo, samples: 400, color: 'ink-3', alpha: 0.12 });
      p.fn(pdfU, { lo, hi: -lo, samples: 400, color: 'ink-2', width: 2 });
      // boundaries
      const pb = cu.length === 2 ? 0 : cu.length / 2; // boundary shown in the inset (first non-negative one)
      bu.forEach((v, i) => p.vline(v, { color: state.parabolas && i === pb ? 'yellow' : 'ink-3', width: state.parabolas && i === pb ? 2 : 1.2, dash: [5, 5] }));
      // centroids
      p.clip(() => {
        for (const v of cu) {
          line(g, p.tx(v), p.ty(0), p.tx(v), p.ty(pdfU(v)), { color: 'teal', width: 2.5 });
          g.fillStyle = color('teal');
          g.beginPath();
          g.moveTo(p.tx(v), p.ty(0) - 11);
          g.lineTo(p.tx(v) - 6, p.ty(0));
          g.lineTo(p.tx(v) + 6, p.ty(0));
          g.closePath();
          g.fill();
        }
      });
      if (cu.length <= 4) {
        for (const v of cu) label(g, v.toFixed(3), p.tx(v), p.ty(pdfU(v)) - 12, { color: 'teal', size: 12, align: 'center', font: 'mono' });
      }
      // header
      text(g, `d = ${d} · b = ${state.b} · ${cu.length} levels · step ${iter}`, 16, 17, { color: 'ink-2', size: 13, font: 'mono' });
      if (state.integrand)
        text(g, `orange: |x−cᵢ|²f${ampCache.amp > 1 ? ` (drawn ×${ampCache.amp})` : ''} · true area = d·C = ${(d * grid.cost(cc)).toFixed(4)}`, w - 16, 17, {
          color: 'accent',
          size: 13,
          align: 'right',
          font: 'mono',
        });

      // parabola inset: why boundaries sit at midpoints
      if (state.parabolas) {
        const i0 = cu.length === 2 ? 0 : cu.length / 2;
        const a = cu[i0];
        const b2 = cu[i0 + 1];
        const m = (a + b2) / 2;
        const gap = b2 - a;
        const iw = Math.min(300, w * 0.36);
        const ih = Math.min(190, h * 0.4);
        const ix = w - iw - 18;
        const iy = 40;
        g.save();
        g.fillStyle = 'rgba(8,14,28,0.92)';
        g.fillRect(ix, iy, iw, ih);
        g.strokeStyle = color('yellow');
        g.globalAlpha = 0.6;
        g.strokeRect(ix, iy, iw, ih);
        g.restore();
        const q = new Plot(g, {
          x: ix,
          y: iy + 18,
          w: iw,
          h: ih - 18,
          xmin: a - 0.7 * gap,
          xmax: b2 + 0.7 * gap,
          ymin: 0,
          ymax: (1.2 * gap) ** 2,
          xticks: [],
          yticks: [],
          margin: { l: 10, r: 10, t: 6, b: 22 },
        });
        q.fn((u) => (u - a) ** 2, { color: 'teal', width: 1.5, alpha: 0.45 });
        q.fn((u) => (u - b2) ** 2, { color: 'teal', width: 1.5, alpha: 0.45 });
        q.fn((u) => Math.min((u - a) ** 2, (u - b2) ** 2), { color: 'yellow', width: 2.5 });
        q.vline(m, { color: 'yellow', dash: [4, 4] });
        for (const v of [a, b2]) {
          g.fillStyle = color('teal');
          g.beginPath();
          g.arc(q.tx(v), q.ty(0), 4, 0, Math.PI * 2);
          g.fill();
        }
        text(g, '(x − cᵢ)²  vs  (x − cᵢ₊₁)²', ix + iw / 2, iy + 12, { color: 'ink-2', size: 12, align: 'center' });
        text(g, 'cᵢ', q.tx(a), q.py + q.ph + 12, { color: 'teal', size: 12, align: 'center' });
        text(g, 'cᵢ₊₁', q.tx(b2), q.py + q.ph + 12, { color: 'teal', size: 12, align: 'center' });
        text(g, 'midpoint', q.tx(m), q.py + q.ph + 12, { color: 'yellow', size: 12, align: 'center' });
      }
    });

    // ---------- convergence sparkline ----------
    const vSpark = viz();
    vSpark.style.height = '128px';
    vSpark.style.minHeight = '0';
    vSpark.style.flex = '0 0 auto';
    const cvSpark = createCanvas(vSpark, (g, w, h) => {
      if (w < 60 || h < 40 || !history.length) return;
      const target = converged ? D() * converged.cost : history[history.length - 1][1];
      const lb = 4 ** -state.b;
      const ys = [...history.map((p) => p[1]), target, lb];
      const ymin = Math.min(...ys) * 0.8;
      const ymax = Math.max(...ys) * 1.25;
      const n = Math.max(30, iter + 1);
      const p = new Plot(g, {
        x: 0,
        y: 16,
        w,
        h: h - 16,
        xmin: 1,
        xmax: n,
        xlog: true,
        ymin,
        ymax,
        ylog: true,
        yticks: [],
        margin: { l: 12, r: 14, t: 6, b: 20 },
      });
      p.frame({ grid: false });
      p.hline(target, { color: 'teal', dash: [4, 4] });
      p.hline(lb, { color: 'ink-3', dash: [2, 4] });
      p.hline(history[0][1], { color: 'red', dash: [4, 4] });
      p.line(
        history.map((q) => q[0] + 1),
        history.map((q) => q[1]),
        { color: 'accent', width: 2 },
      );
      text(g, 'd·C vs step (log–log) · red: uniform · teal: optimum · gray: 4⁻ᵇ', 12, 10, { color: 'ink-3', size: 11 });
    });

    // ---------- controls ----------
    const rIter = readout('step');
    const rCur = readout('$d\\cdot\\mathcal C$ now');
    const rUni = readout('uniform grid');
    const rConv = readout('converged');
    const rLow = readout('$4^{-b}$ (lower)');
    const rUp = readout('$\\frac{\\sqrt3\\pi}{2}4^{-b}$ (upper)');
    const cTxt = rich('div.note', '');
    const refTxt = rich('div.note.muted', '');
    function update() {
      if (!grid) return;
      const d = D();
      rIter.set(String(iter));
      rCur.set((d * grid.cost(c)).toFixed(4), 'teal');
      rUni.set(uniformCost.toFixed(4), 'red');
      rConv.set(converged ? (d * converged.cost).toFixed(4) : '…');
      rLow.set((4 ** -state.b).toFixed(4));
      rUp.set(M.bounds.mseUpper(state.b).toFixed(4));
      const pos = Array.from(c)
        .filter((v) => v > 0)
        .map((v) => (v * Math.sqrt(d)).toFixed(3));
      cTxt.innerHTML = texify(`$\\sqrt d\\,c^{+}$: <span class="mono" style="color:var(--teal)">${pos.join(', ')}</span>`);
      const ref = REF[state.b];
      refTxt.innerHTML = texify(`${ref.src}${state.b > 2 ? ': ' + ref.vals.map((v) => v.toFixed(3)).join(', ') : ''}`);
      cvMain.request();
      cvSpark.request();
    }

    const sB = slider({ label: 'bits $b$', min: 1, max: 4, step: 1, value: state.b, onInput: (v) => ((state.b = v), stop(), reset()) });
    const sD = slider({
      label: '$d$',
      min: Math.log2(3),
      max: 10,
      step: 0.01,
      value: state.ld,
      format: (v) => `${Math.max(3, Math.round(2 ** v))}`,
      onInput: (v) => ((state.ld = v), stop(), reset()),
    });
    const bStep = button('Step', () => {
      stop();
      step(true);
      update();
    });
    const bRun = button('', () => (state.running ? stop() : run()), 'primary');
    const bReset = button('Reset', () => (stop(), reset()));
    function stop() {
      state.running = false;
      bRun.innerHTML = texify('▶ Run');
    }
    function run() {
      state.running = true;
      bRun.innerHTML = texify('❚❚ Pause');
    }
    stop();
    const tInt = toggle({ label: 'show integrand', value: state.integrand, onChange: (v) => ((state.integrand = v), cvMain.request()) });
    const tPar = toggle({ label: 'why midpoints?', value: state.parabolas, onChange: (v) => ((state.parabolas = v), cvMain.request()) });

    vMain.style.height = 'auto';
    vMain.style.flex = '1 1 0';
    vMain.style.minHeight = '300px';
    const left = el(
      'div.col',
      { style: { gap: '12px' } },
      vMain,
      el(
        'div.card',
        {
          'data-reveal': '1',
          style: { padding: '8px 18px', gap: '0', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' },
        },
        eq('\\mathcal C(f_X,b)=\\min_{c_1\\le\\dots\\le c_{2^b}}\\sum_{i}\\int_{\\frac{c_{i-1}+c_i}{2}}^{\\frac{c_i+c_{i+1}}{2}}|x-c_i|^2f_X(x)\\,dx'),
        rich('div.note', 'Solved once per $(d,b)$, offline: a lookup table. No data.', { style: { maxWidth: '20ch', color: 'var(--teal)' } }),
      ),
    );
    const panel = el(
      'div.col',
      { style: { gap: '10px', minHeight: '0' } },
      el(
        'div.panel',
        { style: { gap: '4px', padding: '10px 16px' } },
        sB.el,
        sD.el,
        el('div.row', { style: { gap: '8px' } }, bStep, bRun, bReset),
        el('div.row', { style: { gap: '14px', marginTop: '4px' } }, tInt.el, tPar.el),
      ),
      el('div.readouts', { style: { gridTemplateColumns: 'repeat(3, minmax(0,1fr))', gap: '8px' } }, rIter.el, rCur.el, rUni.el, rConv.el, rLow.el, rUp.el),
      el('div', { style: { display: 'flex', flexDirection: 'column', gap: '2px' } }, cTxt, refTxt),
      vSpark,
      legend([
        { label: 'density $f_X$', color: 'ink-2' },
        { label: 'centroids', color: 'teal' },
        { label: '$|x-c_i|^2f_X$', color: 'accent' },
      ]),
    );
    body.append(el('div.stage', { style: { minHeight: '0' } }, left, panel));

    ctx.onEnter(() => {
      if (!grid) reset();
    });
    let acc = 0;
    ctx.loop((dt) => {
      if (tween < 1) {
        tween = Math.min(1, tween + dt / 0.35);
        cvMain.redraw();
      }
      if (!state.running) return;
      acc += dt;
      // slow, animated steps first; then speed up so b = 4 converges in a few seconds
      if (iter < 8) {
        if (tween >= 1 && acc > 0.15) {
          acc = 0;
          step(true);
          update();
        }
      } else {
        let delta = 0;
        const per = Math.min(400, Math.ceil(iter / 3));
        for (let n = 0; n < per; n++) delta = step(false);
        update();
        if (delta < 1e-8 || iter > 60000) stop();
      }
    });
  },
};

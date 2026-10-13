import { el, rich, slider, segmented, toggle, button, readout, viz, legend, texify } from '../lib/ui.js';
import { createCanvas, View2D, circle, line, label, text, alpha } from '../lib/canvas.js';
import { Plot } from '../lib/plot.js';
import * as M from '../lib/math.js';
import { responsiveGrid } from './_shared-layout.js';

const NS = 20000; // Monte-Carlo points per d
const NSCAT = 3000; // points drawn in the scatter

/** Gamma(a, 1) sampler (Marsaglia–Tsang, with the a < 1 boost). */
function gammaSample(a, rng) {
  if (a <= 0) return 0;
  if (a < 1) return gammaSample(a + 1, rng) * Math.pow(rng.next(), 1 / a);
  const d = a - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x;
    let v;
    do {
      x = rng.normal();
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng.next();
    if (u < 1 - 0.0331 * x ** 4) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

/**
 * First two coordinates (y1, y2) of NS uniform points on S^{d−1}. Exact: y = g/‖g‖ with
 * g ~ N(0, I_d), where ‖g‖² = g1² + g2² + χ²_{d−2} and χ²_k = 2·Gamma(k/2).
 */
const cache = new Map();
function sampleD(d) {
  if (cache.has(d)) return cache.get(d);
  const rng = new M.RNG(1500 + d);
  const y1 = new Float64Array(NS);
  const y2 = new Float64Array(NS);
  for (let i = 0; i < NS; i++) {
    const a = rng.normal();
    const b = rng.normal();
    const rest = d > 2 ? 2 * gammaSample((d - 2) / 2, rng) : 0;
    const n = Math.sqrt(a * a + b * b + rest);
    y1[i] = a / n;
    y2[i] = b / n;
  }
  // statistics
  let s2 = 0;
  let s4 = 0;
  for (let i = 0; i < NS; i++) {
    s2 += y1[i] ** 2;
    s4 += y1[i] ** 4;
  }
  const m2 = s2 / NS;
  const m4 = s4 / NS;
  const corr = (u, v) => {
    const mu = M.mean(u);
    const mv = M.mean(v);
    let suv = 0;
    let suu = 0;
    let svv = 0;
    for (let i = 0; i < u.length; i++) {
      suv += (u[i] - mu) * (v[i] - mv);
      suu += (u[i] - mu) ** 2;
      svv += (v[i] - mv) ** 2;
    }
    return suv / Math.sqrt(suu * svv);
  };
  const sq1 = y1.map((v) => v * v);
  const sq2 = y2.map((v) => v * v);
  const res = { y1, y2, dVar: d * m2, kurt: m4 / (m2 * m2) - 3, corr: corr(y1, y2), corrSq: corr(sq1, sq2) };
  cache.set(d, res);
  return res;
}

export default {
  id: 'concentration',
  chapter: 'Rotation',
  kicker: 'Concentration of measure',
  title: 'High dimensions: a Gaussian, nearly independent',
  notes: `Now crank up the dimension. Left: the density of one coordinate of a uniform point on S^{d−1}. Teal is the exact formula (TurboQuant Lemma 1); orange bars are a live Monte-Carlo histogram (${NS.toLocaleString('en-US')} points per d, sampled when you first visit that d, then cached); the dashed line is the Gaussian N(0, 1/d).

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

Sampling detail: instead of drawing d Gaussians and normalizing, we draw g₁, g₂ and the rest of ‖g‖² as a chi-square with d−2 degrees of freedom. That is exactly the same distribution, just cheaper.`,
  mount(body, ctx) {
    const state = { ld: Math.log2(16), rescale: false, sweeping: false };
    let entered = false; // sample lazily, only once the slide is first shown
    const D = () => Math.max(2, Math.round(2 ** state.ld));

    // ---------- density ----------
    const vPdf = viz();
    const cvPdf = createCanvas(vPdf, (g, w, h) => {
      if (w < 60 || h < 60 || !entered) return;
      const d = D();
      const S = sampleD(d);
      const k = state.rescale ? Math.sqrt(d) : 1;
      const xr = state.rescale ? 4 : 1;
      const f = (u) => M.sphereCoordPdf(u / k, d) / k; // density in display units
      let ymax;
      if (d === 2) ymax = state.rescale ? 0.9 : 2.2;
      else ymax = f(0) * 1.3;
      const p = new Plot(g, {
        x: 0,
        y: 26,
        w,
        h: h - 26,
        xmin: -xr,
        xmax: xr,
        ymin: 0,
        ymax,
        xlabel: state.rescale ? '√d · y₁' : 'y₁',
        margin: { l: 56, r: 16, t: 10, b: 42 },
      });
      p.frame();
      const vals = new Float64Array(NS);
      for (let i = 0; i < NS; i++) vals[i] = S.y1[i] * k;
      p.bars(M.histogram(vals, -xr, xr, 80), -xr, xr, { color: 'accent', alpha: 0.45, gap: 1 });
      const lo = Math.max(-xr, -k + 1e-6);
      p.fn(f, { lo, hi: -lo, samples: 600, color: 'teal', width: 2.5 });
      const sig = state.rescale ? 1 : 1 / Math.sqrt(d);
      p.fn((u) => M.normalPdf(u, sig), { samples: 400, color: 'ink-2', width: 1.8, dash: [6, 5] });
      text(g, `d = ${d}`, 16, 16, { color: 'ink', size: 15, weight: 600, font: 'mono' });
      let tag = '';
      if (d === 2) tag = 'arcsine: 1/(π√(1−x²))';
      else if (d === 3) tag = 'flat: ½ (hat-box)';
      else if (d === 4) tag = 'semicircle: (2/π)√(1−x²)';
      else if (d >= 64) tag = '≈ 𝒩(0, 1/d)';
      if (tag) label(g, tag, 34 + `d = ${d}`.length * 9.5, 16, { color: 'teal', size: 14 });
      text(g, `${NS.toLocaleString('en-US')} sampled points (live)`, w - 14, 16, { color: 'ink-3', size: 12, align: 'right' });
    });

    // ---------- scatter (y1, y2) ----------
    const vSc = viz();
    const cvSc = createCanvas(vSc, (g, w, h) => {
      if (w < 60 || h < 60 || !entered) return;
      const d = D();
      const S = sampleD(d);
      const k = state.rescale ? Math.sqrt(d) : 1;
      const xr = state.rescale ? 4 : 1.15;
      const view = new View2D({ x: 0, y: 26, w, h: h - 26, xmin: -xr, xmax: xr, ymin: -xr, ymax: xr, pad: 22 });
      line(g, view.sx(-xr), view.sy(0), view.sx(xr), view.sy(0), { color: 'line-2' });
      line(g, view.sx(0), view.sy(-xr), view.sx(0), view.sy(xr), { color: 'line-2' });
      // the unit circle (in display units it has radius k)
      circle(g, view.sx(0), view.sy(0), view.len(k), { stroke: 'ink-3', width: 1, dash: [5, 5] });
      if (state.rescale) circle(g, view.sx(0), view.sy(0), view.len(1), { stroke: 'ink', width: 1.5, dash: [3, 4] });
      g.fillStyle = alpha('accent', d > 64 ? 0.35 : 0.5);
      for (let i = 0; i < NSCAT; i++) {
        g.fillRect(view.sx(S.y1[i] * k) - 1.1, view.sy(S.y2[i] * k) - 1.1, 2.2, 2.2);
      }
      text(g, state.rescale ? '(√d·y₁, √d·y₂)' : '(y₁, y₂)', 14, 16, { color: 'ink-2', size: 13 });
      text(g, state.rescale ? 'dotted: radius 1 · dashed: the sphere' : 'dashed: unit circle', w - 14, 16, { color: 'ink-3', size: 12, align: 'right' });
    });

    // ---------- controls & readouts ----------
    const rVar = readout('$d\\cdot\\Var(y_1)$');
    const rKurt = readout('excess kurtosis');
    const rC = readout('corr$(y_1,y_2)$');
    const rC2 = readout('corr$(y_1^2,y_2^2)$');
    const rC2x = readout('exact $-\\frac{1}{d-1}$');
    const update = () => {
      const d = D();
      const S = sampleD(d);
      rVar.set(S.dVar.toFixed(3));
      rKurt.set(`${S.kurt.toFixed(3)} <span class="muted" style="font-size:.8em">(${(-6 / (d + 2)).toFixed(3)})</span>`);
      rC.set(S.corr.toFixed(3));
      rC2.set(S.corrSq.toFixed(3), 'accent');
      rC2x.set((-1 / (d - 1)).toFixed(3), 'teal');
      quick.set(d, false);
      cvPdf.request();
      cvSc.request();
    };
    const sD = slider({
      label: '$d$',
      min: 1,
      max: 10,
      step: 0.01,
      value: state.ld,
      format: (v) => `${Math.max(2, Math.round(2 ** v))}`,
      onInput: (v) => {
        state.ld = v;
        setSweep(false);
        update();
      },
    });
    const quick = segmented({
      options: [2, 3, 4, 16, 128, 1024],
      value: 16,
      onChange: (d) => {
        state.ld = Math.log2(d);
        sD.set(state.ld, false);
        setSweep(false);
        update();
      },
    });
    const tRescale = toggle({
      label: 'rescale by $\\sqrt d$',
      onChange: (on) => {
        state.rescale = on;
        update();
      },
    });
    const bSweep = button('', () => setSweep(!state.sweeping));
    function setSweep(on) {
      state.sweeping = on;
      if (on && state.ld >= 9.99) state.ld = 1;
      bSweep.innerHTML = texify(on ? '❚❚ stop' : '▶ sweep $d$');
    }
    setSweep(false);

    body.style.gap = '14px';
    body.append(
      responsiveGrid(el('div', { style: { gap: '16px' } }, vPdf, vSc), 'minmax(0,1.55fr) minmax(0,1fr)', { vizzes: [vPdf, vSc], fill: true }),
      responsiveGrid(
        el(
          'div',
          { style: { flex: '0 0 auto', gap: '14px', alignItems: 'stretch' } },
          el(
            'div.panel',
            { style: { gap: '8px', padding: '12px 18px' } },
            sD.el,
            quick.el,
            el('div.row', {}, tRescale.el, bSweep),
            legend([
              { label: 'exact $f_d$', color: 'teal' },
              { label: 'sampled', color: 'accent' },
              { label: '$\\mathcal N(0,1/d)$', color: 'ink-2', dash: true },
            ]),
          ),
          el(
            'div.readouts',
            { style: { gridTemplateColumns: 'repeat(3, minmax(0,1fr))', gap: '8px', alignContent: 'start' } },
            rVar.el,
            rKurt.el,
            rC.el,
            rC2.el,
            rC2x.el,
          ),
          el(
            'div.card.hot',
            { 'data-reveal': '1', style: { justifyContent: 'center', gap: '6px' } },
            rich('div', 'Every coordinate: the same known density, depending only on $d$', {
              style: { fontFamily: 'var(--serif)', fontSize: '1.15em', lineHeight: '1.25' },
            }),
            rich('div', '⇒ one fixed codebook, no per-block scale', { style: { color: 'var(--teal)', fontWeight: '600' } }),
          ),
        ),
        'minmax(0,1.15fr) minmax(0,1.25fr) minmax(0,1fr)',
      ),
    );

    ctx.onEnter(() => {
      if (!entered) {
        entered = true;
        update();
      }
    });
    ctx.loop((dt) => {
      if (!state.sweeping) return;
      state.ld = Math.min(10, state.ld + dt * 1.1);
      sD.set(state.ld, false);
      update();
      if (state.ld >= 10) setSweep(false);
    });
  },
};

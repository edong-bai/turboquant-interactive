import { el, rich, eq, slider, readout, viz, button, legend, texify } from '../lib/ui.js';
import { createCanvas, line, circle, label, text, draggable, alpha } from '../lib/canvas.js';
import { Orbit3D } from '../lib/orbit3d.js';
import * as M from '../lib/math.js';
import { responsiveGrid } from './_shared-layout.js';

const NP = 30000;
const BINS = 40;

export default {
  id: 'sphere-d4',
  chapter: 'Rotation',
  kicker: 'Where do coordinates land?',
  title: '$d=4$: a flipbook of spheres',
  notes: `We can't draw S³ (the unit sphere in R⁴), but we can flip through it like a flipbook, one height at a time.

${NP.toLocaleString('en-US')} uniform random points on S³ (live sample). Choose a height t for the 4th coordinate. Take the thin slab of points with |x₄ − t| < δ and plot them by their first three coordinates (x₁, x₂, x₃). They form an ordinary 2-sphere of radius √(1 − t²) (yellow), sitting inside the dashed unit sphere. At t = 0 the slice is the full unit sphere; as t → ±1 it shrinks to a point. It plays automatically; dragging the slider or the histogram pauses it.

Same two effects as in d = 3:
• slice: the slice is now a 2-sphere, so its area is 4πr², proportional to r² (one more power of r than the circle in d = 3);
• tilt: still 1/r, exactly the same Pythagorean correction as before (it only involves the one coordinate we're slicing along and the radial direction).
Product: 4πr² · (1/r) / |S³| = 4πr / (2π²) = (2/π)·r = (2/π)√(1 − x²). That's the semicircle law (teal) in the histogram; the dashed line is the flat d = 3 density for comparison. The cancellation is gone: the extra r tips the balance toward the equator.

Count readout: the number of points in the slab is about N·2δ·f(t). Watch it drop as the slice sphere shrinks toward the poles.

Every extra dimension adds one more factor of r to the slice. That is the (1 − x²)^{(d−3)/2} of Lemma 1; at large d this is a Gaussian, next slide.`,
  mount(body, ctx) {
    const state = { t: 0, delta: 0.03, playing: true, phase: 0 };
    let P = null; // Float32Array(NP·4), sampled lazily
    let hist = null;
    const ensure = () => {
      if (P) return;
      const rng = new M.RNG(404);
      P = new Float32Array(NP * 4);
      const x4 = new Float64Array(NP);
      for (let i = 0; i < NP; i++) {
        const a = rng.normal();
        const b = rng.normal();
        const c = rng.normal();
        const d = rng.normal();
        const n = Math.hypot(a, b, c, d);
        P[4 * i] = a / n;
        P[4 * i + 1] = b / n;
        P[4 * i + 2] = c / n;
        P[4 * i + 3] = d / n;
        x4[i] = d / n;
      }
      hist = M.histogram(x4, -1, 1, BINS);
    };
    const slab = () => {
      const out = [];
      if (!P) return out;
      const { t, delta } = state;
      for (let i = 0; i < NP; i++) if (Math.abs(P[4 * i + 3] - t) < delta) out.push(i);
      return out;
    };
    const semicircle = (x) => (2 / Math.PI) * Math.sqrt(Math.max(0, 1 - x * x));
    let current = [];

    // ---------- 3-D slice ----------
    const v3 = viz('', 'drag to orbit');
    let cam;
    const cv3 = createCanvas(v3, (g, w, h) => {
      if (w < 60 || h < 60 || !cam) return;
      const k = Math.min(w, h) * 0.38;
      cam.viewport(w / 2, h / 2 + 4, k);
      const rho = Math.sqrt(Math.max(0, 1 - state.t ** 2));
      // dashed unit sphere (orthographic: silhouette is a circle of radius k)
      circle(g, w / 2, h / 2 + 4, k, { stroke: 'ink-3', width: 1.2, dash: [6, 6] });
      cam.sphere(g, { rings: 6, a: 0.05 });
      // the slice sphere
      cam.sphere(g, { rings: 6, a: 0.16, c: 'yellow', radius: rho });
      circle(g, w / 2, h / 2 + 4, k * rho, { stroke: 'yellow', width: 1.6 });
      // axes
      for (const [ax, name] of [
        [[1.2, 0, 0], 'x₁'],
        [[0, 1.2, 0], 'x₂'],
        [[0, 0, 1.2], 'x₃'],
      ]) {
        cam.path(g, [[0, 0, 0], ax], { color: 'ink-3', alpha: 0.45 });
        const q = cam.project(ax[0] * 1.08, ax[1] * 1.08, ax[2] * 1.08);
        text(g, name, q.x, q.y, { color: 'ink-3', size: 13, align: 'center' });
      }
      const proj = current.map((i) => cam.project(P[4 * i], P[4 * i + 1], P[4 * i + 2]));
      proj.sort((a, b) => a.z - b.z);
      for (const s of proj) {
        const depth = 0.3 + 0.7 * ((s.z / Math.max(rho, 0.05) + 1) / 2);
        g.fillStyle = alpha('accent', 0.25 + 0.7 * M.clamp(depth, 0, 1));
        g.beginPath();
        g.arc(s.x, s.y, 2.3, 0, Math.PI * 2);
        g.fill();
      }
      label(g, `radius √(1 − x₄²) = ${rho.toFixed(3)}`, w / 2, h - 18, { color: 'yellow', size: 14, align: 'center', font: 'mono' });
      text(g, `points with |x₄ − ${state.t.toFixed(2)}| < ${state.delta.toFixed(3)}, plotted by (x₁, x₂, x₃)`, w / 2, 18, {
        color: 'ink-2',
        size: 12.5,
        align: 'center',
      });
    });
    cam = new Orbit3D(cv3.canvas, { yaw: 0.6, pitch: 0.3, spin: 0.12, perspective: 0 });

    // ---------- histogram of x4 (sideways) ----------
    const vH = viz();
    let geom = null;
    const cvH = createCanvas(vH, (g, w, h) => {
      if (w < 60 || h < 60) return;
      const top = 34;
      const bot = h - 40;
      const left = 34;
      const right = w - 12;
      const dmax = 0.8;
      const Y = (z) => bot - ((z + 1) / 2) * (bot - top);
      const X = (f) => left + (f / dmax) * (right - left);
      geom = { top, bot };
      text(g, 'histogram of x₄', 12, 16, { color: 'ink-2', size: 12.5 });
      line(g, left, top, left, bot, { color: 'line-2' });
      line(g, left, bot, right, bot, { color: 'line-2' });
      for (const z of [-1, -0.5, 0, 0.5, 1]) text(g, String(z), left - 6, Y(z), { color: 'ink-3', size: 11, align: 'right', font: 'mono' });
      for (const f of [0, 0.5]) text(g, String(f), X(f), bot + 12, { color: 'ink-3', size: 11, align: 'center', font: 'mono' });
      text(g, 'density', (left + right) / 2, bot + 28, { color: 'ink-3', size: 11, align: 'center' });
      if (hist) {
        const bh = (bot - top) / BINS;
        g.fillStyle = alpha('accent', 0.38);
        for (let i = 0; i < BINS; i++) g.fillRect(left, bot - (i + 1) * bh + 0.5, X(hist[i]) - left, bh - 1);
      }
      // flat d = 3 reference and the d = 4 semicircle
      line(g, X(0.5), top, X(0.5), bot, { color: 'ink-2', width: 1.5, dash: [6, 5] });
      g.save();
      g.strokeStyle = alpha('teal', 1);
      g.lineWidth = 2.5;
      g.beginPath();
      for (let i = 0; i <= 200; i++) {
        const z = -1 + (2 * i) / 200;
        const xx = X(semicircle(z));
        if (i === 0) g.moveTo(xx, Y(z));
        else g.lineTo(xx, Y(z));
      }
      g.stroke();
      g.restore();
      // band
      const z1 = Math.max(-1, state.t - state.delta);
      const z2 = Math.min(1, state.t + state.delta);
      const dens = current.length / NP / (z2 - z1);
      g.fillStyle = alpha('yellow', 0.9);
      g.fillRect(left, Y(z2), X(dens) - left, Math.max(2, Y(z1) - Y(z2)));
      line(g, left, Y(state.t), right, Y(state.t), { color: 'yellow', alpha: 0.35 });
    });

    // ---------- controls ----------
    const rCount = readout('points in slab');
    const rR = readout('$r=\\sqrt{1-t^2}$');
    const rSlice = readout('slice $4\\pi r^2$');
    const rTilt = readout('tilt $1/r$');
    const rF = readout('$\\frac{2}{\\pi}r$');
    const rMeas = readout('measured');
    const sT = slider({
      label: 'height $t$ of $x_4$',
      min: -0.98,
      max: 0.98,
      step: 0.01,
      value: state.t,
      format: (v) => v.toFixed(2),
      onInput: (v) => {
        state.t = v;
        setPlaying(false);
        refresh();
      },
    });
    const sD = slider({
      label: 'slab $\\delta$',
      min: 0.005,
      max: 0.1,
      step: 0.005,
      value: state.delta,
      format: (v) => v.toFixed(3),
      onInput: (v) => {
        state.delta = v;
        refresh();
      },
    });
    const bPlay = button('', () => setPlaying(!state.playing));
    function setPlaying(on) {
      state.playing = on;
      if (on) state.phase = Math.asin(M.clamp(state.t / 0.95, -1, 1));
      bPlay.innerHTML = texify(on ? '❚❚ pause' : '▶ play');
    }
    setPlaying(true);

    function refresh() {
      current = slab();
      const r = Math.sqrt(Math.max(0, 1 - state.t ** 2));
      const z1 = Math.max(-1, state.t - state.delta);
      const z2 = Math.min(1, state.t + state.delta);
      rCount.set(P ? current.length.toLocaleString('en-US') : '…', 'accent');
      rR.set(r.toFixed(3), 'yellow');
      rSlice.set((4 * Math.PI * r * r).toFixed(3));
      rTilt.set((1 / r).toFixed(3), 'blue');
      rF.set(semicircle(state.t).toFixed(3), 'teal');
      rMeas.set(P ? (current.length / NP / (z2 - z1)).toFixed(3) : '…', 'yellow');
      cv3.request();
      cvH.request();
    }

    draggable(cvH.canvas, {
      pick: () => true,
      move: (_, px, py) => {
        if (!geom) return;
        const z = -1 + (2 * (geom.bot - py)) / (geom.bot - geom.top);
        state.t = M.clamp(Math.round(z * 100) / 100, -0.98, 0.98);
        sT.set(state.t, false);
        setPlaying(false);
        refresh();
      },
    });

    const panel = el(
      'div.col',
      { style: { gap: '12px', minHeight: '0' } },
      el('div.panel', { style: { gap: '6px', padding: '12px 18px' } }, sT.el, sD.el, el('div.row', {}, bPlay)),
      el('div.readouts', { style: { gridTemplateColumns: 'repeat(2, minmax(0,1fr))', gap: '8px' } }, rCount.el, rR.el, rSlice.el, rTilt.el, rF.el, rMeas.el),
      el(
        'div.card',
        { 'data-reveal': '1', style: { gap: '4px' } },
        eq('f_4(x)=\\frac{4\\pi r^2\\cdot\\frac1r}{|\\S^3|}=\\frac{2}{\\pi}\\sqrt{1-x^2}'),
        rich('div.note', 'slice $4\\pi r^2$ · tilt $1/r$ · total area $|\\S^3|=2\\pi^2$', { style: { textAlign: 'center' } }),
      ),
      legend([
        { label: '$d=4$: $\\frac2\\pi\\sqrt{1-x^2}$', color: 'teal' },
        { label: '$d=3$: flat $\\frac12$', color: 'ink-2', dash: true },
        { label: 'slab', color: 'yellow' },
      ]),
    );
    body.append(
      responsiveGrid(el('div', { style: { gap: '16px' } }, v3, vH, panel), 'minmax(0,1.35fr) minmax(0,0.5fr) minmax(260px,0.85fr)', {
        vizzes: [v3, vH],
        fill: true,
      }),
    );

    ctx.onEnter(() => {
      ensure();
      refresh();
    });
    ctx.loop((dt) => {
      cam.tick(dt);
      if (state.playing && P) {
        state.phase += dt * 0.42;
        state.t = Math.round(0.95 * Math.sin(state.phase) * 1000) / 1000;
        sT.set(state.t, false);
        refresh();
      }
      cv3.redraw();
    });
  },
};

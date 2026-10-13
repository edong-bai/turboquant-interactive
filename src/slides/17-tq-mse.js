import { el, eq, slider, toggle, button, readout, viz, legend } from '../lib/ui.js';
import { createCanvas, View2D, arrow, circle, dot, line, label, draggable } from '../lib/canvas.js';
import { RNG, sphereCodebook, nearestIndex } from '../lib/math.js';
import { sim as getSim } from './_shared-tq.js';
import { drawDmsePlot } from './_shared-tq-draw.js';

export default {
  id: 'tq-mse',
  chapter: 'TurboQuant',
  kicker: 'Algorithm 1',
  title: 'TurboQuant$_{\\text{mse}}$',
  notes: `Put the last three slides together and you have the whole algorithm. Rotate, then quantize each coordinate independently with the Lloyd–Max codebook for the known coordinate density, then rotate back. No data-dependent tuning, no per-block scale or zero point: the codebook depends only on (d, b) and is computed once.

LEFT, the 2-D picture. The faint tilted axes are the rows of Π, i.e. the rotated frame. Dropping perpendiculars from x onto those axes gives the coordinates y = Πx (dashed orange). Each coordinate snaps to its nearest centroid (teal ticks on the axes; the codebook here is the d = 2 Lloyd–Max codebook, b bits per axis), which picks a point of the rotated product grid: ỹ. Reading that grid point back in the original frame is x̃ = Πᵀỹ (teal). The violet arrow is the error x − x̃.
- Drag x around the circle: the grid point jumps cell to cell.
- "re-roll Π": a different random frame; the same x lands in a different cell. The guarantee is in expectation over Π, for every fixed x (worst case over x).
- b slider: 1 → 3 bits per coordinate; the grid gets finer, the error shrinks about 4× per bit.
- "spin Π": sweeps the frame continuously, showing that what x̃ is depends only on where x falls relative to the random frame.
In 2-D this is a crude picture (two coordinates are far from independent). The point of the earlier slides is that in high d the coordinates of Πx are nearly i.i.d. ~ N(0, 1/d), so a product of scalar codebooks is close to optimal.

RIGHT, the measurement: D_mse vs b on a log scale. Teal dots are a live simulation at d = 128 (640 random unit vectors, one fixed Π, Lloyd–Max codebooks for the exact Beta-like coordinate density). Yellow squares are the paper's measured values (Fig. 3b, d = 1536: 0.36, 0.117, 0.035, 0.0095, 0.0027). The simulation gives ≈ 0.360, 0.116, 0.034, 0.0094, 0.0024 (the exact codebook cost d·C(f_X, b) at d = 128 is 0.361, 0.116, 0.034, 0.0093, 0.0025): the same numbers as the paper's d = 1536, because D_mse depends on d only through tiny finite-d corrections (the coordinate density is ≈ N(0, 1/d) and the cost scales out).
The shaded band is between the lower bound 4^{-b} (any quantizer, Theorem 3, next slides) and the Theorem 1 upper bound (√3π/2)·4^{-b} ≈ 2.72·4^{-b}. Every bit buys a factor of 4: the right exponential rate.

Theorem 1 text quotes 0.36, 0.117, 0.03, 0.009 for b = 1..4. For b ≤ 4 these are the numerically solved Lloyd–Max costs; the closed-form bound for large b comes from Panter–Dite high-resolution analysis: C ≈ (1/12)(∫f^{1/3})³ 4^{-b} = (√3π/2)·4^{-b}/d for a Gaussian.

Note the proof of the upper bound only needs that each coordinate of Πx has the known marginal and linearity of expectation; independence is not needed for MSE.`,
  mount(body, ctx) {
    const state = { b: 2, phi: 0.5, xa: 0.75, spin: false };
    const rng = new RNG(1717);

    const rY = readout('$y=\\Pi x$');
    const rIdx = readout('idx (bits)');
    const rErr = readout('$\\|x-\\tilde x\\|^2$');

    // ---------- 2-D geometric view ----------
    const v2 = viz('', 'd = 2 picture · drag $x$');
    let view;
    const cv = createCanvas(v2, (g, w, h) => {
      if (w < 40 || h < 40) return;
      view = new View2D({ w, h, xmin: -1.3, xmax: 1.3, ymin: -1.3, ymax: 1.3, pad: 16 });
      const { centroids, boundaries } = sphereCodebook(2, state.b);
      const c = Math.cos(state.phi);
      const s = Math.sin(state.phi);
      // Rows of Π: the rotated axes.
      const e1 = [c, -s];
      const e2 = [s, c];
      const P = (u, v) => [u * e1[0] + v * e2[0], u * e1[1] + v * e2[1]]; // Πᵀ(u, v)
      const L = (p, q, o) => line(g, view.sx(p[0]), view.sy(p[1]), view.sx(q[0]), view.sy(q[1]), o);

      circle(g, view.sx(0), view.sy(0), view.len(1), { stroke: 'line-2', width: 1.5 });
      // cell boundaries of the product grid (in the rotated frame)
      for (const m of boundaries) {
        L(P(m, -1.25), P(m, 1.25), { color: 'ink-3', alpha: 0.22, dash: [3, 5] });
        L(P(-1.25, m), P(1.25, m), { color: 'ink-3', alpha: 0.22, dash: [3, 5] });
      }
      // rotated axes
      L(P(-1.28, 0), P(1.28, 0), { color: 'ink-2', alpha: 0.45, width: 1.2 });
      L(P(0, -1.28), P(0, 1.28), { color: 'ink-2', alpha: 0.45, width: 1.2 });
      const t1 = P(1.22, 0);
      const t2 = P(0, 1.22);
      label(g, 'y₁', view.sx(t1[0]), view.sy(t1[1]), { color: 'ink-3', size: 14, align: 'center' });
      label(g, 'y₂', view.sx(t2[0]), view.sy(t2[1]), { color: 'ink-3', size: 14, align: 'center' });
      // centroid ticks on both axes and the grid points
      for (const a of centroids) {
        for (const bb of centroids) {
          const p = P(a, bb);
          dot(g, view.sx(p[0]), view.sy(p[1]), 3, 'ink-3');
        }
        for (const [p, n] of [
          [P(a, 0), e2],
          [P(0, a), e1],
        ]) {
          const k = 0.035;
          L([p[0] - k * n[0], p[1] - k * n[1]], [p[0] + k * n[0], p[1] + k * n[1]], { color: 'teal', width: 2 });
        }
      }

      const x = [Math.cos(state.xa), Math.sin(state.xa)];
      const y1 = x[0] * e1[0] + x[1] * e1[1];
      const y2 = x[0] * e2[0] + x[1] * e2[1];
      const i1 = nearestIndex(y1, boundaries);
      const i2 = nearestIndex(y2, boundaries);
      const q1 = centroids[i1];
      const q2 = centroids[i2];
      const xt = P(q1, q2);

      // projections of x on the rotated axes (y_j) and their snapped centroids (ỹ_j)
      const f1 = P(y1, 0);
      const f2 = P(0, y2);
      L(x, f1, { color: 'accent', dash: [4, 4], alpha: 0.75 });
      L(x, f2, { color: 'accent', dash: [4, 4], alpha: 0.75 });
      dot(g, view.sx(f1[0]), view.sy(f1[1]), 4, 'accent');
      dot(g, view.sx(f2[0]), view.sy(f2[1]), 4, 'accent');
      const s1 = P(q1, 0);
      const s2 = P(0, q2);
      L(f1, s1, { color: 'teal', width: 4, alpha: 0.7 });
      L(f2, s2, { color: 'teal', width: 4, alpha: 0.7 });
      circle(g, view.sx(s1[0]), view.sy(s1[1]), 6, { stroke: 'teal', width: 2 });
      circle(g, view.sx(s2[0]), view.sy(s2[1]), 6, { stroke: 'teal', width: 2 });
      L(s1, xt, { color: 'teal', dash: [2, 4], alpha: 0.6 });
      L(s2, xt, { color: 'teal', dash: [2, 4], alpha: 0.6 });

      arrow(g, view.sx(0), view.sy(0), view.sx(x[0]), view.sy(x[1]), { color: 'accent', width: 3.5 });
      arrow(g, view.sx(0), view.sy(0), view.sx(xt[0]), view.sy(xt[1]), { color: 'teal', width: 2.8 });
      arrow(g, view.sx(xt[0]), view.sy(xt[1]), view.sx(x[0]), view.sy(x[1]), { color: 'violet', width: 2, head: 8 });
      label(g, 'x', view.sx(x[0] * 1.1), view.sy(x[1] * 1.1), { color: 'accent', size: 19, weight: 600, align: 'center' });
      label(g, 'x̃ = Πᵀỹ', view.sx(xt[0]) + 10, view.sy(xt[1]) + 16, { color: 'teal', size: 15, weight: 600 });

      const err = (x[0] - xt[0]) ** 2 + (x[1] - xt[1]) ** 2;
      rY.set(`(${y1.toFixed(2)}, ${y2.toFixed(2)})`, 'accent');
      const bits = (i) => i.toString(2).padStart(state.b, '0');
      rIdx.set(`${bits(i1)} · ${bits(i2)}`, 'teal');
      rErr.set(err.toFixed(4), 'violet');
    });

    draggable(cv.canvas, {
      pick(px, py) {
        if (!view) return null;
        const x = [Math.cos(state.xa), Math.sin(state.xa)];
        return Math.hypot(view.sx(x[0]) - px, view.sy(x[1]) - py) < 24 ? 'x' : null;
      },
      move(_, px, py) {
        state.xa = Math.atan2(view.wy(py), view.wx(px));
        cv.request();
      },
    });

    const sB = slider({
      label: '$b$',
      min: 1,
      max: 3,
      step: 1,
      value: state.b,
      format: (v) => `${v} bit${v > 1 ? 's' : ''}`,
      onInput: (val) => {
        state.b = val;
        cv.request();
        plot.request();
      },
    });
    const reroll = button('re-roll $\\Pi$', () => {
      state.phi = rng.uniform(0, 2 * Math.PI);
      cv.request();
    });
    const tSpin = toggle({ label: 'spin $\\Pi$', onChange: (on) => (state.spin = on) });

    // ---------- D_mse vs b plot ----------
    const sim = getSim();
    const vp = viz('', '');
    const plot = createCanvas(vp, (g, w, h) => drawDmsePlot(g, w, h, { sim, highlight: state.b }));

    const algo = el('pre.algo', {
      html: `<span class="cm">// once per (d, b)</span>
Π ← random rotation  <span class="cm">(QR of Gaussian)</span>
c₁…c₂ᵇ ← Lloyd–Max for f<sub>X</sub>
<span class="hl">Quant</span>(x):
  y ← Π·x
  idx<sub>j</sub> ← argmin<sub>k</sub> |y<sub>j</sub> − c<sub>k</sub>|   <span class="cm">b bits each</span>
<span class="hl">DeQuant</span>(idx):
  ỹ<sub>j</sub> ← c<sub>idx<sub>j</sub></sub>
  x̃ ← Πᵀ·ỹ`,
    });

    const grow = (node, min) => Object.assign(node.style, { flex: '1 1 0', height: 'auto', minHeight: min });
    grow(v2, '300px');
    grow(vp, '200px');
    const left = el(
      'div.col',
      {},
      v2,
      el('div.row', {}, el('div', { style: { flex: '1 1 200px' } }, sB.el), reroll, tSpin.el),
      el('div.readouts', {}, rY.el, rIdx.el, rErr.el),
    );
    const right = el(
      'div.col',
      {},
      el('div', { 'data-reveal': '1' }, algo),
      vp,
      legend([
        { label: 'live sim, $d=128$', color: 'teal' },
        { label: 'paper, Fig. 3b', color: 'yellow' },
        { label: 'Thm 1 bound', color: 'red', dash: true },
        { label: '$4^{-b}$', color: 'blue', dash: true },
      ]),
      el('div', { 'data-reveal': '2' }, eq('\\textbf{Thm 1:}\\quad \\Dmse \\le \\tfrac{\\sqrt3\\,\\pi}{2}\\,4^{-b}\\approx 2.72\\cdot 4^{-b}')),
    );
    body.append(el('div.stage', {}, left, right));

    ctx.onEnter(() => {
      cv.request();
      plot.request();
    });
    let seen = -1;
    ctx.loop((dt) => {
      if (!sim.done) sim.step(5);
      if (sim.count !== seen) {
        seen = sim.count;
        plot.redraw();
      }
      if (state.spin) {
        state.phi += dt * 0.35;
        cv.redraw();
      }
    });
  },
};

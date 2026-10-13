import { el, eq, segmented, readout, viz, legend } from '../lib/ui.js';
import { createCanvas, View2D, arrow, dot, line, label, text, color, FONT } from '../lib/canvas.js';
import { Plot } from '../lib/plot.js';
import { sim as getSim, ipPairs, slopeThroughOrigin, measuredDmse, codebookDmse } from './_shared-tq.js';
import { progressBadge } from './_shared-tq-draw.js';

// Illustrative attention logits ⟨q, k_i⟩ (not from the paper).
const LOGITS = [4.0, 2.7, 2.1, 1.4, 0.9, 0.3, -0.4, -1.1];

const softmax = (z) => {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};

export default {
  id: 'mse-bias',
  chapter: 'TurboQuant',
  kicker: 'The catch',
  title: 'MSE-optimal shrinks inner products',
  notes: `TurboQuant_mse is near-optimal for reconstruction, but it is a biased inner-product estimator: on average it underestimates every inner product by a constant factor.

LEFT: live simulation at d = 128. 640 random unit vectors x, each paired with 6 unit queries y = cosθ·x + sinθ·u (u ⟂ x, θ uniform), so ⟨y,x⟩ spans [−1, 1]. Horizontal: the true ⟨y,x⟩; vertical: ⟨y,x̃_mse⟩. The dashed diagonal is y = x; the red line is the least-squares slope through the origin. Pick b:
- b = 1: slope ≈ 0.64 (2/π = 0.637)
- b = 2: ≈ 0.89
- b = 3: ≈ 0.97
- b = 4: ≈ 0.99
(measured values are shown live; they match 1 − D_mse to the third digit.)

Why exactly 1 − D_mse? Middle-right picture. Each coordinate is reconstructed by its cell's conditional mean (that is the Lloyd–Max centroid condition). Conditional means have orthogonal errors: E[(y_j − c(y_j))·c(y_j)] = 0. Summing over coordinates and rotating back: E⟨x − x̃, x̃⟩ = 0, so E⟨x, x̃⟩ = E‖x̃‖² and then D_mse = 1 − 2E⟨x,x̃⟩ + E‖x̃‖² = 1 − E‖x̃‖². In the picture: x̃ sits on the Thales circle over [0, x] (right angle at x̃), with ‖x̃‖² = 1 − D. Its projection on x, which by rotational symmetry is E[x̃], is (1 − D)·x. Hence E⟨y, x̃⟩ = (1 − D_mse)·⟨y, x⟩ for every y. The shrinkage is not a bug of the codebook; it is what any MSE-optimal (conditional-mean) reconstruction does: when uncertain, shrink toward the mean.

The paper's b = 1 calculation (Section 3.2): the 1-bit codebook is ±√(2/(πd)), so Q(x) = sign(Πx) and x̃ = √(2/(πd))·Πᵀsign(Πx). This is QJL with S replaced by a rotation and no Gaussian projection on the query: E⟨y, x̃⟩ = (2/π)⟨y, x⟩. So 36% of every inner product is lost at 1 bit.

Bottom-right: why it matters for attention. Shrinking every logit by the slope is like raising the softmax temperature by 1/slope: the attention weights flatten. The logits here are illustrative (not from the paper). At b = 1 the top weight drops visibly; at b = 4 the effect is ~1%. Only the expected logits are shown; the quantization noise is on top.

Bias shrinks with b (the paper says so empirically; our 1 − D_mse identity makes it exact). Next slide: remove the bias entirely.`,
  mount(body, ctx) {
    const sim = getSim();
    const state = { b: 1, dShown: codebookDmse(1) };
    const cache = new Map(); // b → {count, ip, est, slope}

    const getPairs = (b) => {
      let c = cache.get(b);
      if (!c || c.count !== sim.count) {
        const { ip, est } = ipPairs('mse', b, sim);
        c = { count: sim.count, ip, est, slope: slopeThroughOrigin(ip, est) };
        cache.set(b, c);
      }
      return c;
    };
    const dmse = (b) => (sim.count > 0 ? measuredDmse(b, sim) : codebookDmse(b));

    const rSlope = readout('measured slope');
    const rOne = readout('$1-\\Dmse$');
    const rRef = readout('$b=1$: $2/\\pi$', (2 / Math.PI).toFixed(3));

    // ---------- scatter ----------
    const vs = viz('', 'live simulation · $d=128$');
    const scatter = createCanvas(vs, (g, w, h) => {
      if (w < 40 || h < 40) return;
      const s = Math.min(w, h);
      const p = new Plot(g, {
        x: (w - s) / 2,
        y: 0,
        w: s,
        h,
        xmin: -1.05,
        xmax: 1.05,
        ymin: -1.05,
        ymax: 1.05,
        xticks: [-1, -0.5, 0, 0.5, 1],
        yticks: [-1, -0.5, 0, 0.5, 1],
        xlabel: '⟨y, x⟩',
        ylabel: '⟨y, x̃_mse⟩',
        margin: { l: 56, r: 16, t: 30, b: 40 },
      });
      p.frame();
      p.line([-1.05, 1.05], [-1.05, 1.05], { color: 'ink-2', width: 1.5, dash: [6, 5] });
      if (sim.count > 0) {
        const c = getPairs(state.b);
        p.points(c.ip, c.est, { color: 'red', r: 1.7, alpha: 0.4 });
        p.line([-1.05, 1.05], [-1.05 * c.slope, 1.05 * c.slope], { color: 'red', width: 2.5 });
        label(g, `slope ${c.slope.toFixed(3)}`, p.tx(0.98), p.ty(0.98 * c.slope) + 22, { color: 'red', size: 15, weight: 600, align: 'right' });
        rSlope.set(c.slope.toFixed(3), 'red');
      }
      label(g, 'y = x', p.tx(0.62), p.ty(0.62) - 14, { color: 'ink-2', size: 13, align: 'right' });
      rOne.set((1 - dmse(state.b)).toFixed(3), 'teal');
      progressBadge(p, sim.progress);
    });

    // ---------- Thales picture: E⟨x, x̃⟩ = E‖x̃‖² = 1 − D ----------
    const vg = viz('', 'why $1-\\Dmse$');
    const geo = createCanvas(vg, (g, w, h) => {
      if (w < 40 || h < 40) return;
      const view = new View2D({ w, h, xmin: -0.08, xmax: 1.14, ymin: -0.2, ymax: 0.66, pad: 12 });
      const D = state.dShown;
      const cphi = Math.sqrt(1 - D);
      const sphi = Math.sqrt(D);
      const xt = [cphi * cphi, cphi * sphi]; // on the Thales circle over [0, x]
      const O = [view.sx(0), view.sy(0)];
      const X = [view.sx(1), view.sy(0)];
      const T = [view.sx(xt[0]), view.sy(xt[1])];
      const F = [view.sx(xt[0]), view.sy(0)];
      // unit-circle arc and Thales circle
      g.save();
      g.strokeStyle = color('line-2');
      g.lineWidth = 1.5;
      g.beginPath();
      g.arc(O[0], O[1], view.len(1), -0.65, 0.42);
      g.stroke();
      g.restore();
      g.save();
      g.strokeStyle = color('ink-3');
      g.setLineDash([3, 5]);
      g.beginPath();
      g.arc(view.sx(0.5), view.sy(0), view.len(0.5), Math.PI, 2 * Math.PI);
      g.stroke();
      g.restore();
      line(g, T[0], T[1], F[0], F[1], { color: 'ink-3', dash: [3, 4] });
      // right angle at x̃
      const u1 = [-xt[0], -xt[1]];
      const u2 = [1 - xt[0], -xt[1]];
      const n1 = Math.hypot(...u1);
      const n2 = Math.hypot(...u2);
      const k = 0.06;
      const a = [xt[0] + (k * u1[0]) / n1, xt[1] + (k * u1[1]) / n1];
      const b = [xt[0] + (k * u2[0]) / n2, xt[1] + (k * u2[1]) / n2];
      const cpt = [a[0] + (k * u2[0]) / n2, a[1] + (k * u2[1]) / n2];
      g.save();
      g.strokeStyle = color('yellow');
      g.lineWidth = 1.5;
      g.beginPath();
      g.moveTo(view.sx(a[0]), view.sy(a[1]));
      g.lineTo(view.sx(cpt[0]), view.sy(cpt[1]));
      g.lineTo(view.sx(b[0]), view.sy(b[1]));
      g.stroke();
      g.restore();

      arrow(g, O[0], O[1], X[0], X[1], { color: 'accent', width: 3.5 });
      arrow(g, O[0], O[1], T[0], T[1], { color: 'teal', width: 3 });
      arrow(g, T[0], T[1], X[0], X[1], { color: 'violet', width: 2, head: 8 });
      dot(g, F[0], F[1], 5, 'ink');
      label(g, 'x', X[0] + 8, X[1] + 2, { color: 'accent', size: 18, weight: 600 });
      label(g, 'x̃', T[0] - 4, T[1] - 14, { color: 'teal', size: 18, weight: 600, align: 'center' });
      label(g, 'E[x̃] = (1−D)·x', F[0], F[1] + 18, { color: 'ink', size: 13, align: 'center' });
      label(g, `‖x̃‖² = ${(1 - D).toFixed(3)}`, view.sx(xt[0] / 2) - 10, view.sy(xt[1] / 2) - 4, { color: 'teal', size: 13, align: 'right' });
      label(g, 'error ⟂ x̃', view.sx((xt[0] + 1) / 2) + 10, view.sy(xt[1] / 2) - 4, { color: 'violet', size: 13 });
    });

    // ---------- softmax flattening ----------
    const vb = viz('', 'attention weights · illustrative logits');
    const bars = createCanvas(vb, (g, w, h) => {
      if (w < 40 || h < 40) return;
      const slope = sim.count > 0 ? getPairs(state.b).slope : 1 - codebookDmse(state.b);
      const p0 = softmax(LOGITS);
      const p1 = softmax(LOGITS.map((z) => z * slope));
      const top = 52;
      const bottom = h - 26;
      const left = 18;
      const right = w - 16;
      const n = LOGITS.length;
      const slot = (right - left) / n;
      const bw = slot * 0.36;
      const scaleY = (bottom - top) / Math.max(...p0);
      for (let i = 0; i < n; i++) {
        const x0 = left + i * slot + slot * 0.12;
        g.fillStyle = color('ink-2');
        g.fillRect(x0, bottom - p0[i] * scaleY, bw, p0[i] * scaleY);
        g.fillStyle = color('red');
        g.fillRect(x0 + bw + 2, bottom - p1[i] * scaleY, bw, p1[i] * scaleY);
      }
      line(g, left, bottom, right, bottom, { color: 'line-2' });
      g.save();
      g.font = `12px ${FONT.mono}`;
      g.fillStyle = color('ink-3');
      g.textAlign = 'center';
      g.textBaseline = 'top';
      for (let i = 0; i < n; i++) g.fillText(`k${i + 1}`, left + i * slot + slot * 0.12 + bw, bottom + 5);
      g.restore();
      text(g, `top weight ${p0[0].toFixed(2)} → ${p1[0].toFixed(2)}`, right, 38, { color: 'red', size: 13, align: 'right', font: 'mono' });
    });

    const seg = segmented({
      label: 'bits $b$',
      options: [1, 2, 3, 4],
      value: state.b,
      onChange: (v) => {
        state.b = v;
        scatter.request();
        bars.request();
      },
    });

    const grow = (node, min) => Object.assign(node.style, { flex: '1 1 0', height: 'auto', minHeight: min });
    grow(vs, '300px');
    grow(vg, '170px');
    grow(vb, '150px');

    const left = el('div.col', {}, vs, el('div.row', {}, seg.el), el('div.readouts', {}, rSlope.el, rOne.el, rRef.el));
    const right = el(
      'div.col',
      {},
      vg,
      el(
        'div.card',
        { 'data-reveal': '1', style: { padding: '8px 14px', gap: '0', fontSize: '0.9em' } },
        eq('\\E\\,\\ip{x}{\\tilde x}=\\E\\norm{\\tilde x}^2=1-\\Dmse'),
        eq('b=1:\\ \\ \\tilde x=\\sqrt{\\tfrac{2}{\\pi d}}\\,\\Pi^{\\top}\\sign(\\Pi x)'),
        eq('\\Rightarrow\\ \\ \\E\\,\\ip{y}{\\tilde x}=\\tfrac{2}{\\pi}\\ip{y}{x}'),
      ),
      vb,
      legend([
        { label: 'exact logits', color: 'ink-2' },
        { label: 'logits × slope', color: 'red' },
      ]),
    );
    body.append(el('div.stage', {}, left, right));

    let seen = -1;
    ctx.onEnter(() => {
      scatter.request();
      geo.request();
      bars.request();
    });
    ctx.loop((dt) => {
      if (!sim.done) sim.step(5);
      if (sim.count !== seen) {
        seen = sim.count;
        scatter.redraw();
        bars.redraw();
      }
      const target = dmse(state.b);
      if (Math.abs(target - state.dShown) > 1e-5) {
        state.dShown += (target - state.dShown) * Math.min(1, dt * 6);
        geo.redraw();
      }
    });
  },
};

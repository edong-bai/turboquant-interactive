import { el, eq, segmented, button, readout, viz, legend, rich } from '../lib/ui.js';
import { createCanvas, View2D, arrow, circle, dot, line, label, draggable, alpha, color } from '../lib/canvas.js';
import { Plot } from '../lib/plot.js';
import { RNG, sphereCodebook, nearestIndex, histogram, bounds } from '../lib/math.js';
import { sim as getSim, ipPairs, ipErrors, slopeThroughOrigin, moments, measuredDmse, D } from './_shared-tq.js';
import { progressBadge } from './_shared-tq-draw.js';

const DRAWS = 360; // draws of the 2-D Gaussian sketch S
const C_HIST = 0.5; // ⟨y, x⟩ used for the error histograms

export default {
  id: 'tq-prod',
  chapter: 'TurboQuant',
  kicker: 'Algorithm 2',
  title: 'Spend the last bit on QJL of the residual',
  notes: `Fixing the bias: spend b − 1 bits on TurboQuant_mse, then the last bit per coordinate on a 1-bit QJL sketch of what is left over, the residual r = x − x̃_mse. Store idx, the d sign bits sign(S·r), and one float γ = ‖r‖. Dequantize as x̃_mse plus the QJL reconstruction √(π/2)/d · γ · Sᵀ·qjl.

Why it is unbiased: QJL is an unbiased inner-product estimator for any vector, so conditionally on the mse stage, E⟨y, x̃⟩ = ⟨y, x̃_mse⟩ + ⟨y, r⟩ = ⟨y, x⟩. The MSE stage's shrinkage is exactly what r contains, and QJL puts it back on average.
Why it is accurate: QJL's error scales with the length of what it encodes. Its variance is ≤ (π/2d)·‖y‖²·‖r‖², and E‖r‖² = D_mse(b − 1), which is already small. That gives D_prod ≤ (π/2d)·‖y‖²·D_mse(b − 1), and with Theorem 1 at b − 1 bits, Theorem 2: D_prod ≤ (√3π²/d)·‖y‖²·4^{-b}. The paper's numbers: d·D_prod ≈ 1.57, 0.56, 0.18, 0.047 for b = 1..4.
b = 1: the mse stage gets 0 bits, x̃_mse = 0, r = x: TurboQuant_prod is exactly QJL with m = d.

MIDDLE (conceptual, d = 2): the grid is the (b − 1)-bit rotated product grid. Teal x̃_mse is the snapped point; violet r is the short residual. Each faint dot is x̃_mse + QJL(r) for a fresh Gaussian S (m rows). The cloud is noisy, but its mean (white ring) lands on x: unbiased. The spread scales with ‖r‖: drag x to a point near a grid point and the cloud collapses; pick b = 1 and the cloud is as big as x itself (pure QJL). m = more sign bits ⇒ tighter cloud (variance ∝ 1/m). In TurboQuant m = d.
- Drag x. Change b (1–4: grid has b − 1 bits per axis). Re-draw S.

RIGHT, live simulation at d = 128 (same 640 vectors and query pairs as the previous slide; one Π, one d×d Gaussian S): the teal scatter is ⟨y, x̃_prod⟩ vs ⟨y, x⟩, with slope ≈ 1.00 at every b (measured: ≈ 0.997, 0.997, 1.000, 0.999 for b = 1..4), compared with the red TurboQuant_mse cloud at the same total b (slope 0.64, 0.89, 0.97, 0.99). Below: histograms of the inner-product error ⟨y,x⟩ − ⟨y,x̃⟩ for queries with ⟨y,x⟩ = 0.5, like the paper's Fig. 1: mse is shifted right by ≈ 0.5·D_mse(b), prod is centered on 0 but wider. That trade (no bias, more variance) is the subject of two slides from now.`,
  mount(body, ctx) {
    const sim = getSim();
    const state = { b: 2, m: 2, xa: 2.17, phi: 0.6, cur: 0, tAcc: 0 };
    const rng = new RNG(1919);
    let sketches = [];
    const redrawS = () => {
      sketches = Array.from({ length: DRAWS }, () => Array.from({ length: state.m }, () => [rng.normal(), rng.normal()]));
    };
    redrawS();

    // ---------- 2-D conceptual diagram ----------
    const v2 = viz('', 'd = 2 picture · drag $x$');
    let view;
    const cv = createCanvas(v2, (g, w, h) => {
      if (w < 40 || h < 40) return;
      view = new View2D({ w, h, xmin: -1.3, xmax: 1.3, ymin: -1.3, ymax: 1.3, pad: 14 });
      const bm = state.b - 1;
      const c = Math.cos(state.phi);
      const s = Math.sin(state.phi);
      const e1 = [c, -s];
      const e2 = [s, c];
      const P = (u, v) => [u * e1[0] + v * e2[0], u * e1[1] + v * e2[1]];

      const x = [Math.cos(state.xa), Math.sin(state.xa)];
      let xm = [0, 0];
      const cb = bm > 0 ? sphereCodebook(2, bm) : null;
      if (cb) {
        const y1 = x[0] * e1[0] + x[1] * e1[1];
        const y2 = x[0] * e2[0] + x[1] * e2[1];
        xm = P(cb.centroids[nearestIndex(y1, cb.boundaries)], cb.centroids[nearestIndex(y2, cb.boundaries)]);
      }
      const r = [x[0] - xm[0], x[1] - xm[1]];
      const rn = Math.hypot(r[0], r[1]);

      // QJL draws of the residual: √(π/2)/m · ‖r‖ · Sᵀ sign(S r)
      const k = Math.sqrt(Math.PI / 2) / state.m;
      let mx = 0;
      let my = 0;
      const pts = sketches.map((Sm) => {
        let ex = 0;
        let ey = 0;
        for (const row of Sm) {
          const sg = row[0] * r[0] + row[1] * r[1] >= 0 ? 1 : -1;
          ex += row[0] * sg;
          ey += row[1] * sg;
        }
        const p = [xm[0] + k * rn * ex, xm[1] + k * rn * ey];
        mx += p[0];
        my += p[1];
        return p;
      });
      mx /= pts.length;
      my /= pts.length;
      const pc = pts[state.cur % pts.length];

      // Draw the scene with a given view; `big` = labeled close-up.
      const scene = (V, big) => {
        const L = (p, q, o) => line(g, V.sx(p[0]), V.sy(p[1]), V.sx(q[0]), V.sy(q[1]), o);
        circle(g, V.sx(0), V.sy(0), V.len(1), { stroke: 'line-2', width: 1.5 });
        if (cb) {
          for (const m of cb.boundaries) {
            L(P(m, -1.25), P(m, 1.25), { color: 'ink-3', alpha: 0.22, dash: [3, 5] });
            L(P(-1.25, m), P(1.25, m), { color: 'ink-3', alpha: 0.22, dash: [3, 5] });
          }
          for (const a of cb.centroids)
            for (const bb of cb.centroids) {
              const q = P(a, bb);
              dot(g, V.sx(q[0]), V.sy(q[1]), big ? 4 : 2.6, 'ink-3');
            }
        }
        g.fillStyle = alpha('teal', big ? 0.32 : 0.22);
        for (const q of pts) {
          g.beginPath();
          g.arc(V.sx(q[0]), V.sy(q[1]), big ? 2.8 : 1.8, 0, Math.PI * 2);
          g.fill();
        }
        if (big) {
          arrow(g, V.sx(xm[0]), V.sy(xm[1]), V.sx(pc[0]), V.sy(pc[1]), { color: 'violet', width: 1.8, head: 9, dash: [5, 4], alpha: 0.8 });
          dot(g, V.sx(pc[0]), V.sy(pc[1]), 5, 'teal');
        }
        arrow(g, V.sx(0), V.sy(0), V.sx(x[0]), V.sy(x[1]), { color: 'accent', width: big ? 4 : 3.2 });
        if (bm > 0) arrow(g, V.sx(0), V.sy(0), V.sx(xm[0]), V.sy(xm[1]), { color: 'teal', width: big ? 3 : 2.4 });
        arrow(g, V.sx(xm[0]), V.sy(xm[1]), V.sx(x[0]), V.sy(x[1]), { color: 'violet', width: big ? 3.2 : 2.4, head: big ? 12 : 8 });
        circle(g, V.sx(mx), V.sy(my), big ? 10 : 6, { stroke: 'ink', width: 2.2 });
      };

      // full view (left part)
      scene(view, false);
      label(g, 'x', view.sx(x[0] * 1.12), view.sy(x[1] * 1.12), { color: 'accent', size: 18, weight: 600, align: 'center' });

      // magnified close-up around x̃_mse → x
      const half = Math.min(0.75, Math.max(0.15, 1.5 * rn));
      const cx = (x[0] + xm[0]) / 2;
      const cy = (x[1] + xm[1]) / 2;
      const box = Math.min(w, h) * 0.5;
      const bx = w - box - 12;
      const by = h - box - 12;
      const zv = new View2D({ x: bx, y: by, w: box, h: box, xmin: cx - half, xmax: cx + half, ymin: cy - half, ymax: cy + half });
      // zoom frame on the main view
      g.save();
      g.strokeStyle = alpha('ink-2', 0.5);
      g.setLineDash([4, 4]);
      g.strokeRect(view.sx(cx - half), view.sy(cy + half), view.len(2 * half), view.len(2 * half));
      g.restore();
      g.save();
      g.fillStyle = 'rgba(8,14,28,0.94)';
      g.fillRect(bx, by, box, box);
      g.strokeStyle = color('line-2');
      g.strokeRect(bx, by, box, box);
      g.beginPath();
      g.rect(bx, by, box, box);
      g.clip();
      scene(zv, true);
      label(g, 'x', zv.sx(x[0]) - 12, zv.sy(x[1]) - 14, { color: 'accent', size: 18, weight: 600, align: 'right' });
      if (bm > 0) label(g, 'x̃_mse', zv.sx(xm[0]) + 10, zv.sy(xm[1]) + 16, { color: 'teal', size: 14, weight: 600 });
      label(g, 'r', zv.sx(cx) - 12, zv.sy(cy) - 10, { color: 'violet', size: 17, weight: 600, align: 'right' });
      label(g, 'mean of draws', zv.sx(mx), zv.sy(my) + 24, { color: 'ink', size: 12, align: 'center' });
      g.restore();
      label(g, `zoom ×${(1.3 / half).toFixed(1)}`, bx + 8, by + 14, { color: 'ink-3', size: 11 });
      label(g, bm === 0 ? 'b = 1: x̃_mse = 0, r = x (pure QJL)' : `‖r‖ = ${rn.toFixed(3)}`, 14, h - 16, { color: 'violet', size: 13 });
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

    // ---------- live simulation: scatter + histograms ----------
    const rSlopeP = readout('slope · prod');
    const rSlopeM = readout('slope · mse');
    const rMean = readout('mean err · prod');
    const rDp = readout('$d\\cdot\\Dprod$ live, $y\\perp x$');
    const rProof = readout('$\\tfrac{\\pi}{2}\\Dmse(b{-}1)$ live');
    const rThm = readout('Thm 2: $\\sqrt3\\pi^2 4^{-b}$');
    const updateA = () => {
      const b = state.b;
      rThm.set((D * bounds.prodUpper(b, D)).toFixed(3), 'ink-2');
      if (sim.count === 0) return;
      rDp.set((D * moments(ipErrors('prod', b, 0, sim)).ms).toFixed(3), 'teal');
      rProof.set(((Math.PI / 2) * (b > 1 ? measuredDmse(b - 1, sim) : 1)).toFixed(3), 'violet');
    };

    const vs = viz('', 'live simulation · $d=128$');
    const scatter = createCanvas(vs, (g, w, h) => {
      if (w < 40 || h < 40) return;
      const p = new Plot(g, {
        x: 0,
        y: 0,
        w,
        h,
        xmin: -1.05,
        xmax: 1.05,
        ymin: -1.15,
        ymax: 1.15,
        xticks: [-1, 0, 1],
        yticks: [-1, 0, 1],
        xlabel: '⟨y, x⟩',
        ylabel: '⟨y, x̃⟩',
        margin: { l: 46, r: 12, t: 26, b: 36 },
      });
      p.frame();
      p.line([-1.1, 1.1], [-1.1, 1.1], { color: 'ink-2', width: 1.2, dash: [6, 5] });
      if (sim.count > 0) {
        const pm = ipPairs('mse', state.b, sim);
        const pp = ipPairs('prod', state.b, sim);
        const sm = slopeThroughOrigin(pm.ip, pm.est);
        const sp = slopeThroughOrigin(pp.ip, pp.est);
        p.points(pm.ip, pm.est, { color: 'red', r: 1.3, alpha: 0.18 });
        p.points(pp.ip, pp.est, { color: 'teal', r: 1.5, alpha: 0.35 });
        p.line([-1.05, 1.05], [-1.05 * sm, 1.05 * sm], { color: 'red', width: 2 });
        p.line([-1.05, 1.05], [-1.05 * sp, 1.05 * sp], { color: 'teal', width: 2.5 });
        rSlopeP.set(sp.toFixed(3), 'teal');
        rSlopeM.set(sm.toFixed(3), 'red');
      }
      progressBadge(p, sim.progress);
    });

    const vh = viz('', `error $\\langle y,x\\rangle-\\langle y,\\tilde x\\rangle$ at $\\langle y,x\\rangle=${C_HIST}$`);
    const hist = createCanvas(vh, (g, w, h) => {
      if (w < 40 || h < 40 || sim.count === 0) return;
      const em = ipErrors('mse', state.b, C_HIST, sim);
      const ep = ipErrors('prod', state.b, C_HIST, sim);
      const mm = moments(em);
      const mp = moments(ep);
      const R = 1.08 * Math.max(Math.abs(mm.mean) + 3.2 * Math.sqrt(mm.var), 3.2 * Math.sqrt(mp.var));
      const bins = 41;
      const hm = histogram(em, -R, R, bins);
      const hp = histogram(ep, -R, R, bins);
      const top = Math.max(...hm, ...hp) * 1.08;
      const p = new Plot(g, {
        x: 0,
        y: 0,
        w,
        h,
        xmin: -R,
        xmax: R,
        ymin: 0,
        ymax: top,
        xticks: 5,
        yticks: [],
        xfmt: (v) => (Math.abs(v) < 1e-9 ? '0' : v.toFixed(R < 0.1 ? 3 : 2)),
        margin: { l: 14, r: 12, t: 30, b: 26 },
      });
      p.frame({ grid: false });
      p.bars(hp, -R, R, { color: 'teal', alpha: 0.55 });
      p.bars(hm, -R, R, { color: 'red', alpha: 0.5 });
      p.vline(0, { color: 'ink-2', dash: [4, 4] });
      p.vline(mm.mean, { color: 'red', dash: null, width: 2 });
      p.vline(mp.mean, { color: 'teal', dash: null, width: 2 });
      rMean.set(mp.mean.toExponential(1), 'teal');
    });

    const segB = segmented({
      label: 'total bits $b$',
      options: [1, 2, 3, 4],
      value: state.b,
      onChange: (v) => {
        state.b = v;
        cv.request();
        scatter.request();
        hist.request();
        updateA();
      },
    });
    const segM = segmented({
      label: '$m$',
      options: [
        { label: '2 (= d)', value: 2 },
        { label: '8', value: 8 },
        { label: '32', value: 32 },
      ],
      value: state.m,
      onChange: (v) => {
        state.m = v;
        redrawS();
        cv.request();
      },
    });
    const reS = button('re-draw $S$', () => {
      redrawS();
      cv.request();
    });

    const grow = (node, min) => Object.assign(node.style, { flex: '1 1 0', height: 'auto', minHeight: min });
    grow(v2, '300px');
    grow(vs, '200px');
    grow(vh, '150px');

    const algo = el('pre.algo', {
      style: { fontSize: '0.78em' },
      html: `<span class="cm">// Quant_mse with b−1 bits;  S_ij ~ N(0,1), d×d</span>
<span class="hl">Quant</span>(x):
  idx ← Quant_mse(x)
  r ← x − DeQuant_mse(idx)   <span class="cm">residual</span>
  qjl ← sign(S·r)            <span class="cm">d bits</span>
  store (idx, qjl, ‖r‖)
<span class="hl">DeQuant</span>(idx, qjl, γ):
  x̃_mse ← DeQuant_mse(idx)
  x̃ ← x̃_mse + √(π/2)/d · γ · Sᵀ·qjl`,
    });

    const colA = el(
      'div.col',
      {},
      el('div', { 'data-reveal': '1' }, algo),
      el(
        'div.card',
        { 'data-reveal': '2', style: { padding: '10px 14px', gap: '0', fontSize: '0.86em' } },
        eq('\\E\\big[\\ip{y}{\\tilde x}\\,\\big|\\,\\tilde x_{\\rm mse}\\big]=\\ip{y}{\\tilde x_{\\rm mse}}+\\ip{y}{r}=\\ip{y}{x}'),
        eq('\\Dprod\\le\\tfrac{\\pi}{2d}\\norm{y}^2\\,\\Dmse(b-1)'),
        eq('\\textbf{Thm 2:}\\ \\ \\Dprod\\le\\tfrac{\\sqrt3\\,\\pi^2\\norm{y}^2}{d}\\,4^{-b}'),
      ),
      el('div.readouts', {}, rDp.el, rProof.el, rThm.el),
      el('div.readouts', {}, rSlopeP.el, rSlopeM.el, rMean.el),
      rich('p.note', '$b=1$: no mse bits, $r=x$ ⇒ pure QJL', { 'data-reveal': '3' }),
    );
    const colB = el('div.col', {}, v2, el('div.row', {}, segB.el), el('div.row', {}, segM.el, reS));
    const colC = el(
      'div.col',
      {},
      vs,
      vh,
      legend([
        { label: 'TurboQuant$_{\\rm prod}$ (unbiased)', color: 'teal' },
        { label: 'TurboQuant$_{\\rm mse}$, same $b$', color: 'red' },
      ]),
    );
    const stage = el('div.stage.three', {}, colA, colB, colC);
    body.append(stage);

    let seen = -1;
    ctx.onEnter(() => {
      cv.request();
      scatter.request();
      hist.request();
    });
    ctx.loop((dt) => {
      if (!sim.done) sim.step(5);
      if (sim.count !== seen) {
        seen = sim.count;
        scatter.redraw();
        hist.redraw();
        updateA();
      }
      state.tAcc += dt;
      if (state.tAcc > 0.6) {
        state.tAcc = 0;
        state.cur++;
        cv.redraw();
      }
    });
  },
};

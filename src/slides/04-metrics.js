import { el, rich, eq, readout, toggle, button, viz, legend } from '../lib/ui.js';
import { createCanvas, View2D, arrow, circle, dot, line, label, gridLines, draggable, alpha, color } from '../lib/canvas.js';
import { RNG, rot2, matVec, matTVec, dot as vdot, uniformLevels, quantizeWith, clamp } from '../lib/math.js';

export default {
  id: 'metrics',
  chapter: 'Why',
  kicker: 'The scoreboard',
  title: 'Two errors to minimize',
  notes: `Drag x (orange, stays on the unit circle) and the query y (blue, anywhere).
The quantizer here is a 2-bit-per-coordinate grid (16 points). x̃ (teal) is the nearest grid point.
MSE: the squared length of the violet error vector. Inner-product error: the gap between the projections of x and x̃ onto y.
Values V are averaged, so they need small MSE. Keys K only meet the query through ⟨q,k⟩, so they need small inner-product error.
Turn on "random grid rotation" + "show 200 draws": the quantizer is now randomized, and the expectation is over that randomness, never over x (x is worst-case).
The mean of the teal cloud (white ring) sits inside the circle: the reconstruction is biased toward 0. Keep that in mind; it comes back as the central problem later.`,
  mount(body, ctx) {
    const v = viz('', 'drag $x$ and $y$');
    const levels = uniformLevels(2, -0.75, 0.75); // ±0.25, ±0.75
    const state = { x: Float64Array.of(Math.cos(0.9), Math.sin(0.9)), y: Float64Array.of(0.95, 0.25), theta: 0, randomize: false, cloud: false };
    const rng = new RNG(3);
    let thetas = Array.from({ length: 200 }, () => rng.uniform(0, 2 * Math.PI));

    const quantize = (x, theta) => {
      const R = rot2(theta);
      return matTVec(R, quantizeWith(matVec(R, x), levels));
    };

    const rMse = readout('$\\|x-\\tilde x\\|^2$');
    const rIp = readout('$\\langle y,x\\rangle$');
    const rIpq = readout('$\\langle y,\\tilde x\\rangle$');
    const rErr = readout('IP error');

    let view;
    const cv = createCanvas(v, (g, w, h) => {
      view = new View2D({ w, h, xmin: -1.35, xmax: 1.35, ymin: -1.35, ymax: 1.35, pad: 18 });
      const { x, y, theta } = state;
      gridLines(g, view, 0.25);
      circle(g, view.sx(0), view.sy(0), view.len(1), { stroke: 'line-2', width: 1.5 });

      // the (possibly rotated) grid of codewords and its cell boundaries
      const R = rot2(theta);
      for (const a of levels)
        for (const b of levels) {
          const p = matTVec(R, Float64Array.of(a, b));
          dot(g, view.sx(p[0]), view.sy(p[1]), 3, 'ink-3');
        }
      for (const m of [-0.5, 0, 0.5]) {
        for (const dir of [0, 1]) {
          const a = dir ? Float64Array.of(m, -1.3) : Float64Array.of(-1.3, m);
          const b = dir ? Float64Array.of(m, 1.3) : Float64Array.of(1.3, m);
          const pa = matTVec(R, a);
          const pb = matTVec(R, b);
          line(g, view.sx(pa[0]), view.sy(pa[1]), view.sx(pb[0]), view.sy(pb[1]), { color: 'ink-3', alpha: 0.18, dash: [3, 5] });
        }
      }

      // cloud of reconstructions under random rotations, and its mean
      if (state.cloud) {
        let mx = 0;
        let my = 0;
        for (const th of thetas) {
          const q = quantize(x, th);
          mx += q[0];
          my += q[1];
          g.fillStyle = alpha('teal', 0.35);
          g.beginPath();
          g.arc(view.sx(q[0]), view.sy(q[1]), 3, 0, Math.PI * 2);
          g.fill();
        }
        mx /= thetas.length;
        my /= thetas.length;
        circle(g, view.sx(mx), view.sy(my), 7, { stroke: 'ink', width: 2 });
        label(g, 'E[x̃]', view.sx(mx) + 10, view.sy(my) + 14, { size: 13, color: 'ink' });
      }

      const xq = quantize(x, theta);

      // query line and projections
      const yn = Math.hypot(y[0], y[1]) || 1;
      const u = [y[0] / yn, y[1] / yn];
      line(g, view.sx(-1.4 * u[0]), view.sy(-1.4 * u[1]), view.sx(1.4 * u[0]), view.sy(1.4 * u[1]), { color: 'blue', alpha: 0.25 });
      const px = vdot(x, u);
      const pq = vdot(xq, u);
      line(g, view.sx(x[0]), view.sy(x[1]), view.sx(px * u[0]), view.sy(px * u[1]), { color: 'accent', dash: [4, 4], alpha: 0.7 });
      line(g, view.sx(xq[0]), view.sy(xq[1]), view.sx(pq * u[0]), view.sy(pq * u[1]), { color: 'teal', dash: [4, 4], alpha: 0.7 });
      // gap between projections on the query line
      g.save();
      g.strokeStyle = color('red');
      g.lineWidth = 6;
      g.globalAlpha = 0.8;
      g.beginPath();
      g.moveTo(view.sx(px * u[0]), view.sy(px * u[1]));
      g.lineTo(view.sx(pq * u[0]), view.sy(pq * u[1]));
      g.stroke();
      g.restore();

      arrow(g, view.sx(0), view.sy(0), view.sx(y[0]), view.sy(y[1]), { color: 'blue', width: 3 });
      arrow(g, view.sx(0), view.sy(0), view.sx(x[0]), view.sy(x[1]), { color: 'accent', width: 3.5 });
      arrow(g, view.sx(0), view.sy(0), view.sx(xq[0]), view.sy(xq[1]), { color: 'teal', width: 2.5 });
      arrow(g, view.sx(xq[0]), view.sy(xq[1]), view.sx(x[0]), view.sy(x[1]), { color: 'violet', width: 2, head: 8 });

      label(g, 'x', view.sx(x[0] * 1.08) + 6, view.sy(x[1] * 1.08), { color: 'accent', size: 18, weight: 600 });
      label(g, 'x̃', view.sx(xq[0]) + 8, view.sy(xq[1]) - 12, { color: 'teal', size: 18, weight: 600 });
      label(g, 'y', view.sx(y[0]) + 8, view.sy(y[1]) - 10, { color: 'blue', size: 18, weight: 600 });

      const mse = (x[0] - xq[0]) ** 2 + (x[1] - xq[1]) ** 2;
      const ip = vdot(y, x);
      const ipq = vdot(y, xq);
      rMse.set(mse.toFixed(4), 'violet');
      rIp.set(ip.toFixed(3), 'accent');
      rIpq.set(ipq.toFixed(3), 'teal');
      rErr.set((ip - ipq).toFixed(3), 'red');
    });

    draggable(cv.canvas, {
      pick(px, py) {
        if (!view) return null;
        const d = (p) => Math.hypot(view.sx(p[0]) - px, view.sy(p[1]) - py);
        if (d(state.x) < 22) return 'x';
        if (d(state.y) < 22) return 'y';
        return null;
      },
      move(h, px, py) {
        const wx = view.wx(px);
        const wy = view.wy(py);
        if (h === 'x') {
          const a = Math.atan2(wy, wx);
          state.x = Float64Array.of(Math.cos(a), Math.sin(a));
        } else {
          state.y = Float64Array.of(clamp(wx, -1.3, 1.3), clamp(wy, -1.3, 1.3));
        }
        cv.request();
      },
    });

    const tRand = toggle({
      label: 'random grid rotation',
      onChange: (on) => {
        state.randomize = on;
        state.theta = on ? rng.uniform(0, 2 * Math.PI) : 0;
        cv.request();
      },
    });
    const tCloud = toggle({
      label: 'show 200 draws of $\\tilde x$',
      onChange: (on) => {
        state.cloud = on;
        cv.request();
      },
    });
    const reroll = button('re-draw', () => {
      state.theta = state.randomize ? rng.uniform(0, 2 * Math.PI) : 0;
      thetas = Array.from({ length: 200 }, () => rng.uniform(0, 2 * Math.PI));
      cv.request();
    });

    const panel = el(
      'div.col',
      {},
      el(
        'div.card',
        { 'data-reveal': '1' },
        rich('h3', 'Reconstruction · values $V$'),
        eq('\\Dmse = \\E_Q\\big[\\,\\|x-\\tilde x\\|_2^2\\,\\big]'),
      ),
      el(
        'div.card',
        { 'data-reveal': '2' },
        rich('h3', 'Inner product · keys $K$'),
        eq('\\Dprod = \\E_Q\\big[\\,|\\langle y,x\\rangle-\\langle y,\\tilde x\\rangle|^2\\,\\big]'),
      ),
      el(
        'div.card',
        { 'data-reveal': '3' },
        el('h3', {}, 'Bonus: unbiased'),
        eq('\\E_Q\\big[\\langle y,\\tilde x\\rangle\\big] = \\langle y,x\\rangle'),
      ),
      el('div.readouts', {}, rMse.el, rIp.el, rIpq.el, rErr.el),
      el('div.row', {}, tRand.el, tCloud.el, reroll),
      legend([
        { label: '$x$ (data)', color: 'accent' },
        { label: '$\\tilde x$', color: 'teal' },
        { label: '$y$ (query)', color: 'blue' },
        { label: 'error', color: 'violet' },
      ]),
    );

    body.append(el('div.stage', {}, v, panel));
    ctx.onEnter(() => cv.request());
  },
};

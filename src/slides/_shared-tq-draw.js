// Canvas drawing helpers shared by the TurboQuant slides (17–21).
import { Plot } from '../lib/plot.js';
import { alpha, color, label, text, FONT } from '../lib/canvas.js';
import { bounds } from '../lib/math.js';
import { BMAX, PAPER, measuredDmse } from './_shared-tq.js';

/** Fill the band between two curves on a Plot (x linear, any y scale). */
export function band(p, lo, hi, { x0, x1, color: c = 'teal', alpha: a = 0.1, samples = 80 } = {}) {
  const a0 = x0 ?? p.o.xmin;
  const a1 = x1 ?? p.o.xmax;
  p.clip(() => {
    const g = p.ctx;
    g.fillStyle = alpha(c, a);
    g.beginPath();
    for (let i = 0; i <= samples; i++) {
      const x = a0 + ((a1 - a0) * i) / samples;
      const X = p.tx(x);
      const Y = p.ty(hi(x));
      if (i === 0) g.moveTo(X, Y);
      else g.lineTo(X, Y);
    }
    for (let i = samples; i >= 0; i--) {
      const x = a0 + ((a1 - a0) * i) / samples;
      g.lineTo(p.tx(x), p.ty(lo(x)));
    }
    g.closePath();
    g.fill();
  });
}

/** Hollow square marker (paper values). */
export function square(p, x, y, { color: c = 'yellow', r = 5, width = 2 } = {}) {
  const g = p.ctx;
  g.save();
  g.strokeStyle = color(c);
  g.lineWidth = width;
  g.strokeRect(p.tx(x) - r, p.ty(y) - r, 2 * r, 2 * r);
  g.restore();
}

/** Filled dot marker. */
export function disc(p, x, y, { color: c = 'teal', r = 5, a = 1 } = {}) {
  const g = p.ctx;
  g.save();
  g.fillStyle = alpha(c, a);
  g.beginPath();
  g.arc(p.tx(x), p.ty(y), r, 0, Math.PI * 2);
  g.fill();
  g.restore();
}

/** "simulating… 42%" badge in a plot's top-left corner. */
export function progressBadge(p, progress) {
  if (progress >= 1) return;
  text(p.ctx, `live simulation… ${Math.round(100 * progress)}%`, p.px + 10, p.py + 12, { color: 'ink-3', size: 12, font: 'mono' });
}

/**
 * Log-y plot of D_mse vs b: shaded gap between 4^{-b} and (√3π/2)·4^{-b},
 * paper values (Fig. 3b), and live-simulation dots at d = 128.
 */
export function drawDmsePlot(g, w, h, { sim, highlight = null, annotate = true, paper = 'fig3b', top = 12 } = {}) {
  if (w < 40 || h < 40) return null;
  const p = new Plot(g, {
    x: 0,
    y: 0,
    w,
    h,
    xmin: 0.6,
    xmax: BMAX + 0.8,
    ymin: 1e-3,
    ymax: 1.5,
    ylog: true,
    xticks: [1, 2, 3, 4, 5],
    xfmt: (v) => `b=${v}`,
    ylabel: 'D_mse',
    margin: { l: 58, r: 16, t: top, b: 30 },
  });
  p.frame();
  if (highlight) {
    p.clip(() => {
      g.fillStyle = alpha('yellow', 0.08);
      g.fillRect(p.tx(highlight - 0.35), p.py, p.tx(highlight + 0.35) - p.tx(highlight - 0.35), p.ph);
    });
  }
  band(p, bounds.mseLower, bounds.mseUpper, { color: 'teal', alpha: 0.1 });
  p.fn(bounds.mseUpper, { color: 'red', width: 1.5, dash: [6, 4] });
  p.fn(bounds.mseLower, { color: 'blue', width: 1.5, dash: [6, 4] });
  label(g, '(√3π/2)·4⁻ᵇ', p.tx(1.9), p.ty(bounds.mseUpper(1.9)) - 16, { color: 'red', size: 12, align: 'left' });
  label(g, '4⁻ᵇ', p.tx(1.55), p.ty(bounds.mseLower(1.55)) + 16, { color: 'blue', size: 12, align: 'right' });

  const pv = paper === 'thm1' ? PAPER.mseThm1 : PAPER.mseFig3b;
  for (let b = 1; b < pv.length; b++) square(p, b, pv[b], { color: 'yellow', r: 6 });

  if (sim && sim.count > 0) {
    for (let b = 1; b <= BMAX; b++) {
      const v = measuredDmse(b, sim);
      disc(p, b, v, { color: 'teal', r: 4.5 });
      if (annotate) {
        g.save();
        g.font = `12px ${FONT.mono}`;
        g.fillStyle = color('teal');
        g.textAlign = 'left';
        g.textBaseline = 'middle';
        g.fillText(v < 0.01 ? v.toFixed(4) : v.toFixed(3), p.tx(b) + 10, p.ty(v) + 1);
        g.restore();
      }
    }
  }
  progressBadge(p, sim ? sim.progress : 0);
  return p;
}

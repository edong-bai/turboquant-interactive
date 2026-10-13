// Minimal 2-D plotting on a canvas region: axes, ticks, curves, bars, points. Linear or log axes.
import { color, alpha, FONT } from './canvas.js';

const fmt = (v) => {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1000 || a < 0.01) return v.toExponential(0).replace('e+', 'e');
  if (a >= 10) return v.toFixed(0);
  if (a >= 1) return Number(v.toFixed(1)).toString();
  return Number(v.toFixed(3)).toString();
};

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * step; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.ceil(Math.log10(lo) - 1e-9); e <= Math.floor(Math.log10(hi) + 1e-9); e++) out.push(10 ** e);
  return out;
}

export class Plot {
  /**
   * @param ctx canvas 2d context
   * @param opts {x,y,w,h} pixel box (including room for tick labels),
   *   xmin,xmax,ymin,ymax, xlog, ylog, xlabel, ylabel, xticks, yticks (arrays or count),
   *   xfmt/yfmt (value → string), margin {l,r,t,b}
   */
  constructor(ctx, opts) {
    this.ctx = ctx;
    this.o = { margin: { l: 52, r: 14, t: 14, b: 40 }, ...opts };
    const m = { l: 52, r: 14, t: 14, b: 40, ...(opts.margin || {}) };
    this.px = opts.x + m.l;
    this.py = opts.y + m.t;
    this.pw = Math.max(10, opts.w - m.l - m.r);
    this.ph = Math.max(10, opts.h - m.t - m.b);
  }
  tx(v) {
    const { xmin, xmax, xlog } = this.o;
    const t = xlog ? (Math.log(v) - Math.log(xmin)) / (Math.log(xmax) - Math.log(xmin)) : (v - xmin) / (xmax - xmin);
    return this.px + t * this.pw;
  }
  ty(v) {
    const { ymin, ymax, ylog } = this.o;
    const t = ylog ? (Math.log(v) - Math.log(ymin)) / (Math.log(ymax) - Math.log(ymin)) : (v - ymin) / (ymax - ymin);
    return this.py + (1 - t) * this.ph;
  }
  /** Pixel x → data value. */
  ix(px) {
    const { xmin, xmax, xlog } = this.o;
    const t = (px - this.px) / this.pw;
    return xlog ? Math.exp(Math.log(xmin) + t * (Math.log(xmax) - Math.log(xmin))) : xmin + t * (xmax - xmin);
  }
  frame({ grid = true } = {}) {
    const { ctx, o } = this;
    const xt = Array.isArray(o.xticks) ? o.xticks : o.xlog ? logTicks(o.xmin, o.xmax) : niceTicks(o.xmin, o.xmax, o.xticks || 6);
    const yt = Array.isArray(o.yticks) ? o.yticks : o.ylog ? logTicks(o.ymin, o.ymax) : niceTicks(o.ymin, o.ymax, o.yticks || 5);
    const xf = o.xfmt || fmt;
    const yf = o.yfmt || fmt;
    ctx.save();
    ctx.font = `12px ${FONT.mono}`;
    ctx.fillStyle = color('ink-3');
    ctx.strokeStyle = color('grid');
    ctx.lineWidth = 1;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const v of xt) {
      const x = this.tx(v);
      if (grid) {
        ctx.beginPath();
        ctx.moveTo(x, this.py);
        ctx.lineTo(x, this.py + this.ph);
        ctx.stroke();
      }
      ctx.fillText(xf(v), x, this.py + this.ph + 6);
    }
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const v of yt) {
      const y = this.ty(v);
      if (grid) {
        ctx.beginPath();
        ctx.moveTo(this.px, y);
        ctx.lineTo(this.px + this.pw, y);
        ctx.stroke();
      }
      ctx.fillText(yf(v), this.px - 6, y);
    }
    ctx.strokeStyle = color('line-2');
    ctx.beginPath();
    ctx.moveTo(this.px, this.py);
    ctx.lineTo(this.px, this.py + this.ph);
    ctx.lineTo(this.px + this.pw, this.py + this.ph);
    ctx.stroke();
    ctx.font = `13px ${FONT.sans}`;
    ctx.fillStyle = color('ink-2');
    if (o.xlabel) {
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(o.xlabel, this.px + this.pw / 2, o.y + o.h - 2);
    }
    if (o.ylabel) {
      ctx.save();
      ctx.translate(o.x + 12, this.py + this.ph / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(o.ylabel, 0, 0);
      ctx.restore();
    }
    ctx.restore();
    return this;
  }
  clip(fn) {
    const { ctx } = this;
    ctx.save();
    ctx.beginPath();
    ctx.rect(this.px, this.py - 1, this.pw, this.ph + 2);
    ctx.clip();
    fn();
    ctx.restore();
  }
  line(xs, ys, { color: c = 'accent', width = 2, dash = null, alpha: a } = {}) {
    this.clip(() => {
      const { ctx } = this;
      ctx.strokeStyle = a != null ? alpha(c, a) : color(c);
      ctx.lineWidth = width;
      ctx.lineJoin = 'round';
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < xs.length; i++) {
        const X = this.tx(xs[i]);
        const Y = this.ty(ys[i]);
        if (!Number.isFinite(X) || !Number.isFinite(Y)) {
          started = false;
          continue;
        }
        if (!started) ctx.moveTo(X, Y);
        else ctx.lineTo(X, Y);
        started = true;
      }
      ctx.stroke();
    });
    return this;
  }
  /** Plot y = f(x) over the x range. */
  fn(f, { samples = 300, lo, hi, ...style } = {}) {
    const a = lo ?? this.o.xmin;
    const b = hi ?? this.o.xmax;
    const xs = [];
    const ys = [];
    for (let i = 0; i <= samples; i++) {
      const x = this.o.xlog ? a * (b / a) ** (i / samples) : a + ((b - a) * i) / samples;
      xs.push(x);
      ys.push(f(x));
    }
    return this.line(xs, ys, style);
  }
  /** Shade between y = f(x) and the baseline (ymin, or `base`). */
  area(f, { lo, hi, samples = 200, color: c = 'accent', alpha: a = 0.25, base } = {}) {
    const x0 = lo ?? this.o.xmin;
    const x1 = hi ?? this.o.xmax;
    const yb = this.ty(base ?? (this.o.ylog ? this.o.ymin : Math.max(this.o.ymin, 0)));
    this.clip(() => {
      const { ctx } = this;
      ctx.fillStyle = alpha(c, a);
      ctx.beginPath();
      ctx.moveTo(this.tx(x0), yb);
      for (let i = 0; i <= samples; i++) {
        const x = this.o.xlog ? x0 * (x1 / x0) ** (i / samples) : x0 + ((x1 - x0) * i) / samples;
        ctx.lineTo(this.tx(x), this.ty(f(x)));
      }
      ctx.lineTo(this.tx(x1), yb);
      ctx.closePath();
      ctx.fill();
    });
    return this;
  }
  /** Histogram bars: `h` densities on equal bins over [lo, hi]. */
  bars(h, lo, hi, { color: c = 'blue', alpha: a = 0.55, gap = 1 } = {}) {
    const w = (hi - lo) / h.length;
    this.clip(() => {
      const { ctx } = this;
      ctx.fillStyle = alpha(c, a);
      const y0 = this.ty(Math.max(this.o.ymin, 0));
      for (let i = 0; i < h.length; i++) {
        const x0 = this.tx(lo + i * w);
        const x1 = this.tx(lo + (i + 1) * w);
        const y = this.ty(h[i]);
        ctx.fillRect(x0 + gap / 2, y, Math.max(0.5, x1 - x0 - gap), y0 - y);
      }
    });
    return this;
  }
  points(xs, ys, { color: c = 'blue', r = 2.5, alpha: a = 0.7 } = {}) {
    this.clip(() => {
      const { ctx } = this;
      ctx.fillStyle = alpha(c, a);
      for (let i = 0; i < xs.length; i++) {
        ctx.beginPath();
        ctx.arc(this.tx(xs[i]), this.ty(ys[i]), r, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    return this;
  }
  vline(x, { color: c = 'ink-3', width = 1, dash = [4, 4] } = {}) {
    const { ctx } = this;
    ctx.save();
    ctx.strokeStyle = color(c);
    ctx.lineWidth = width;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(this.tx(x), this.py);
    ctx.lineTo(this.tx(x), this.py + this.ph);
    ctx.stroke();
    ctx.restore();
    return this;
  }
  hline(y, { color: c = 'ink-3', width = 1, dash = [4, 4] } = {}) {
    const { ctx } = this;
    ctx.save();
    ctx.strokeStyle = color(c);
    ctx.lineWidth = width;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(this.px, this.ty(y));
    ctx.lineTo(this.px + this.pw, this.ty(y));
    ctx.stroke();
    ctx.restore();
    return this;
  }
  /** Legend box in the top-right (or `where`: 'tl', 'tr', 'bl', 'br'). items: [{label, color, dash}] */
  legend(items, where = 'tr') {
    const { ctx } = this;
    ctx.save();
    ctx.font = `13px ${FONT.sans}`;
    const lh = 20;
    const w = Math.max(...items.map((i) => ctx.measureText(i.label).width)) + 40;
    const h = items.length * lh + 10;
    const x = where.includes('l') ? this.px + 10 : this.px + this.pw - w - 10;
    const y = where.includes('b') ? this.py + this.ph - h - 10 : this.py + 8;
    ctx.fillStyle = 'rgba(10,16,30,0.75)';
    ctx.fillRect(x, y, w, h);
    items.forEach((it, i) => {
      const yy = y + 5 + lh * i + lh / 2;
      ctx.strokeStyle = color(it.color);
      ctx.lineWidth = 3;
      ctx.setLineDash(it.dash || []);
      ctx.beginPath();
      ctx.moveTo(x + 8, yy);
      ctx.lineTo(x + 28, yy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color('ink-2');
      ctx.textBaseline = 'middle';
      ctx.fillText(it.label, x + 34, yy);
    });
    ctx.restore();
    return this;
  }
}

// Helpers shared by the QJL slides (10-qjl, 11-qjl-why):
// draggable q/k handles, random-direction lines, half-plane shading, a sample queue
// that animates "+1 / +100" and a number-line histogram strip with markers.
import { line, label, text, draggable, alpha, color } from '../lib/canvas.js';
import { clamp } from '../lib/math.js';

/** Drag q (blue) and k (orange) in a View2D; state.q / state.k are [x, y]. */
export function dragQK(cv, getView, state, onChange, { rmin = 0.25, rmax = 1.4 } = {}) {
  draggable(cv.canvas, {
    pick(px, py) {
      const view = getView();
      if (!view) return null;
      const d = (p) => Math.hypot(view.sx(p[0]) - px, view.sy(p[1]) - py);
      const dk = d(state.k);
      const dq = d(state.q);
      if (Math.min(dk, dq) > 24) return null;
      return dk <= dq ? 'k' : 'q';
    },
    move(h, px, py) {
      const view = getView();
      let x = view.wx(px);
      let y = view.wy(py);
      const r = Math.hypot(x, y);
      const rc = clamp(r, rmin, rmax);
      if (r > 1e-9) {
        x *= rc / r;
        y *= rc / r;
      } else {
        x = rmin;
        y = 0;
      }
      state[h] = [x, y];
      onChange();
    },
  });
}

/** Clip drawing to a pixel rectangle. */
export function clipRect(g, box, fn) {
  g.save();
  g.beginPath();
  g.rect(box.x, box.y, box.w, box.h);
  g.clip();
  fn();
  g.restore();
}

/** Line through the origin along direction s (world coords), clipped by the caller. */
export function dirLine(g, view, s, opts) {
  const n = Math.hypot(s[0], s[1]) || 1;
  const L = 10;
  const u = [s[0] / n, s[1] / n];
  line(g, view.sx(-L * u[0]), view.sy(-L * u[1]), view.sx(L * u[0]), view.sy(L * u[1]), opts);
}

/** Shade the half-plane {x : n·x > 0} (caller clips to the plane box). */
export function halfPlane(g, view, n, c, a = 0.08) {
  const r = Math.hypot(n[0], n[1]) || 1;
  const nn = [n[0] / r, n[1] / r];
  const u = [-nn[1], nn[0]];
  const L = 10;
  const pts = [
    [L * u[0], L * u[1]],
    [L * u[0] + L * nn[0], L * u[1] + L * nn[1]],
    [-L * u[0] + L * nn[0], -L * u[1] + L * nn[1]],
    [-L * u[0], -L * u[1]],
  ];
  g.save();
  g.fillStyle = alpha(c, a);
  g.beginPath();
  pts.forEach(([x, y], i) => (i ? g.lineTo(view.sx(x), view.sy(y)) : g.moveTo(view.sx(x), view.sy(y))));
  g.closePath();
  g.fill();
  g.restore();
}

/**
 * Queue of random 2-D Gaussian directions s ~ N(0, I₂). `add(n)` schedules n samples;
 * `step(dt)` releases a few per frame so "+100" animates. `recent` holds the newest
 * samples with their age (seconds) for drop/flash effects.
 */
export class DirSampler {
  constructor(rng, max = 6000) {
    this.rng = rng;
    this.max = max;
    this.reset();
  }
  reset() {
    this.s = [];
    this.pending = 0;
    this.recent = [];
  }
  add(n) {
    this.pending = Math.min(this.pending + n, this.max - this.s.length);
  }
  get busy() {
    return this.pending > 0 || this.recent.length > 0;
  }
  /** Returns the number of samples released this frame. */
  step(dt, perFrame = 4) {
    for (const r of this.recent) r.age += dt;
    this.recent = this.recent.filter((r) => r.age < 0.9);
    // A handful of samples play one at a time so each line can be seen; bigger batches stream.
    this.clock = (this.clock || 0) + dt;
    let n = 0;
    if (this.pending > 12) n = Math.min(this.pending, perFrame);
    else if (this.pending > 0 && this.clock > 0.22) n = 1;
    if (n) this.clock = 0;
    for (let i = 0; i < n; i++) {
      this.s.push([this.rng.normal(), this.rng.normal()]);
      this.recent.push({ i: this.s.length - 1, age: 0 });
    }
    this.pending -= n;
    if (this.recent.length > 60) this.recent.splice(0, this.recent.length - 60);
    return n;
  }
}

/** Running means of an array. */
export function runningMeans(vals) {
  const out = new Float64Array(vals.length);
  let s = 0;
  for (let i = 0; i < vals.length; i++) {
    s += vals[i];
    out[i] = s / (i + 1);
  }
  return out;
}

const fmtTick = (v) => (Math.abs(v) < 1e-9 ? '0' : Math.abs(v) >= 10 ? v.toFixed(0) : Number(v.toFixed(2)).toString());

/**
 * Number-line histogram strip in the pixel box {x, y, w, h}.
 * opts: lo, hi, bins, values (array), color, markers [{v, color, label, dash}],
 *       drops [{v, t}] (t∈[0,1] fall progress), pdf (density to overlay, same units as values),
 *       title (top-left caption).
 */
export function strip(g, box, opts) {
  const { lo, hi, bins = 56, values = [], color: c = 'blue', markers = [], drops = [], pdf = null, title = '' } = opts;
  const X = (v) => box.x + ((clamp(v, lo, hi) - lo) / (hi - lo)) * box.w;
  const base = box.y + box.h - 20;
  const top = box.y + (title ? 26 : 8);
  const H = Math.max(10, base - top - 4);

  if (title) text(g, title, box.x, box.y + 9, { size: 13, color: 'ink-3' });

  // histogram
  const counts = new Float64Array(bins);
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    const k = clamp(Math.floor(((v - lo) / (hi - lo)) * bins), 0, bins - 1);
    counts[k]++;
  }
  let maxc = 0;
  for (const n of counts) maxc = Math.max(maxc, n);
  const bw = box.w / bins;
  const scaleH = maxc > 0 ? H / maxc : 0;
  g.save();
  g.fillStyle = alpha(c, 0.5);
  for (let i = 0; i < bins; i++) {
    if (!counts[i]) continue;
    const hh = counts[i] * scaleH;
    g.fillRect(box.x + i * bw + 0.5, base - hh, Math.max(0.5, bw - 1), hh);
  }
  g.restore();

  // density overlay (expected counts per bin)
  if (pdf && values.length > 0 && maxc > 0) {
    const width = (hi - lo) / bins;
    g.save();
    g.strokeStyle = color('ink');
    g.globalAlpha = 0.7;
    g.lineWidth = 1.5;
    g.beginPath();
    for (let i = 0; i <= 200; i++) {
      const v = lo + ((hi - lo) * i) / 200;
      const y = base - Math.min(H * 1.15, values.length * width * pdf(v) * scaleH);
      if (i) g.lineTo(X(v), y);
      else g.moveTo(X(v), y);
    }
    g.stroke();
    g.restore();
  }

  // axis + ticks
  line(g, box.x, base, box.x + box.w, base, { color: 'line-2', width: 1 });
  const span = hi - lo;
  const raw = span / 6;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= 6) || 10 * mag;
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
    const x = X(v);
    line(g, x, base, x, base + 4, { color: 'line-2' });
    text(g, fmtTick(v), x, base + 12, { size: 11, color: 'ink-3', align: 'center', font: 'mono' });
  }

  // falling dots
  for (const d of drops) {
    const t = clamp(d.t, 0, 1);
    const y = top + (base - 4 - top) * t * t;
    g.save();
    g.globalAlpha = 1 - 0.6 * t;
    g.fillStyle = color(c);
    g.beginPath();
    g.arc(X(d.v), y, 3.5, 0, Math.PI * 2);
    g.fill();
    g.restore();
  }

  // markers
  markers.forEach((m, i) => {
    if (m.v == null || !Number.isFinite(m.v)) return;
    const x = X(m.v);
    line(g, x, top - 4, x, base, { color: m.color || 'ink', width: m.width || 2, dash: m.dash || null });
    if (m.label) {
      const right = x < box.x + box.w * 0.7;
      label(g, m.label, x + (right ? 6 : -6), top + 4 + i * 17, {
        size: 12.5,
        color: m.color || 'ink',
        align: right ? 'left' : 'right',
      });
    }
  });
  return { X, base, top, H };
}

// Canvas helpers: HiDPI sizing, theme colors, world↔pixel views, drawing primitives, dragging.

const colorCache = {};

/** Resolve a CSS custom property (e.g. 'accent' → value of --accent). */
export function color(name) {
  if (name.startsWith('#') || name.startsWith('rgb')) return name;
  if (!(name in colorCache)) {
    colorCache[name] = getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim() || name;
  }
  return colorCache[name];
}

/** Hex/rgb color with alpha. */
export function alpha(c, a) {
  const v = color(c);
  if (v.startsWith('#')) {
    const n = v.length === 4 ? v.replace(/#(.)(.)(.)/, '#$1$1$2$2$3$3') : v;
    const r = parseInt(n.slice(1, 3), 16);
    const g = parseInt(n.slice(3, 5), 16);
    const b = parseInt(n.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${a})`;
  }
  return v;
}

export const FONT = {
  sans: "'IBM Plex Sans', system-ui, sans-serif",
  mono: "'JetBrains Mono', ui-monospace, monospace",
  serif: "'Source Serif 4 Variable', Georgia, serif",
};

/**
 * Create a canvas that fills `container` (a .viz element), sized for devicePixelRatio.
 * `draw(ctx, w, h)` is called on resize and whenever you call `redraw()`.
 * Returns { canvas, ctx, get w, get h, redraw }.
 */
export function createCanvas(container, draw) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block;';
  if (getComputedStyle(container).position === 'static') container.style.position = 'relative';
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  const state = { w: 0, h: 0, dpr: 1 };
  let queued = false;

  function resize() {
    const r = container.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    state.w = Math.max(1, r.width);
    state.h = Math.max(1, r.height);
    state.dpr = dpr;
    canvas.width = Math.round(state.w * dpr);
    canvas.height = Math.round(state.h * dpr);
    paint();
  }

  function paint() {
    queued = false;
    ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    ctx.clearRect(0, 0, state.w, state.h);
    // Skip until the container has a real layout (it is 1×1 while detached).
    if (draw && state.w >= 8 && state.h >= 8) draw(ctx, state.w, state.h);
  }

  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  return {
    canvas,
    ctx,
    get w() {
      return state.w;
    },
    get h() {
      return state.h;
    },
    /** Repaint now (use inside an animation loop). */
    redraw: paint,
    /** Coalesce multiple repaint requests into one frame (use from input handlers). */
    request() {
      if (!queued) {
        queued = true;
        requestAnimationFrame(paint);
      }
    },
  };
}

/**
 * Maps a world rectangle into a pixel rectangle. With `equal: true` (default) it keeps
 * aspect ratio 1 and centers the world box inside the pixel box.
 */
export class View2D {
  constructor({ x = 0, y = 0, w, h, xmin = -1, xmax = 1, ymin = -1, ymax = 1, equal = true, pad = 0 }) {
    this.set({ x, y, w, h, xmin, xmax, ymin, ymax, equal, pad });
  }
  set({ x, y, w, h, xmin, xmax, ymin, ymax, equal = true, pad = 0 }) {
    Object.assign(this, { xmin, xmax, ymin, ymax });
    const iw = w - 2 * pad;
    const ih = h - 2 * pad;
    let kx = iw / (xmax - xmin);
    let ky = ih / (ymax - ymin);
    if (equal) kx = ky = Math.min(kx, ky);
    this.kx = kx;
    this.ky = ky;
    this.ox = x + pad + (iw - kx * (xmax - xmin)) / 2;
    this.oy = y + pad + (ih - ky * (ymax - ymin)) / 2;
    return this;
  }
  sx(x) {
    return this.ox + (x - this.xmin) * this.kx;
  }
  sy(y) {
    return this.oy + (this.ymax - y) * this.ky;
  }
  /** World length → pixels (x scale). */
  len(l) {
    return l * this.kx;
  }
  wx(px) {
    return this.xmin + (px - this.ox) / this.kx;
  }
  wy(py) {
    return this.ymax - (py - this.oy) / this.ky;
  }
}

// ---------- Primitives (pixel coordinates) ----------

export function line(ctx, x0, y0, x1, y1, { color: c = 'ink-3', width = 1, dash = null, alpha: a } = {}) {
  ctx.save();
  ctx.strokeStyle = a != null ? alpha(c, a) : color(c);
  ctx.lineWidth = width;
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.restore();
}

export function arrow(ctx, x0, y0, x1, y1, { color: c = 'accent', width = 3, head = 12, dash = null, alpha: a } = {}) {
  const ang = Math.atan2(y1 - y0, x1 - x0);
  const L = Math.hypot(x1 - x0, y1 - y0);
  const hh = Math.min(head, L * 0.6);
  const col = a != null ? alpha(c, a) : color(c);
  ctx.save();
  ctx.strokeStyle = col;
  ctx.fillStyle = col;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1 - Math.cos(ang) * hh * 0.8, y1 - Math.sin(ang) * hh * 0.8);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - hh * Math.cos(ang - 0.4), y1 - hh * Math.sin(ang - 0.4));
  ctx.lineTo(x1 - hh * Math.cos(ang + 0.4), y1 - hh * Math.sin(ang + 0.4));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

export function circle(ctx, x, y, r, { stroke = null, fill = null, width = 1, dash = null } = {}) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, Math.max(0, r), 0, Math.PI * 2);
  if (fill) {
    ctx.fillStyle = color(fill);
    ctx.fill();
  }
  if (stroke) {
    ctx.strokeStyle = color(stroke);
    ctx.lineWidth = width;
    if (dash) ctx.setLineDash(dash);
    ctx.stroke();
  }
  ctx.restore();
}

export function dot(ctx, x, y, r = 4, c = 'ink') {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = color(c);
  ctx.fill();
}

export function text(ctx, str, x, y, { color: c = 'ink-2', size = 14, align = 'left', baseline = 'middle', font = 'sans', weight = 400, alpha: a } = {}) {
  ctx.save();
  ctx.fillStyle = a != null ? alpha(c, a) : color(c);
  ctx.font = `${weight} ${size}px ${FONT[font] || font}`;
  ctx.textAlign = align;
  ctx.textBaseline = baseline;
  ctx.fillText(str, x, y);
  ctx.restore();
}

/** Text with a dark halo so it stays readable over busy plots. */
export function label(ctx, str, x, y, opts = {}) {
  ctx.save();
  ctx.font = `${opts.weight || 500} ${opts.size || 14}px ${FONT[opts.font || 'sans']}`;
  ctx.textAlign = opts.align || 'left';
  ctx.textBaseline = opts.baseline || 'middle';
  ctx.lineWidth = 4;
  ctx.strokeStyle = 'rgba(10,16,30,0.85)';
  ctx.lineJoin = 'round';
  ctx.strokeText(str, x, y);
  ctx.fillStyle = color(opts.color || 'ink');
  ctx.fillText(str, x, y);
  ctx.restore();
}

/** Light background grid lines in world units for a View2D. */
export function gridLines(ctx, view, step = 0.5, { color: c = 'grid' } = {}) {
  ctx.save();
  ctx.strokeStyle = color(c);
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = Math.ceil(view.xmin / step) * step; x <= view.xmax + 1e-9; x += step) {
    ctx.moveTo(view.sx(x), view.sy(view.ymin));
    ctx.lineTo(view.sx(x), view.sy(view.ymax));
  }
  for (let y = Math.ceil(view.ymin / step) * step; y <= view.ymax + 1e-9; y += step) {
    ctx.moveTo(view.sx(view.xmin), view.sy(y));
    ctx.lineTo(view.sx(view.xmax), view.sy(y));
  }
  ctx.stroke();
  ctx.restore();
}

/**
 * Pointer dragging on a canvas. `pick(px, py)` returns a handle (truthy) or null;
 * `move(handle, px, py)` updates it. Coordinates are CSS pixels relative to the canvas.
 */
export function draggable(canvas, { pick, move, end, hover }) {
  let active = null;
  const pos = (e) => {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  canvas.addEventListener('pointerdown', (e) => {
    const [x, y] = pos(e);
    const h = pick(x, y);
    if (h) {
      active = h;
      canvas.setPointerCapture(e.pointerId);
      move(active, x, y);
      e.preventDefault();
    }
  });
  canvas.addEventListener('pointermove', (e) => {
    const [x, y] = pos(e);
    if (active) move(active, x, y);
    else {
      const h = pick(x, y);
      canvas.style.cursor = h ? 'grab' : 'default';
      if (hover) hover(h, x, y);
    }
  });
  const stop = () => {
    if (active && end) end(active);
    active = null;
  };
  canvas.addEventListener('pointerup', stop);
  canvas.addEventListener('pointercancel', stop);
}

// A tiny 3-D camera for canvas sketches: drag to orbit, optional auto-spin, painter's-order helpers.
import { color, alpha } from './canvas.js';

export class Orbit3D {
  /**
   * @param canvas element to attach drag handlers to
   * @param opts {yaw, pitch, spin (rad/s), zoom (fraction of min(w,h) for unit length), perspective}
   */
  constructor(canvas, { yaw = 0.6, pitch = 0.35, spin = 0.15, zoom = 0.36, perspective = 3.2 } = {}) {
    Object.assign(this, { yaw, pitch, spin, zoom, perspective });
    this.cx = 0;
    this.cy = 0;
    this.k = 100;
    this.dragging = false;
    let last = null;
    canvas.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || this.locked) return;
      this.dragging = true;
      last = [e.clientX, e.clientY];
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      this.yaw += (e.clientX - last[0]) * 0.008;
      this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch + (e.clientY - last[1]) * 0.008));
      last = [e.clientX, e.clientY];
      if (this.onChange) this.onChange();
    });
    const up = () => (this.dragging = false);
    canvas.addEventListener('pointerup', up);
    canvas.addEventListener('pointercancel', up);
  }
  /** Set the viewport: center (cx, cy) in px and unit length k in px. */
  viewport(cx, cy, k) {
    this.cx = cx;
    this.cy = cy;
    this.k = k;
  }
  tick(dt) {
    if (!this.dragging && this.spin) this.yaw += this.spin * dt;
  }
  /** World (x, y, z) with z up → {x, y (px), z (depth, larger = closer), s (perspective scale)}. */
  project(x, y, z) {
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);
    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    // yaw about z, then pitch about the screen x-axis
    const x1 = cy * x - sy * y;
    const y1 = sy * x + cy * y;
    const depth = y1 * cp + z * sp; // toward viewer (negated below)
    const up = z * cp - y1 * sp;
    const s = this.perspective ? this.perspective / (this.perspective + depth) : 1;
    return { x: this.cx + x1 * this.k * s, y: this.cy - up * this.k * s, z: -depth, s };
  }
  /** Draw a 3-D polyline. */
  path(ctx, pts, { color: c = 'ink-3', width = 1, alpha: a, dash, closed = false } = {}) {
    ctx.save();
    ctx.strokeStyle = a != null ? alpha(c, a) : color(c);
    ctx.lineWidth = width;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    pts.forEach((p, i) => {
      const q = this.project(p[0], p[1], p[2]);
      if (i === 0) ctx.moveTo(q.x, q.y);
      else ctx.lineTo(q.x, q.y);
    });
    if (closed) ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }
  /** Circle of latitude at height z on the unit sphere (radius √(1−z²)). */
  latitude(z, n = 96) {
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const t = (i / n) * Math.PI * 2;
      pts.push([r * Math.cos(t), r * Math.sin(t), z]);
    }
    return pts;
  }
  /** Draw the unit sphere's silhouette + a few faint latitude/longitude rings. */
  sphere(ctx, { rings = 6, c = 'ink-3', a = 0.18, radius = 1 } = {}) {
    for (let i = 1; i < rings; i++) {
      const z = -1 + (2 * i) / rings;
      this.path(ctx, this.latitude(z).map((p) => p.map((v) => v * radius)), { color: c, alpha: a });
    }
    for (let j = 0; j < rings; j++) {
      const phi = (j / rings) * Math.PI;
      const pts = [];
      for (let i = 0; i <= 96; i++) {
        const t = (i / 96) * Math.PI * 2;
        pts.push([radius * Math.cos(t) * Math.cos(phi), radius * Math.cos(t) * Math.sin(phi), radius * Math.sin(t)]);
      }
      this.path(ctx, pts, { color: c, alpha: a });
    }
  }
}

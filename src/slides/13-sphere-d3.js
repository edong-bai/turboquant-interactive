import { el, rich, eq, slider, readout, viz } from '../lib/ui.js';
import { createCanvas, line, circle, label, text, draggable, alpha, color } from '../lib/canvas.js';
import { Orbit3D } from '../lib/orbit3d.js';
import * as M from '../lib/math.js';
import { responsiveGrid } from './_shared-layout.js';

const N = 1500;
const BINS = 20;

export default {
  id: 'sphere-d3',
  chapter: 'Rotation',
  kicker: 'Where do coordinates land?',
  title: '$d=3$: every band holds the same mass',
  notes: `After a random rotation, Πx is a uniform point on the sphere. So the question "what does one coordinate of Πx look like?" is: pick a uniform point on S^{d−1}, look at one coordinate x_j. Start with d = 3, where we can see it.

Left: 1500 uniform random points on the 2-sphere (live sample, drag to orbit). The yellow band is all points whose height (one coordinate, z = x_j) lies in [x − dx/2, x + dx/2]. Middle: histogram of the height of all 1500 points, drawn sideways so "up" means the same thing in every panel. It is flat at 1/2: every band of the same width holds the same fraction of points, whether it is around the equator or near a pole. That is Archimedes' hat-box theorem (the sphere's area between two heights equals the area of the circumscribed cylinder between those heights).

Drag the x slider (or drag in the middle/right panels) from the equator toward a pole. Two things change, and they cancel:
1. The slice shrinks. The band is a ring of radius r = √(1 − x²), so its circumference is 2πr: proportional to r. Near the pole r → 0.
2. The band widens along the surface. Near the pole the sphere is tilted almost flat, so a height step dx covers more surface: arc length ds = dx / r.
Product: 2πr · (1/r) = 2π, divided by the total area 4π gives 1/2, flat.

Right panel, the "Pythagorean correction": the sphere cut through the x_j axis is a circle (up = x_j, sideways = r). The band is the yellow strip; it meets the circle in a short arc. Zoom box: that arc is, to first order, the hypotenuse ds of a small right triangle with legs dx (vertical) and |dr| (horizontal). The tangent is perpendicular to the radius, so the small triangle is similar to the big one (legs r and x, hypotenuse 1), rotated by 90°: dx corresponds to r, |dr| to x, ds to 1. Hence ds/dx = 1/r. Shrink dx with the slider: the measured ratio (chord/dx) converges to 1/r.

General d (the equation): the slice at height x is a (d−2)-sphere of radius r, with area ∝ r^{d−2}; the tilt factor is still 1/r; divide by the total area. That gives TurboQuant's Lemma 1: f_X(x) ∝ (1 − x²)^{(d−3)/2}, with the Gamma-function constant making it integrate to 1. At d = 3 the exponent is 0: flat. For d > 3 the slice factor wins and the mass piles up at the equator, which is where we go next.`,
  mount(body, ctx) {
    const rng = new M.RNG(303);
    const pts = [];
    for (let i = 0; i < N; i++) pts.push(M.randomUnit(3, rng));
    const state = { x: 0.55, dx: 0.12 };
    const band = () => [Math.max(-1, state.x - state.dx / 2), Math.min(1, state.x + state.dx / 2)];
    const hist = M.histogram(
      pts.map((p) => p[2]),
      -1,
      1,
      BINS,
    );
    const inBand = (z, [z1, z2]) => z >= z1 && z <= z2;

    // ---------- 3-D sphere ----------
    const vSphere = viz('', 'drag to orbit');
    let cam;
    const cvSphere = createCanvas(vSphere, (g, w, h) => {
      if (w < 60 || h < 60) return;
      if (!cam) return;
      cam.viewport(w / 2, h / 2 + 8, Math.min(w, h) * 0.37);
      const [z1, z2] = band();
      cam.sphere(g, { rings: 6, a: 0.1 });
      cam.path(
        g,
        [
          [0, 0, -1.3],
          [0, 0, 1.3],
        ],
        { color: 'ink-3', dash: [4, 4], alpha: 0.7 },
      );
      const top = cam.project(0, 0, 1.38);
      label(g, 'xⱼ', top.x + 8, top.y, { color: 'ink-2', size: 15 });
      // band ribbon
      const n = 72;
      const r1 = Math.sqrt(1 - z1 * z1);
      const r2 = Math.sqrt(1 - z2 * z2);
      g.fillStyle = alpha('yellow', 0.13);
      for (let i = 0; i < n; i++) {
        const a0 = (i / n) * 2 * Math.PI;
        const a1 = ((i + 1) / n) * 2 * Math.PI;
        const q = [
          cam.project(r1 * Math.cos(a0), r1 * Math.sin(a0), z1),
          cam.project(r1 * Math.cos(a1), r1 * Math.sin(a1), z1),
          cam.project(r2 * Math.cos(a1), r2 * Math.sin(a1), z2),
          cam.project(r2 * Math.cos(a0), r2 * Math.sin(a0), z2),
        ];
        g.beginPath();
        g.moveTo(q[0].x, q[0].y);
        for (let k = 1; k < 4; k++) g.lineTo(q[k].x, q[k].y);
        g.closePath();
        g.fill();
      }
      cam.path(g, cam.latitude(z1), { color: 'yellow', alpha: 0.85, width: 1.4 });
      cam.path(g, cam.latitude(z2), { color: 'yellow', alpha: 0.85, width: 1.4 });
      const proj = pts.map((p) => ({ s: cam.project(p[0], p[1], p[2]), hot: inBand(p[2], [z1, z2]) }));
      proj.sort((a, b) => a.s.z - b.s.z);
      for (const { s, hot } of proj) {
        const depth = 0.3 + 0.7 * ((s.z + 1) / 2);
        g.fillStyle = hot ? alpha('yellow', 0.55 + 0.45 * depth) : alpha('accent', 0.12 + 0.55 * depth);
        g.beginPath();
        g.arc(s.x, s.y, (hot ? 2.8 : 2) * s.s, 0, Math.PI * 2);
        g.fill();
      }
      text(g, `${N} uniform points on 𝕊²`, w / 2, h - 14, { color: 'ink-3', size: 12, align: 'center' });
    });
    cam = new Orbit3D(cvSphere.canvas, { yaw: 0.5, pitch: 0.32, spin: 0.18 });

    // ---------- histogram (sideways: height up, density right) ----------
    const vHist = viz();
    let histGeom = null;
    const cvHist = createCanvas(vHist, (g, w, h) => {
      if (w < 60 || h < 60) return;
      const top = 34;
      const bot = h - 40;
      const left = 34;
      const right = w - 14;
      const dmax = 0.8;
      const Y = (z) => bot - ((z + 1) / 2) * (bot - top);
      const X = (f) => left + (f / dmax) * (right - left);
      histGeom = { Y, top, bot };
      text(g, 'histogram of xⱼ', 14, 16, { color: 'ink-2', size: 12.5 });
      // axes
      line(g, left, top, left, bot, { color: 'line-2' });
      line(g, left, bot, right, bot, { color: 'line-2' });
      for (const z of [-1, -0.5, 0, 0.5, 1]) text(g, String(z), left - 6, Y(z), { color: 'ink-3', size: 11, align: 'right', font: 'mono' });
      for (const f of [0, 0.5]) text(g, String(f), X(f), bot + 12, { color: 'ink-3', size: 11, align: 'center', font: 'mono' });
      text(g, 'density', (left + right) / 2, bot + 28, { color: 'ink-3', size: 11, align: 'center' });
      // bars
      const bh = (bot - top) / BINS;
      g.fillStyle = alpha('accent', 0.4);
      for (let i = 0; i < BINS; i++) g.fillRect(left, bot - (i + 1) * bh + 1, X(hist[i]) - left, bh - 2);
      // band bar (measured)
      const [z1, z2] = band();
      let cnt = 0;
      for (const p of pts) if (inBand(p[2], [z1, z2])) cnt++;
      const dens = cnt / N / (z2 - z1);
      g.fillStyle = alpha('yellow', 0.85);
      g.fillRect(left, Y(z2), X(dens) - left, Math.max(2, Y(z1) - Y(z2)));
      // theory: flat 1/2
      line(g, X(0.5), top - 6, X(0.5), bot, { color: 'teal', width: 2, dash: [6, 5] });
      label(g, '½', X(0.5) + 5, top - 4, { color: 'teal', size: 14 });
      label(g, dens.toFixed(2), Math.min(right - 30, X(dens) + 6), (Y(z1) + Y(z2)) / 2, { color: 'yellow', size: 12, font: 'mono' });
    });

    // ---------- cross-section with the similar triangles ----------
    const vCut = viz();
    let cutGeom = null;
    const rr = (z) => Math.sqrt(Math.max(0, 1 - z * z));
    const cvCut = createCanvas(vCut, (g, w, h) => {
      if (w < 60 || h < 60) return;
      const zoomW = Math.min(w * 0.4, h * 0.6);
      const R = Math.min((w - zoomW - 70) * 0.44, (h - 80) / 2);
      const cx = 22 + R;
      const cy = h / 2 + 4;
      const P = (r, z) => [cx + r * R, cy - z * R];
      cutGeom = { cy, R };
      text(g, 'the sphere cut through the xⱼ axis', 14, 16, { color: 'ink-2', size: 12.5 });
      const [z1, z2] = band();
      const x = state.x;
      const r = rr(x);
      // band strip
      g.fillStyle = alpha('yellow', 0.1);
      g.fillRect(cx - R - 10, cy - z2 * R, 2 * R + 20, (z2 - z1) * R);
      line(g, cx - R - 10, cy - z2 * R, cx + R + 10, cy - z2 * R, { color: 'yellow', alpha: 0.4 });
      line(g, cx - R - 10, cy - z1 * R, cx + R + 10, cy - z1 * R, { color: 'yellow', alpha: 0.4 });
      // axes and circle
      line(g, cx, cy - R - 16, cx, cy + R + 10, { color: 'ink-3', alpha: 0.6 });
      line(g, cx - R - 10, cy, cx + R + 16, cy, { color: 'ink-3', alpha: 0.6 });
      label(g, 'xⱼ', cx + 6, cy - R - 14, { color: 'ink-2', size: 14 });
      label(g, 'r', cx + R + 10, cy - 10, { color: 'ink-2', size: 14 });
      circle(g, cx, cy, R, { stroke: 'line-2', width: 1.5 });
      // band arcs (right is the r ≥ 0 half we care about)
      const a1 = Math.asin(z1);
      const a2 = Math.asin(z2);
      g.save();
      g.lineCap = 'round';
      g.strokeStyle = color('yellow');
      g.lineWidth = 5;
      g.beginPath();
      g.arc(cx, cy, R, -a2, -a1);
      g.stroke();
      g.strokeStyle = alpha('yellow', 0.35);
      g.lineWidth = 3;
      g.beginPath();
      g.arc(cx, cy, R, Math.PI + a1, Math.PI + a2);
      g.stroke();
      g.restore();
      // big triangle O, H = (0, x), Pt = (r, x)
      const [ox, oy] = P(0, 0);
      const [hx, hy] = P(0, x);
      const [px, py] = P(r, x);
      g.fillStyle = alpha('blue', 0.12);
      g.beginPath();
      g.moveTo(ox, oy);
      g.lineTo(hx, hy);
      g.lineTo(px, py);
      g.closePath();
      g.fill();
      line(g, ox, oy, px, py, { color: 'blue', width: 2 });
      line(g, hx, hy, px, py, { color: 'blue', width: 2 });
      line(g, ox, oy, hx, hy, { color: 'blue', width: 2 });
      label(g, 'r', (hx + px) / 2, hy + (x >= 0 ? -11 : 12), { color: 'blue', size: 14, align: 'center' });
      label(g, 'x', hx - 12, (oy + hy) / 2, { color: 'blue', size: 14, align: 'center' });
      label(g, '1', (ox + px) / 2 + 9, (oy + py) / 2 + (x >= 0 ? 10 : -10), { color: 'blue', size: 14, align: 'center' });

      // magnifier
      const r1 = rr(z1);
      const r2 = rr(z2);
      const chord = Math.hypot(r1 - r2, z2 - z1);
      const mx = (r1 + r2) / 2;
      const mz = (z1 + z2) / 2;
      const half = Math.max(chord * 0.75, 9 / R);
      const [mpx, mpy] = P(mx, mz);
      g.save();
      g.strokeStyle = alpha('ink', 0.7);
      g.lineWidth = 1;
      g.strokeRect(mpx - half * R, mpy - half * R, 2 * half * R, 2 * half * R);
      g.restore();
      const bx = w - zoomW - 14;
      const by = (h - zoomW) / 2 + 6;
      line(g, mpx + half * R, mpy - half * R, bx, by, { color: 'ink-3', alpha: 0.4 });
      line(g, mpx + half * R, mpy + half * R, bx, by + zoomW, { color: 'ink-3', alpha: 0.4 });
      g.save();
      g.fillStyle = 'rgba(8,14,28,0.9)';
      g.fillRect(bx, by, zoomW, zoomW);
      g.strokeStyle = color('line-2');
      g.strokeRect(bx, by, zoomW, zoomW);
      g.beginPath();
      g.rect(bx, by, zoomW, zoomW);
      g.clip();
      const Zs = (0.55 * zoomW) / Math.max(chord, 1e-6);
      const Q = (r0, z0) => [bx + zoomW / 2 + (r0 - mx) * Zs, by + zoomW / 2 - (z0 - mz) * Zs];
      // band strip in zoom
      g.fillStyle = alpha('yellow', 0.08);
      const [, qz2] = Q(0, z2);
      const [, qz1] = Q(0, z1);
      g.fillRect(bx, qz2, zoomW, qz1 - qz2);
      // circle arc in zoom
      const phiC = Math.atan2(mz, mx);
      const span = (zoomW / Zs) * 1.2;
      g.strokeStyle = color('line-2');
      g.lineWidth = 1.5;
      g.beginPath();
      for (let i = 0; i <= 80; i++) {
        const ph = phiC - span + (2 * span * i) / 80;
        const [a, b] = Q(Math.cos(ph), Math.sin(ph));
        if (i === 0) g.moveTo(a, b);
        else g.lineTo(a, b);
      }
      g.stroke();
      // radius direction through the midpoint (toward the center)
      const [m0x, m0y] = Q(mx, mz);
      const ux = -mx;
      const uz = -mz;
      const ul = Math.hypot(ux, uz);
      const far = Q(mx + ((ux / ul) * (0.45 * zoomW)) / Zs, mz + ((uz / ul) * (0.45 * zoomW)) / Zs);
      line(g, m0x, m0y, far[0], far[1], { color: 'blue', dash: [5, 4], width: 1.5 });
      // small triangle: P1 = (r1, z1), P2 = (r2, z2), C = (r2, z1)
      const A = Q(r1, z1);
      const B = Q(r2, z2);
      const C = Q(r2, z1);
      g.fillStyle = alpha('yellow', 0.12);
      g.beginPath();
      g.moveTo(A[0], A[1]);
      g.lineTo(B[0], B[1]);
      g.lineTo(C[0], C[1]);
      g.closePath();
      g.fill();
      line(g, C[0], C[1], B[0], B[1], { color: 'accent', width: 2.5 });
      line(g, A[0], A[1], C[0], C[1], { color: 'ink-2', width: 2 });
      line(g, A[0], A[1], B[0], B[1], { color: 'yellow', width: 3 });
      // right-angle marker at C
      const sgn = Math.sign(A[0] - C[0]) || 1;
      const sz = 9;
      g.save();
      g.strokeStyle = alpha('ink', 0.6);
      g.beginPath();
      g.moveTo(C[0] + sgn * sz, C[1]);
      g.lineTo(C[0] + sgn * sz, C[1] - sz);
      g.lineTo(C[0], C[1] - sz);
      g.stroke();
      g.restore();
      g.restore();
      label(g, 'dx', C[0] - sgn * 12, (C[1] + B[1]) / 2, { color: 'accent', size: 14, align: sgn > 0 ? 'right' : 'left', weight: 600 });
      label(g, '|dr|', (A[0] + C[0]) / 2, C[1] + 14, { color: 'ink-2', size: 14, align: 'center', weight: 600 });
      label(g, 'ds', (A[0] + B[0]) / 2 + sgn * 12, (A[1] + B[1]) / 2 - 8, { color: 'yellow', size: 14, align: sgn > 0 ? 'left' : 'right', weight: 600 });
      text(g, 'zoom', bx + 8, by + 12, { color: 'ink-3', size: 11 });
      const meas = chord / (z2 - z1);
      text(g, `ds/dx = ${meas.toFixed(4)}`, bx + zoomW / 2, by + zoomW + 16, { color: 'yellow', size: 13, align: 'center', font: 'mono' });
      text(g, `1/r = ${(1 / r).toFixed(4)}`, bx + zoomW / 2, by + zoomW + 34, { color: 'blue', size: 13, align: 'center', font: 'mono' });
      text(g, '(dx, |dr|, ds) ∼ (r, x, 1) turned 90°', bx + zoomW / 2, by - 12, { color: 'ink-2', size: 12, align: 'center' });
    });

    // ---------- controls ----------
    const rR = readout('$r=\\sqrt{1-x^2}$');
    const rSlice = readout('slice $2\\pi r$');
    const rTilt = readout('tilt $1/r$');
    const rProd = readout('slice × tilt / $4\\pi$');
    const rMeas = readout('band mass $/\\,dx$');
    const update = () => {
      const r = rr(state.x);
      rR.set(r.toFixed(3));
      rSlice.set((2 * Math.PI * r).toFixed(3), 'accent');
      rTilt.set((1 / r).toFixed(3), 'blue');
      rProd.set('0.500', 'teal');
      const [z1, z2] = band();
      let cnt = 0;
      for (const p of pts) if (inBand(p[2], [z1, z2])) cnt++;
      rMeas.set((cnt / N / (z2 - z1)).toFixed(3), 'yellow');
      cvSphere.request();
      cvHist.request();
      cvCut.request();
    };
    const sX = slider({
      label: 'height $x$',
      min: -0.9,
      max: 0.9,
      step: 0.01,
      value: state.x,
      format: (v) => v.toFixed(2),
      onInput: (v) => ((state.x = v), update()),
    });
    const sDx = slider({
      label: 'width $dx$',
      min: 0.01,
      max: 0.5,
      step: 0.005,
      value: state.dx,
      format: (v) => v.toFixed(3),
      onInput: (v) => ((state.dx = v), update()),
    });

    // Drag the band vertically in the histogram or cross-section.
    const dragZ = (cv, geomZ) =>
      draggable(cv.canvas, {
        pick: () => true,
        move: (_, px, py) => {
          const z = geomZ(py);
          if (z == null) return;
          state.x = M.clamp(Math.round(z * 100) / 100, -0.9, 0.9);
          sX.set(state.x, false);
          update();
        },
      });
    dragZ(cvHist, (py) => (histGeom ? -1 + (2 * (histGeom.bot - py)) / (histGeom.bot - histGeom.top) : null));
    dragZ(cvCut, (py) => (cutGeom ? (cutGeom.cy - py) / cutGeom.R : null));

    const top = responsiveGrid(el('div', { style: { gap: '14px' } }, vSphere, vHist, vCut), 'minmax(0,1.05fr) minmax(0,0.42fr) minmax(0,1.35fr)', {
      vizzes: [vSphere, vHist, vCut],
      fill: true,
    });
    const bottom = el(
      'div',
      { style: { flex: '0 0 auto', gap: '14px', alignItems: 'stretch' } },
      el('div.panel', { style: { justifyContent: 'center', gap: '6px', padding: '12px 18px' } }, sX.el, sDx.el),
      el('div.readouts', { style: { gridTemplateColumns: 'repeat(3, minmax(0,1fr))', gap: '8px' } }, rR.el, rSlice.el, rTilt.el, rProd.el, rMeas.el),
      el(
        'div.card',
        { 'data-reveal': '1', style: { padding: '10px 16px', gap: '2px', justifyContent: 'center' } },
        eq(
          'f_X(x)=\\underbrace{\\tfrac{|\\S^{d-2}|\\,r^{d-2}}{|\\S^{d-1}|}}_{\\text{slice}}\\cdot\\underbrace{\\tfrac{1}{r}}_{\\text{tilt}}=\\frac{\\Gamma(d/2)}{\\sqrt\\pi\\,\\Gamma(\\frac{d-1}{2})}(1-x^2)^{\\frac{d-3}{2}}',
        ),
        rich('div.note', '$r=\\sqrt{1-x^2}$. At $d=3$: $\\;\\frac{2\\pi r}{4\\pi}\\cdot\\frac1r=\\frac12$, the factors cancel.', {
          style: { textAlign: 'center' },
        }),
      ),
    );
    responsiveGrid(bottom, 'minmax(0,0.8fr) minmax(0,1.1fr) minmax(0,1.5fr)');
    body.style.gap = '14px';
    body.append(top, bottom);

    update();
    ctx.loop((dt) => {
      cam.tick(dt);
      cvSphere.redraw();
    });
  },
};

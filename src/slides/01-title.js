import { el, texify } from '../lib/ui.js';
import { createCanvas, alpha } from '../lib/canvas.js';
import { Orbit3D } from '../lib/orbit3d.js';
import { RNG, randomUnit, randomRotation, matVec, matTVec, sphereCodebook, quantizeWith, smoothstep } from '../lib/math.js';

export default {
  id: 'title',
  chapter: 'Why',
  bare: true,
  className: 'statement',
  title: 'TurboQuant, interactively',
  notes: `Welcome. Two papers by roughly the same group: QJL (2024) and TurboQuant (2025).
The goal is not to present them; it's to rediscover them. Each step will be forced by a failure of the step before.
Background animation: random points on a sphere snapping to TurboQuant's 2-bit grid (in a random rotated frame) and back.
Keys: ↓/space next, ↑ back, N notes, F fullscreen, P dims the chrome.`,
  mount(body, ctx) {
    body.style.position = 'relative';
    body.style.justifyContent = 'center';

    const bg = el('div', { style: { position: 'absolute', inset: '0', right: '0' } });
    body.append(bg);

    const rng = new RNG(7);
    const N = 650;
    const Pi = randomRotation(3, rng);
    const { centroids } = sphereCodebook(3, 2);
    const pts = [];
    for (let i = 0; i < N; i++) {
      const x = randomUnit(3, rng);
      const q = matTVec(Pi, quantizeWith(matVec(Pi, x), centroids));
      pts.push({ x, q });
    }
    // Grid points of the rotated codebook (4×4×4).
    const grid = [];
    for (const a of centroids) for (const b of centroids) for (const c of centroids) grid.push(matTVec(Pi, Float64Array.of(a, b, c)));

    let cam;
    let blend = 0;
    const view = createCanvas(bg, (g, w, h) => {
      if (!cam) return;
      const k = Math.min(w, h) * (w > 900 ? 0.32 : 0.3);
      cam.viewport(w > 900 ? w * 0.74 : w * 0.5, h * 0.5, k);
      // grid
      for (const p of grid) {
        const s = cam.project(p[0], p[1], p[2]);
        g.fillStyle = alpha('teal', 0.12 + 0.35 * blend);
        g.beginPath();
        g.arc(s.x, s.y, 2.2 * s.s, 0, Math.PI * 2);
        g.fill();
      }
      cam.sphere(g, { rings: 8, a: 0.08 });
      const proj = pts.map((p) => {
        const v = [0, 1, 2].map((i) => p.x[i] + (p.q[i] - p.x[i]) * blend);
        return cam.project(v[0], v[1], v[2]);
      });
      proj.sort((a, b) => a.z - b.z);
      for (const s of proj) {
        const depth = 0.35 + 0.65 * ((s.z + 1) / 2);
        g.fillStyle = blend > 0.5 ? alpha('teal', depth) : alpha('accent', depth);
        g.beginPath();
        g.arc(s.x, s.y, 2.4 * s.s, 0, Math.PI * 2);
        g.fill();
      }
    });
    cam = new Orbit3D(view.canvas, { yaw: 0.4, pitch: 0.38, spin: 0.12 });

    ctx.loop((dt, t) => {
      cam.tick(dt);
      // hold on sphere, snap to grid, hold, release
      const phase = (t % 8) / 8;
      const target = phase < 0.4 ? 0 : phase < 0.5 ? smoothstep((phase - 0.4) / 0.1) : phase < 0.9 ? 1 : 1 - smoothstep((phase - 0.9) / 0.1);
      blend = target;
      view.redraw();
    });

    const text = el(
      'div',
      { style: { position: 'relative', maxWidth: '640px', display: 'flex', flexDirection: 'column', gap: '18px', pointerEvents: 'none' } },
      el('h1', {
        style: { fontFamily: 'var(--serif)', fontSize: 'clamp(48px, 7.5vw, 140px)', lineHeight: '0.95', margin: '0', fontWeight: '600' },
        html: 'Turbo<span style="color:var(--accent)">Quant</span>',
      }),
      el(
        'div',
        { style: { display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '18px', fontSize: '0.95em', color: 'var(--ink-3)' } },
        el('div', { html: '<b style="color:var(--ink-2)">QJL</b> · Zandieh, Daliri, Han · arXiv 2406.03482' }),
        el('div', { html: '<b style="color:var(--ink-2)">TurboQuant</b> · Zandieh, Daliri, Hadian, Mirrokni · arXiv 2504.19874' }),
      ),
      el('div', {
        style: { marginTop: '28px', fontSize: '0.85em', color: 'var(--ink-3)', letterSpacing: '0.08em' },
        html: texify('scroll or press ↓ &nbsp;·&nbsp; N notes &nbsp;·&nbsp; F fullscreen &nbsp;·&nbsp; drag the sphere'),
      }),
    );
    body.append(text);
    // Let the sphere receive drags everywhere except over the text block.
    bg.style.zIndex = '0';
    text.style.zIndex = '1';
  },
};

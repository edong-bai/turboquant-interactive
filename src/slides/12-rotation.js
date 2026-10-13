import { el, rich, eq, segmented, button, readout, legend, viz, texify } from '../lib/ui.js';
import { createCanvas, View2D, arrow, circle, dot, line, label, text, alpha, color } from '../lib/canvas.js';
import { Plot } from '../lib/plot.js';
import * as M from '../lib/math.js';

const D = 64;
const POOL = 64; // random rotations used for the "range over Π" whiskers

/** Four fixed unit inputs in R^64 (seeded, so the talk is reproducible). */
function makePresets() {
  const rng = new M.RNG(6401);
  const spike = M.zeros(D);
  spike[9] = 1;
  const spread = new Float64Array(D);
  for (let i = 0; i < D; i++) spread[i] = (rng.next() < 0.5 ? -1 : 1) / Math.sqrt(D);
  const out = M.randn(D, rng);
  out[23] = 12; // one large channel (illustrative magnitude) on top of N(0,1) noise
  const gauss = M.randomUnit(D, rng);
  return { spike, spread, outlier: M.normalize(out), gauss };
}

const KEYS = ['spike', 'spread', 'outlier', 'gauss'];
const NAMES = { spike: 'spike', spread: 'spread', outlier: 'outlier', gauss: 'Gaussian' };

/**
 * The naive quantizer from the earlier slides: 2^b − 1 evenly spaced levels on [−s, s]
 * (including 0), with s = 1 ("normalize, then round") or s = max_i |v_i| (per-vector absmax).
 */
function naiveQuant(v, b, mode) {
  let s = 1;
  if (mode === 'max') {
    s = 0;
    for (const t of v) s = Math.max(s, Math.abs(t));
    s = s || 1;
  }
  const K = 2 ** (b - 1) - 1;
  const step = s / K;
  const q = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) q[i] = M.clamp(Math.round(v[i] / step), -K, K) * step;
  return { q, s, step, K };
}

function err2(v, q) {
  let e = 0;
  for (let i = 0; i < v.length; i++) e += (v[i] - q[i]) ** 2;
  return e;
}

export default {
  id: 'rotation',
  chapter: 'Rotation',
  kicker: 'Back to the failure',
  title: 'What if no input could be adversarial?',
  sub: '<span class="c-teal">Rotate first: the worst case becomes the average case.</span>',
  notes: `Open with the question parked on the naive-fails slide: what if no input could be adversarial? The previous slide gave the answer in QJL's setting: a random projection makes every coordinate N(0, ‖k‖²) whatever k was, so the sign's lost magnitude is known, errors are random, and energy is spread evenly. Here we want the same thing but we must also RECONSTRUCT x (values need MSE), so instead of a Gaussian projection we use a random ROTATION: it preserves lengths and is invertible.

Recall the two failures of "normalize then round": a spread vector (every coordinate ±1/√d) falls between the grid levels and rounds to 0; an outlier channel sets the scale and crushes everything else. Both are adversarial INPUTS. We can't choose the input. But we can choose a random rotation Π, once, and quantize Πx instead of x.

What to do:
• Pick a preset (spike, spread, outlier + noise, Gaussian). The orange bars are the 64 coordinates of the unit vector x; dashed lines are the quantizer's levels; teal ticks are where each coordinate rounds to; violet is the per-coordinate error.
• Press "apply Π": the bars morph into Πx. Whatever you started with, Πx looks the same: 64 small coordinates of size about ±1/√d = ±0.125, Gaussian-looking. Re-roll Π: a different but statistically identical picture.
• Bottom left: the SAME naive quantizer's squared error ‖x − x̃‖² for each preset, before (red) and after (teal) rotation. The teal whisker is the range over ${POOL} independent random Π. After rotation, the error no longer depends on which input you fed in. Worst case = average case.
• Switch the range between "fixed [−1,1]" and "per-vector max". With the fixed range, rotation makes everything uniformly bad (all coordinates are ~0.125, smaller than half a step, so most round to 0). That's not a bug in the argument: rotation only makes the input distribution KNOWN. The next slides find what that distribution is and design the codebook for it.

Why it works (equations): for every fixed unit x, Πx is uniform on the sphere, because a Haar-random rotation maps any fixed point to a uniform point. Rotation preserves lengths, so ‖Πx − ỹ‖ = ‖x − Πᵀỹ‖: quantizing Πx with a fixed grid is exactly the same as quantizing x with a randomly rotated grid (inset, bottom right: left half rotates the input, right half rotates the grid; the violet errors have the same length).

"Randomize the algorithm ⇔ randomize the input": this is Yao-flavored intuition. A randomized algorithm on a worst-case input behaves like a deterministic algorithm on a random input. Yao's minimax principle is used formally later, on the 'bounds' slide, for the lower bound.

Practicalities (paper: "Generate Π by QR decomposition on a random matrix with i.i.d. Normal entries"): Π is drawn once and shared by every vector, so storing it is a one-time cost, and applying it is one d×d matrix–vector product per vector. A consequence of the math (not something the paper spells out): for attention, ⟨q, k⟩ = ⟨Πq, Πk⟩, so we can rotate the query once instead of de-rotating every key.

The presets are synthetic (the outlier channel's size, 12× the noise level, is illustrative). Naive quantizer used here: 2^b − 1 evenly spaced levels on [−s, s] including 0.`,
  mount(body, ctx) {
    const presets = makePresets();
    const rng = new M.RNG(1212);
    const state = { preset: 'outlier', b: 3, mode: 'max', rotated: false, Pi: null };
    // morph: displayed vector = normalize((1−s)·from + s·to)
    const morph = { from: presets.outlier, to: presets.outlier, p: 1 };
    const pool = []; // pool[j][key] = Π_j x_key
    let rotX = null; // rotX[key] = Π x_key for the current Π
    let theta = 0.9; // 2-D inset angle

    const current = () => {
      const s = M.smoothstep(M.clamp(morph.p, 0, 1));
      const v = new Float64Array(D);
      for (let i = 0; i < D; i++) v[i] = (1 - s) * morph.from[i] + s * morph.to[i];
      return M.normalize(v);
    };
    const target = () => (state.rotated && rotX ? rotX[state.preset] : presets[state.preset]);
    const goTo = (to) => {
      morph.from = current();
      morph.to = to;
      morph.p = 0;
    };
    const ensurePi = () => {
      if (state.Pi) return;
      state.Pi = M.randomRotation(D, rng);
      rotX = Object.fromEntries(KEYS.map((k) => [k, M.matVec(state.Pi, presets[k])]));
    };

    // ---------- main bar chart ----------
    const rBefore = readout('error before: $\\|x-\\tilde x\\|^2$');
    const rAfter = readout('after: $\\|\\Pi x-\\tilde y\\|^2$');
    let yScale = 1.1;
    const vMain = viz();
    vMain.style.height = 'auto';
    vMain.style.flex = '1 1 0';
    vMain.style.minHeight = '240px';
    const cvMain = createCanvas(vMain, (g, w, h) => {
      if (w < 60 || h < 60) return;
      const v = current();
      const { q, s: scale, step, K } = naiveQuant(v, state.b, state.mode);
      let vmax = 0;
      for (const t of v) vmax = Math.max(vmax, Math.abs(t));
      const want = Math.max(0.2, vmax * 1.18);
      yScale = want; // follows the morph (it is already smooth)
      const p = new Plot(g, {
        x: 0,
        y: 26,
        w,
        h: h - 26,
        xmin: -0.5,
        xmax: D - 0.5,
        ymin: -yScale,
        ymax: yScale,
        xticks: [0, 16, 32, 48, 63],
        xfmt: (t) => `${t + 1}`,
        yticks: 5,
        margin: { l: 56, r: 16, t: 12, b: 34 },
      });
      p.frame({ grid: false });
      // quantizer levels
      p.clip(() => {
        for (let k = -K; k <= K; k++) {
          const y = p.ty(k * step);
          line(g, p.px, y, p.px + p.pw, y, { color: k === 0 ? 'ink-3' : 'ink-3', alpha: k === 0 ? 0.6 : 0.35, dash: k === 0 ? null : [4, 5] });
        }
      });
      // ±1/√d guide
      const u = 1 / Math.sqrt(D);
      p.clip(() => {
        for (const sg of [-1, 1]) line(g, p.px, p.ty(sg * u), p.px + p.pw, p.ty(sg * u), { color: 'blue', alpha: 0.35, dash: [2, 3] });
      });
      p.bars(v, -0.5, D - 0.5, { color: 'accent', alpha: 0.8, gap: 2 });
      // quantized values and errors
      p.clip(() => {
        const bw = p.pw / D;
        for (let i = 0; i < D; i++) {
          const cx = p.tx(i);
          if (Math.abs(v[i] - q[i]) > 1e-9) line(g, cx, p.ty(v[i]), cx, p.ty(q[i]), { color: 'violet', width: Math.max(1.5, bw * 0.28) });
          line(g, cx - bw * 0.45, p.ty(q[i]), cx + bw * 0.45, p.ty(q[i]), { color: 'teal', width: 2.5 });
        }
      });
      const e = err2(v, q);
      const lab = morph.p >= 1 ? (state.rotated ? 'Πx' : 'x') : '…';
      label(g, lab, 16, 17, { color: 'accent', size: 18, weight: 600 });
      text(g, `d = ${D} coordinates · range ±${scale.toFixed(3)} · ${2 * K + 1} levels · ‖v − Q(v)‖² = ${e.toFixed(3)}`, 52, 17, {
        color: 'ink-2',
        size: 13,
        font: 'mono',
      });
      text(g, '±1/√d', p.px + p.pw - 4, p.ty(u) - 9, { color: 'blue', size: 12, align: 'right', alpha: 0.8 });
    });

    // ---------- per-preset error bars ----------
    const vErr = viz();
    vErr.style.height = '100%';
    vErr.style.minHeight = '0';
    const cvErr = createCanvas(vErr, (g, w, h) => {
      if (w < 60 || h < 60) return;
      const before = KEYS.map((k) => err2(presets[k], naiveQuant(presets[k], state.b, state.mode).q));
      const after = rotX ? KEYS.map((k) => err2(rotX[k], naiveQuant(rotX[k], state.b, state.mode).q)) : null;
      const ranges = KEYS.map((k) => {
        if (!pool.length) return null;
        let lo = Infinity;
        let hi = -Infinity;
        for (const P of pool) {
          const e = err2(P[k], naiveQuant(P[k], state.b, state.mode).q);
          lo = Math.min(lo, e);
          hi = Math.max(hi, e);
        }
        return [lo, hi];
      });
      let top = Math.max(...before, ...(after || [0]), ...ranges.map((r) => (r ? r[1] : 0)));
      top = Math.max(0.05, top * 1.15);
      const p = new Plot(g, { x: 0, y: 24, w, h: h - 24, xmin: 0, xmax: 4, ymin: 0, ymax: top, xticks: [], yticks: 4, margin: { l: 50, r: 10, t: 8, b: 26 } });
      p.frame();
      const gw = p.pw / 4;
      KEYS.forEach((k, i) => {
        const x0 = p.px + i * gw;
        if (k === state.preset) {
          g.fillStyle = alpha('yellow', 0.08);
          g.fillRect(x0 + 2, p.py, gw - 4, p.ph);
        }
        const bw = gw * 0.26;
        const xb = x0 + gw * 0.2;
        const xa = x0 + gw * 0.54;
        g.fillStyle = alpha('red', 0.8);
        g.fillRect(xb, p.ty(before[i]), bw, p.ty(0) - p.ty(before[i]));
        if (after) {
          g.fillStyle = alpha('teal', 0.85);
          g.fillRect(xa, p.ty(after[i]), bw, p.ty(0) - p.ty(after[i]));
        }
        if (ranges[i]) {
          const cx = xa + bw / 2;
          line(g, cx, p.ty(ranges[i][0]), cx, p.ty(ranges[i][1]), { color: 'ink', width: 1.5 });
          line(g, cx - 5, p.ty(ranges[i][0]), cx + 5, p.ty(ranges[i][0]), { color: 'ink', width: 1.5 });
          line(g, cx - 5, p.ty(ranges[i][1]), cx + 5, p.ty(ranges[i][1]), { color: 'ink', width: 1.5 });
        }
        text(g, NAMES[k], x0 + gw / 2, p.py + p.ph + 13, { color: k === state.preset ? 'yellow' : 'ink-2', size: 12, align: 'center' });
      });
      text(g, 'naive ‖x − x̃‖² per input', 14, 15, { color: 'ink-2', size: 12.5 });
      text(g, 'before', w - 168, 15, { color: 'red', size: 12 });
      text(g, 'after Π (range: 64 Π)', w - 10, 15, { color: 'teal', size: 12, align: 'right' });
      if (pool.length && pool.length < POOL) text(g, `sampling Π… ${pool.length}/${POOL}`, p.px + 6, p.py + 8, { color: 'ink-3', size: 11, font: 'mono' });
    });

    // ---------- 2-D inset: rotate the input ⇔ rotate the grid ----------
    const vIn = viz('', '');
    vIn.style.height = '100%';
    vIn.style.minHeight = '0';
    const lv2 = M.uniformLevels(2, -0.75, 0.75);
    const x2 = Float64Array.of(Math.cos(0.38), Math.sin(0.38));
    const cvIn = createCanvas(vIn, (g, w, h) => {
      if (w < 60 || h < 60) return;
      const R = M.rot2(theta);
      const halves = [
        { x: 0, title: 'rotate input: Πx, grid fixed' },
        { x: w / 2, title: 'rotate grid: x, Πᵀ·grid' },
      ];
      const y = M.matVec(R, x2);
      const yq = M.quantizeWith(y, lv2);
      const xq = M.matTVec(R, yq);
      halves.forEach((hv, idx) => {
        const view = new View2D({ x: hv.x, y: 22, w: w / 2, h: h - 22, xmin: -1.15, xmax: 1.15, ymin: -1.15, ymax: 1.15, pad: 10 });
        circle(g, view.sx(0), view.sy(0), view.len(1), { stroke: 'line-2', width: 1 });
        for (const a of lv2)
          for (const b of lv2) {
            const pt = idx === 0 ? [a, b] : M.matTVec(R, Float64Array.of(a, b));
            dot(g, view.sx(pt[0]), view.sy(pt[1]), 2.6, 'ink-3');
          }
        const v = idx === 0 ? y : x2;
        const r = idx === 0 ? yq : xq;
        arrow(g, view.sx(0), view.sy(0), view.sx(v[0]), view.sy(v[1]), { color: 'accent', width: 2.2, head: 8 });
        dot(g, view.sx(r[0]), view.sy(r[1]), 4.5, 'teal');
        line(g, view.sx(r[0]), view.sy(r[1]), view.sx(v[0]), view.sy(v[1]), { color: 'violet', width: 2.5 });
        text(g, hv.title, hv.x + w / 4, 13, { color: 'ink-2', size: 11.5, align: 'center' });
      });
      line(g, w / 2, 26, w / 2, h - 8, { color: 'line', width: 1 });
      const e = Math.hypot(y[0] - yq[0], y[1] - yq[1]);
      text(g, `|err| = ${e.toFixed(3)}`, w / 4, h - 10, { color: 'violet', size: 12, align: 'center', font: 'mono' });
      text(g, `|err| = ${e.toFixed(3)}`, (3 * w) / 4, h - 10, { color: 'violet', size: 12, align: 'center', font: 'mono' });
    });

    // ---------- controls ----------
    const refresh = () => {
      cvErr.request();
      cvMain.request();
    };
    const sPreset = segmented({
      options: KEYS.map((k) => ({
        label: { spike: 'one spike', spread: 'spread $\\pm1/\\sqrt d$', outlier: 'outlier + noise', gauss: 'Gaussian' }[k],
        value: k,
      })),
      value: state.preset,
      onChange: (k) => {
        state.preset = k;
        goTo(target());
        refresh();
      },
    });
    const sBits = segmented({
      label: 'bits $b$',
      options: [2, 3, 4],
      value: state.b,
      onChange: (b) => {
        state.b = b;
        refresh();
      },
    });
    const sMode = segmented({
      label: 'range',
      options: [
        { label: 'fixed $[-1,1]$', value: 'fixed' },
        { label: 'per-vector max', value: 'max' },
      ],
      value: state.mode,
      onChange: (m) => {
        state.mode = m;
        refresh();
      },
    });
    const bApply = button(
      'apply $\\Pi$',
      () => {
        ensurePi();
        state.rotated = !state.rotated;
        bApply.innerHTML = texify(state.rotated ? 'undo $\\Pi$' : 'apply $\\Pi$');
        goTo(target());
        refresh();
      },
      'primary',
    );
    const bReroll = button('re-roll $\\Pi$', () => {
      state.Pi = null;
      ensurePi();
      theta = rng.uniform(0, 2 * Math.PI);
      if (state.rotated) goTo(target());
      refresh();
      cvIn.request();
    });

    const panel = el(
      'div.col',
      { style: { gap: '12px' } },
      el('div', { style: { display: 'flex', flexDirection: 'column', gap: '10px' } }, sPreset.el, el('div.row', {}, sBits.el), el('div.row', {}, sMode.el)),
      el('div.row', {}, bApply, bReroll),
      el('div.readouts', {}, rBefore.el, rAfter.el),
      el(
        'div.card',
        { 'data-reveal': '1' },
        rich('h3', 'For every fixed $x\\in\\S^{d-1}$'),
        eq('\\Pi x\\sim\\mathrm{Unif}(\\S^{d-1})'),
        eq('\\|\\Pi x-\\tilde y\\|=\\|x-\\Pi^\\top\\tilde y\\|'),
      ),
      el(
        'div.card.hot',
        { 'data-reveal': '2' },
        el('div', {
          style: { fontFamily: 'var(--serif)', fontSize: '1.25em', lineHeight: '1.25' },
          html: 'Randomize the algorithm &nbsp;⇔&nbsp; randomize the input',
        }),
      ),
      legend([
        { label: 'coordinates', color: 'accent' },
        { label: 'rounded', color: 'teal' },
        { label: 'error', color: 'violet' },
      ]),
    );

    const bottom = el(
      'div',
      { style: { display: 'grid', gridTemplateColumns: 'minmax(0,1.25fr) minmax(0,1fr)', gap: '14px', height: 'clamp(170px, 27vh, 300px)', flex: '0 0 auto' } },
      vErr,
      vIn,
    );
    const left = el('div.col', { style: { gap: '14px' } }, vMain, bottom);
    body.append(el('div.stage', { style: { minHeight: '0' } }, left, panel));

    const updateReadouts = () => {
      const x = presets[state.preset];
      rBefore.set(err2(x, naiveQuant(x, state.b, state.mode).q).toFixed(3), 'red');
      if (rotX) {
        const y = rotX[state.preset];
        rAfter.set(err2(y, naiveQuant(y, state.b, state.mode).q).toFixed(3), 'teal');
      } else rAfter.set('apply Π', 'ink-3');
    };

    ctx.onEnter(() => {
      ensurePi();
      refresh();
    });
    ctx.loop((dt) => {
      if (morph.p < 1) morph.p = Math.min(1, morph.p + dt / 1.1);
      // fill the whisker pool a couple of rotations per frame
      if (pool.length < POOL) {
        for (let n = 0; n < 2 && pool.length < POOL; n++) {
          const P = M.randomRotation(D, rng);
          pool.push(Object.fromEntries(KEYS.map((k) => [k, M.matVec(P, presets[k])])));
        }
        cvErr.redraw();
      }
      theta += dt * 0.22;
      updateReadouts();
      cvMain.redraw();
      cvIn.redraw();
    });
  },
};

import { el, rich, eq, slider, segmented, button, readout, legend, viz } from '../lib/ui.js';
import { createCanvas, text, label, alpha, color } from '../lib/canvas.js';
import * as M from '../lib/math.js';

// ---------- Synthetic key model (illustrative, after QJL Fig. 2) ----------
const D = 128;
const N_KEYS = 200; // keys used by the live simulation
const N_TOK = 48; // tokens shown in the heatmap
const OUTLIER_CH = [9, 46, 77, 113]; // "approximately four" fixed channels (QJL Fig. 2 caption)
const MAG_MAX = 14;

function makeKeys(layer, seed) {
  const rng = new M.RNG(seed);
  const deep = layer === 'deep';
  const sigma = Array.from({ length: D }, () => (deep ? 0.7 + 0.6 * rng.next() : 0.4 + 0.5 * rng.next()));
  const sgn = OUTLIER_CH.map(() => (rng.next() < 0.5 ? -1 : 1));
  const keys = [];
  for (let t = 0; t < N_KEYS; t++) {
    const k = new Float64Array(D);
    for (let c = 0; c < D; c++) k[c] = M.clamp(sigma[c] * rng.normal(), -2.5, 2.5);
    if (deep) OUTLIER_CH.forEach((c, j) => (k[c] = sgn[j] * (12 + 2 * rng.next())));
    keys.push(k);
  }
  // Channels ranked by energy over these keys (how we pick the outlier set; the paper doesn't say).
  const energy = new Float64Array(D);
  for (const k of keys) for (let c = 0; c < D; c++) energy[c] += k[c] * k[c];
  const order = [...Array(D).keys()].sort((a, b) => energy[b] - energy[a]);
  let total = 0;
  for (let c = 0; c < D; c++) total += energy[c];
  return { keys, energy, order, total };
}

// ---------- Quantizers ----------
/** TurboQuant_mse on an arbitrary-norm vector of any dimension: store ‖x‖, quantize x/‖x‖. */
function tqSub(x, P, b) {
  const n = M.norm(x);
  if (n === 0) return new Float64Array(x.length);
  const y = M.matVec(P, M.scale(x, 1 / n));
  const { centroids } = M.sphereCodebook(x.length, b);
  return M.scale(M.matTVec(P, M.quantizeWith(y, centroids)), n);
}

/** math.js turboMse (unit-norm) with the norm stored separately. */
function tqFull(x, Pi, b) {
  const n = M.norm(x);
  return M.scale(M.turboMse(M.scale(x, 1 / n), Pi, b), n);
}

/**
 * One instance at the same average bit budget: rotate all 128 channels once, then give n_o of
 * the (exchangeable) rotated coordinates b_hi bits and the rest b_lo bits.
 */
function tqMixed(x, Pi, no, bh, bl) {
  if (no === 0 || bh === bl) return tqFull(x, Pi, bl);
  if (no === D) return tqFull(x, Pi, bh);
  const n = M.norm(x);
  const y = M.matVec(Pi, M.scale(x, 1 / n));
  const yq = new Float64Array(D);
  yq.set(M.quantizeWith(y.subarray(0, no), M.sphereCodebook(D, bh).centroids), 0);
  yq.set(M.quantizeWith(y.subarray(no), M.sphereCodebook(D, bl).centroids), no);
  return M.scale(M.matTVec(Pi, yq), n);
}

const relErr = (x, xt) => {
  const e = M.sub(x, xt);
  return M.dot(e, e) / M.dot(x, x);
};

const bitsStr = (b) => `${b} bit${b === 1 ? '' : 's'}`;
const effBits = (no, bh, bl) => (no * bh + (D - no) * bl) / D;
const BIT_COLOR = { 1: 'ink-3', 2: 'blue', 3: 'teal', 4: 'yellow' };

export default {
  id: 'outliers',
  chapter: 'Practice',
  kicker: 'In the KV cache',
  title: 'Outlier channels get more bits',
  notes: `Top-left heatmap: key magnitudes |k| for 48 tokens × 128 channels. It is ILLUSTRATIVE (synthetic), drawn after QJL Fig. 2 (Llama-2 key magnitudes). Toggle the layer: in an early layer (QJL shows layer 0) there is no outlier structure and every magnitude stays below about 2.5. In deep layers (QJL shows layers 15 and 31) a few fixed channels, "approximately four" per the Fig. 2 caption, sit at about 12–14 for every token, and the rest stay around 1.
Why this matters: QJL's and TurboQuant's errors scale with the vector's norm. TurboQuant_mse error is about D_mse·‖k‖², and QJL's inner-product variance is at most π/(2m)·‖q‖²‖k‖². Four channels at ~13 hold most of ‖k‖² (the "outlier energy" readout), so the error they cause spreads over all 128 coordinates after the rotation.
The fix both papers use: split the channels into an outlier set and a non-outlier set, then run two INDEPENDENT quantizer instances, each with its own rotation, codebook and norm. Give the outliers more bits. Effective bits = (n_o·b_hi + (128 − n_o)·b_lo)/128.
TurboQuant's 2.5-bit setting (§4.3): "32 outlier channels are quantized at 3 bits, while the remaining 96 channels use 2 bits", and the paper prints (32×3 + 96×2)/128 = 2.5. CAREFUL: that arithmetic is 288/128 = 2.25, not 2.5 (checked against the arXiv v1 HTML). The slide defaults to this split and shows the honest 2.25. Either the channel counts or the result is a typo (64 @ 3 + 64 @ 2 would give exactly 2.5). Another reconciliation, purely our guess: two fp16 norms per 128-dim vector add 32/128 = 0.25 bits/channel, and 2.25 + 0.25 = 2.5. We don't know which one is right.
For 3.5 bits the paper only says "a different ratio" and doesn't give the split; drag the sliders to explore (e.g. 64 @ 4 + 64 @ 3 = 3.5). Memory vs 16-bit at the nominal widths: 16/2.5 = 6.4× and 16/3.5 ≈ 4.57× (not counting norms).
Bottom-left live simulation: 200 synthetic keys from the selected layer model, d = 128. Bars show the mean relative error ‖k − k̃‖²/‖k‖²:
 • one TurboQuant_mse instance at b_lo or b_hi everywhere (grey, for reference; math.js turboMse with one 128×128 rotation, norm stored);
 • one instance at the SAME average bits (violet): rotate all 128 channels, then give n_o of the rotated coordinates b_hi bits. After a rotation the coordinates are exchangeable, so it doesn't matter which ones;
 • the split (teal): two independent instances of dimension n_o and 128 − n_o, each with its own Haar rotation, Lloyd–Max codebook for that dimension, and norm.
In the deep layer with the paper's split (32 @ 3 + 96 @ 2) the split brings the error down to about the 3-bit single instance, because the bits go where the energy is. In the early layer there's no energy concentration, so the split gains little. The win needs the outlier energy share to be much bigger than n_o/128.
How we pick the outlier set here: the n_o channels with the most energy across these keys, a stand-in for QJL's "identified during the prompt" phase.
What TurboQuant does NOT say: how outlier channels are chosen; whether keys and values are treated the same; whether it uses the mse or the prod variant in the KV experiments; whether the extra per-instance norm (two floats per vector instead of one, about +0.25 bits/channel at fp16 for d = 128) is counted in "2.5 bits". QJL says outliers are identified during the prompt and use two independent QJL quantizers with more bits for outliers.`,
  mount(body, ctx) {
    const state = { layer: 'deep', mix: 1, mixTarget: 1, no: 32, bh: 3, bl: 2, hover: null };
    const data = { early: makeKeys('early', 101), deep: makeKeys('deep', 202) };
    let sim = null; // lazily built: rotations + results

    // ---------- Heatmap + allocation strip ----------
    const vHeat = viz('', 'key magnitude $|k|$ · illustrative, after QJL Fig. 2');
    vHeat.style.height = 'clamp(250px, 41vh, 620px)';
    let geo = null;

    const allocation = () => {
      const bits = new Int8Array(D).fill(state.bl);
      for (let i = 0; i < state.no; i++) bits[data[state.layer].order[i]] = state.bh;
      return bits;
    };

    const heatColor = (m) => {
      const t = Math.pow(M.clamp(m / MAG_MAX, 0, 1), 0.55);
      return alpha('accent', 0.06 + 0.94 * t);
    };

    const cvHeat = createCanvas(vHeat, (g, w, h) => {
      const L = 56;
      const R = 70;
      const T = 38;
      const stripH = 16;
      const B = 46;
      const hmH = h - T - B - stripH - 14;
      const cw = (w - L - R) / D;
      const rh = hmH / N_TOK;
      geo = { L, T, cw, rh, hmH, w, h };
      const ke = data.early.keys;
      const kd = data.deep.keys;
      const s = M.smoothstep(state.mix);
      for (let t = 0; t < N_TOK; t++) {
        for (let c = 0; c < D; c++) {
          const m = Math.abs(ke[t][c]) * (1 - s) + Math.abs(kd[t][c]) * s;
          g.fillStyle = heatColor(m);
          g.fillRect(L + c * cw, T + t * rh, Math.ceil(cw), Math.ceil(rh));
        }
      }
      // axes
      g.save();
      g.translate(L - 14, T + hmH / 2);
      g.rotate(-Math.PI / 2);
      text(g, 'tokens →', 0, 0, { align: 'center', size: 12, color: 'ink-3' });
      g.restore();
      if (!state.hover) text(g, 'channel', L + (w - L - R) / 2, h - 10, { align: 'center', size: 12, color: 'ink-3' });
      for (const c of [0, 32, 64, 96, 127]) text(g, String(c), L + (c + 0.5) * cw, h - 28, { align: 'center', size: 11, font: 'mono', color: 'ink-3' });

      // allocation strip aligned with the channel axis
      const sy = T + hmH + 8;
      const bits = allocation();
      for (let c = 0; c < D; c++) {
        g.fillStyle = color(BIT_COLOR[bits[c]]);
        g.globalAlpha = bits[c] === state.bh && state.no > 0 && state.bh !== state.bl ? 1 : 0.55;
        g.fillRect(L + c * cw + 0.5, sy, Math.max(1, cw - 1), stripH);
      }
      g.globalAlpha = 1;
      text(g, 'bits', L - 8, sy + stripH / 2, { align: 'right', size: 12, color: 'ink-3' });

      // color bar
      const bx = w - R + 18;
      const bh = hmH;
      for (let i = 0; i < 60; i++) {
        g.fillStyle = heatColor(MAG_MAX * (1 - i / 60));
        g.fillRect(bx, T + (i * bh) / 60, 12, bh / 60 + 1);
      }
      for (const m of [0, 2.5, 7, 14]) {
        text(g, String(m), bx + 18, T + bh * (1 - m / MAG_MAX), { size: 11, font: 'mono', color: 'ink-3' });
      }

      if (state.hover) {
        const { t, c } = state.hover;
        const m = Math.abs(ke[t][c]) * (1 - s) + Math.abs(kd[t][c]) * s;
        g.strokeStyle = color('ink');
        g.lineWidth = 1.5;
        g.strokeRect(L + c * cw, T, cw, hmH);
        label(g, `channel ${c} · token ${t} · |k| = ${m.toFixed(1)} · ${bitsStr(bits[c])}`, L + (w - L - R) / 2, h - 10, { size: 12, color: 'ink', align: 'center' });
      }
    });

    cvHeat.canvas.addEventListener('pointermove', (e) => {
      if (!geo) return;
      const r = cvHeat.canvas.getBoundingClientRect();
      const x = e.clientX - r.left;
      const y = e.clientY - r.top;
      const c = Math.floor((x - geo.L) / geo.cw);
      const t = Math.floor((y - geo.T) / geo.rh);
      state.hover = c >= 0 && c < D && t >= 0 && t < N_TOK ? { c, t } : null;
      cvHeat.request();
    });
    cvHeat.canvas.addEventListener('pointerleave', () => {
      state.hover = null;
      cvHeat.request();
    });

    // ---------- Live simulation bars ----------
    const vSim = viz('', 'live simulation · 200 synthetic keys · $d=128$');
    vSim.style.height = 'clamp(190px, 25vh, 380px)';
    let results = null;

    const cvSim = createCanvas(vSim, (g, w, h) => {
      if (!results) {
        text(g, 'computing…', w / 2, h / 2, { align: 'center', color: 'ink-3' });
        return;
      }
      const L = Math.min(300, w * 0.42);
      const Rm = 70;
      const T = 36;
      const rowH = (h - T - 12) / results.length;
      const vmax = Math.max(...results.map((r) => r.v)) * 1.05;
      text(g, '‖k − k̃‖² / ‖k‖²', w - 14, 16, { align: 'right', size: 12, color: 'ink-3' });
      results.forEach((r, i) => {
        const y = T + i * rowH;
        const bh = Math.min(26, rowH * 0.6);
        const cy = y + rowH / 2;
        text(g, r.label, L - 12, cy, { align: 'right', size: 13, color: r.hot ? 'ink' : 'ink-2', weight: r.hot ? 600 : 400 });
        const bw = ((w - L - Rm) * r.v) / vmax;
        g.fillStyle = r.hot ? color(r.color) : alpha(r.color, 0.75);
        g.beginPath();
        g.roundRect(L, cy - bh / 2, Math.max(2, bw), bh, [0, 4, 4, 0]);
        g.fill();
        text(g, r.v.toFixed(4), L + bw + 8, cy, { size: 12, font: 'mono', color: 'ink-2' });
      });
    });

    function runSim() {
      if (!sim) return;
      const { keys, order } = data[state.layer];
      const { no, bh, bl } = state;
      const set = order.slice(0, no);
      const rest = order.slice(no);
      const Po = no > 0 ? sim.rot(no) : null;
      const Pr = no < D ? sim.rot(D - no) : null;
      const acc = { lo: 0, mixed: 0, split: 0, hi: 0 };
      for (const k of keys) {
        acc.lo += relErr(k, tqFull(k, sim.Pi, bl));
        acc.hi += relErr(k, tqFull(k, sim.Pi, bh));
        acc.mixed += relErr(k, tqMixed(k, sim.Pi, no, bh, bl));
        const out = new Float64Array(D);
        if (Po) {
          const q = tqSub(Float64Array.from(set, (c) => k[c]), Po, bh);
          set.forEach((c, j) => (out[c] = q[j]));
        }
        if (Pr) {
          const q = tqSub(Float64Array.from(rest, (c) => k[c]), Pr, bl);
          rest.forEach((c, j) => (out[c] = q[j]));
        }
        acc.split += relErr(k, out);
      }
      const n = keys.length;
      const b = effBits(no, bh, bl);
      results = [
        { label: `1 instance · ${bitsStr(bl)} everywhere`, v: acc.lo / n, color: 'ink-3' },
        { label: `1 instance · ${fmtBits(b)} bits (same budget)`, v: acc.mixed / n, color: 'violet' },
        { label: `split: 2 instances · ${fmtBits(b)} bits`, v: acc.split / n, color: 'teal', hot: true },
        { label: `1 instance · ${bitsStr(bh)} everywhere`, v: acc.hi / n, color: 'ink-3' },
      ];
      cvSim.request();
    }
    let simQueued = false;
    const requestSim = () => {
      if (simQueued) return;
      simQueued = true;
      requestAnimationFrame(() => {
        simQueued = false;
        runSim();
      });
    };

    // ---------- Controls ----------
    const fmtBits = (b) => (Number.isInteger(b) ? String(b) : b.toFixed(3).replace(/0+$/, ''));
    const rBits = readout('effective bits');
    const rMem = readout('memory saving $16/\\bar b$');
    const rEnergy = readout('outlier energy share');
    const rShare = readout('outlier channel share');

    function update() {
      const b = effBits(state.no, state.bh, state.bl);
      rBits.set(fmtBits(b), 'teal');
      rMem.set(`${(16 / b).toFixed(2)}×`, 'teal');
      const { energy, order, total } = data[state.layer];
      let eo = 0;
      for (let i = 0; i < state.no; i++) eo += energy[order[i]];
      rEnergy.set(`${((100 * eo) / total).toFixed(0)}%`, 'accent');
      rShare.set(`${((100 * state.no) / D).toFixed(0)}%`);
      cvHeat.request();
      requestSim();
    }

    const sNo = slider({ label: 'outliers $n_o$', min: 0, max: 128, step: 8, value: state.no, onInput: (v) => ((state.no = v), syncPreset(), update()) });
    const sHi = slider({ label: '$b_{\\text{hi}}$ (outliers)', min: 1, max: 4, step: 1, value: state.bh, onInput: (v) => ((state.bh = v), syncPreset(), update()) });
    const sLo = slider({ label: '$b_{\\text{lo}}$ (the rest)', min: 1, max: 4, step: 1, value: state.bl, onInput: (v) => ((state.bl = v), syncPreset(), update()) });

    const presets = [
      { key: 'paper', no: 32, bh: 3, bl: 2, label: "paper's 2.5-bit split: 32 @ 3 + 96 @ 2" },
    ];
    const presetBtns = presets.map((p) => {
      const b = button(p.label, () => {
        Object.assign(state, { no: p.no, bh: p.bh, bl: p.bl });
        sNo.set(p.no, false);
        sHi.set(p.bh, false);
        sLo.set(p.bl, false);
        syncPreset();
        update();
      });
      b.style.textAlign = 'left';
      return { p, b };
    });
    function syncPreset() {
      for (const { p, b } of presetBtns) {
        const on = p.no === state.no && p.bh === state.bh && p.bl === state.bl;
        b.classList.toggle('primary', on);
        if (p.key === 'paper') flag.style.display = on ? '' : 'none';
      }
    }
    const flag = rich(
      'p.note',
      'Paper prints $\\tfrac{32\\cdot3+96\\cdot2}{128}=2.5$, but that is $2.25$.',
      { style: { fontSize: '0.8em', color: 'var(--yellow)', margin: '0' } },
    );

    const layerSeg = segmented({
      options: [
        { label: 'early layer', value: 'early' },
        { label: 'deep layer', value: 'deep' },
      ],
      value: state.layer,
      onChange: (v) => {
        state.layer = v;
        state.mixTarget = v === 'deep' ? 1 : 0;
        update();
      },
    });

    const panel = el(
      'div.panel',
      { style: { gap: '10px', paddingTop: '14px', paddingBottom: '14px' } },
      el('div.row', { style: { justifyContent: 'space-between' } }, el('h3', {}, 'Layer'), layerSeg.el),
      el('h3', { style: { marginTop: '4px' } }, 'Split the channels'),
      sNo.el,
      sHi.el,
      sLo.el,
      eq('\\bar b = \\frac{n_o\\, b_{\\text{hi}} + (128-n_o)\\, b_{\\text{lo}}}{128}'),
      el('div.col', { style: { gap: '8px' } }, ...presetBtns.map((x) => x.b)),
      flag,
      el('div.readouts', { style: { gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' } }, rBits.el, rMem.el, rEnergy.el, rShare.el),
      legend([
        { label: '1 bit', color: 'ink-3' },
        { label: '2 bits', color: 'blue' },
        { label: '3 bits', color: 'teal' },
        { label: '4 bits', color: 'yellow' },
      ]),
    );

    body.append(el('div.stage', {}, el('div.col', {}, vHeat, vSim), panel));
    syncPreset();
    update();

    ctx.loop((dt) => {
      if (state.mix !== state.mixTarget) {
        const step = dt / 0.9;
        state.mix = state.mixTarget > state.mix ? Math.min(state.mixTarget, state.mix + step) : Math.max(state.mixTarget, state.mix - step);
        cvHeat.redraw();
      }
    });

    ctx.onEnter(() => {
      if (!sim) {
        const rng = new M.RNG(2504);
        const rots = new Map();
        sim = {
          Pi: M.randomRotation(D, rng),
          rot(n) {
            if (!rots.has(n)) rots.set(n, M.randomRotation(n, new M.RNG(1000 + n)));
            return rots.get(n);
          },
        };
        runSim();
      }
      cvHeat.request();
    });
  },
};

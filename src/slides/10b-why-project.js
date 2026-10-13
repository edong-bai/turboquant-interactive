import { el, rich, slider, button, readout, viz, legend } from '../lib/ui.js';
import { createCanvas, View2D, arrow, circle, dot, line, label, draggable, alpha, color } from '../lib/canvas.js';
import { Plot } from '../lib/plot.js';
import { RNG, normalPdf } from '../lib/math.js';

const SQ = Math.sqrt(Math.PI / 2);
const sgn = (v) => (v >= 0 ? 1 : -1);
const K0 = [10, 0.1];

export default {
  id: 'why-project',
  chapter: 'QJL',
  kicker: 'Why randomness helps',
  title: 'Why project before taking the sign?',
  notes: `Bridge from QJL to TurboQuant. QJL = project with a random S, then keep signs. A natural question: why not skip the projection and just keep the signs of k itself, i.e. k̂ = (‖k‖/√d)·sign(k)? Same bit budget. Answer: the sign throws away magnitudes, and only after the projection are those magnitudes predictable.

Concrete example (default state): k = (10, 0.1), q = (0, 1). True ⟨q,k⟩ = 0.1.
Raw signs: sign(k) = (+1, +1). Even rescaled to keep the norm (‖k‖/√2 ≈ 7.07 per coordinate), the estimate is ≈ 7.07, off by ~70×. The big channel's 10 and the small channel's 0.1 became indistinguishable. In 2-D there are only 4 representable directions (the diagonals); drag k around and the red arrow just snaps between them.
And it's wrong the same way every time: rotate q (right card 2) — the red curve is a fixed, wrong function of the query. Nothing averages it away.

Project first with a random Gaussian S, then take signs (this is QJL, from the previous slides). Three things change:

1. The information lost by the sign becomes recoverable. Every coordinate of Sk is N(0, ‖k‖²) whatever k looked like, so the average magnitude the sign throws away is known exactly: E|sᵀk| = ‖k‖·√(2/π). That's why the estimator just multiplies by √(π/2) and the stored norm, and why no per-block scales or zero points are needed (card 1: raw k is two spikes; Sk/‖k‖ is always a standard normal). The QJL authors describe this as the JL preconditioning letting them quantize to one bit without normalizing.

2. Systematic error becomes random error that averages out. Raw quantization is deterministic: a given key always has the same error, aligned with the same query directions. With a random S each row is an independent estimate that is correct on average, so errors cancel as rows are added: variance ∝ 1/m, and the guarantee holds for any q and k, not just well-behaved data (card 2: teal = one draw of S, scattered around the true curve; re-draw S).

3. Energy is spread evenly. A key with almost all its weight in one channel becomes, after projection, a vector whose coordinates are all equally informative, so each sign bit carries about the same amount of information instead of most bits encoding near-zero channels (card 3).

Bottom histogram: many independent draws of S (300–1500 depending on m). QJL estimates center on 0.1 (mean readout), with standard deviation ≈ √(((π/2)‖q‖²‖k‖² − ⟨q,k⟩²)/m) ≈ 0.78 at m = 256 — not zero, but random noise around the right answer that shrinks with m. The red line (raw signs, 7.07) never moves.

Transition — the same idea fixes our naive quantizer. Points 1–3 are exactly what the naive "normalize, then round" slides were missing: there, the input's coordinates were whatever the data made them (a spread vector rounding to 0, an outlier channel crushing the rest), so errors were systematic and input-dependent. Randomize first and the coordinates follow a KNOWN distribution whatever the input was, the error becomes random, and energy is spread evenly. Next slide: do this with a random rotation (keeps lengths, so we can still reconstruct) instead of a projection: what if no input could be adversarial?`,
  mount(body, ctx) {
    const st = { ka: Math.atan2(K0[1], K0[0]), kn: Math.hypot(K0[0], K0[1]), qa: Math.PI / 2, mExp: 8 };
    const m = () => 2 ** st.mExp;
    const kv = () => [st.kn * Math.cos(st.ka), st.kn * Math.sin(st.ka)];
    const qv = () => [Math.cos(st.qa), Math.sin(st.qa)];
    const naiveVec = (k) => [(st.kn / Math.SQRT2) * sgn(k[0]), (st.kn / Math.SQRT2) * sgn(k[1])];

    // One fixed draw of S (up to 1024 rows of 2-D Gaussians), re-drawable.
    const rngS = new RNG(41);
    let S = [];
    const drawS = () => {
      S = Array.from({ length: 1024 }, () => [rngS.normal(), rngS.normal()]);
    };
    drawS();
    const qjlOnce = (q, k, mm) => {
      let acc = 0;
      for (let i = 0; i < mm; i++) acc += sgn(S[i][0] * k[0] + S[i][1] * k[1]) * (S[i][0] * q[0] + S[i][1] * q[1]);
      return (SQ / mm) * st.kn * acc;
    };

    // Many independent draws of S → distribution of the QJL estimate.
    const rngT = new RNG(7);
    let est = null;
    let dirty = true;
    function simulate() {
      const k = kv();
      const q = qv();
      const mm = m();
      // Keep each recompute to ~300k samples so dragging stays smooth at large m.
      const trials = Math.max(300, Math.min(1500, Math.round(300000 / mm)));
      est = new Float64Array(trials);
      for (let t = 0; t < trials; t++) {
        let acc = 0;
        for (let i = 0; i < mm; i++) {
          const a = rngT.normal();
          const b = rngT.normal();
          acc += sgn(a * k[0] + b * k[1]) * (a * q[0] + b * q[1]);
        }
        est[t] = (SQ / mm) * st.kn * acc;
      }
      dirty = false;
    }

    // ---------- readouts ----------
    const rTrue = readout('true $\\langle q,k\\rangle$');
    const rNaive = readout('raw signs');
    const rQjl = readout('QJL mean ± sd');
    const rTheory = readout('theory sd');

    // ---------- main sandbox ----------
    const vSand = viz('short', 'drag $k$ and $q$ · arrows show direction');
    let view;
    const sand = createCanvas(vSand, (g, w, h) => {
      view = new View2D({ w, h, xmin: -1.45, xmax: 1.45, ymin: -1.2, ymax: 1.2, pad: 10 });
      const k = kv();
      const q = qv();
      const ku = [Math.cos(st.ka), Math.sin(st.ka)];
      // a few random directions from S, colored by the sign they record for k
      for (let i = 0; i < 28; i++) {
        const n = Math.hypot(S[i][0], S[i][1]) || 1;
        const u = [S[i][0] / n, S[i][1] / n];
        const pos = S[i][0] * k[0] + S[i][1] * k[1] >= 0;
        line(g, view.sx(0), view.sy(0), view.sx(1.2 * u[0]), view.sy(1.2 * u[1]), { color: pos ? 'teal' : 'ink-3', alpha: pos ? 0.35 : 0.18 });
      }
      circle(g, view.sx(0), view.sy(0), view.len(1), { stroke: 'line-2', width: 1.5 });
      // the only 4 directions raw sign codes can represent
      for (const sx of [-1, 1])
        for (const sy of [-1, 1]) {
          const p = [sx / Math.SQRT2, sy / Math.SQRT2];
          circle(g, view.sx(p[0]), view.sy(p[1]), 7, { stroke: 'red', width: 1.5 });
        }
      const nv = naiveVec(k);
      arrow(g, view.sx(0), view.sy(0), view.sx(nv[0] / st.kn), view.sy(nv[1] / st.kn), { color: 'red', width: 3, dash: [6, 5] });
      arrow(g, view.sx(0), view.sy(0), view.sx(ku[0]), view.sy(ku[1]), { color: 'accent', width: 3.5 });
      arrow(g, view.sx(0), view.sy(0), view.sx(q[0]), view.sy(q[1]), { color: 'blue', width: 3 });
      label(g, `k = (${k[0].toFixed(2)}, ${k[1].toFixed(2)})`, view.sx(ku[0]) + 10, view.sy(ku[1]) + (ku[1] > 0.9 ? 16 : -14), { color: 'accent', size: 15, weight: 600 });
      label(g, 'q', view.sx(q[0]) + 8, view.sy(q[1]) - 10, { color: 'blue', size: 16, weight: 600 });
      label(g, '‖k‖/√2 · sign(k)', view.sx(nv[0] / st.kn) + 10, view.sy(nv[1] / st.kn) + 16, { color: 'red', size: 14 });
      label(g, 'only 4 codes', view.sx(-1 / Math.SQRT2) - 8, view.sy(-1 / Math.SQRT2) + 22, { color: 'red', size: 12, align: 'center' });
    });
    draggable(sand.canvas, {
      pick(px, py) {
        if (!view) return null;
        const near = (a) => Math.hypot(view.sx(Math.cos(a)) - px, view.sy(Math.sin(a)) - py) < 24;
        if (near(st.ka)) return 'k';
        if (near(st.qa)) return 'q';
        return null;
      },
      move(hd, px, py) {
        const a = Math.atan2(view.wy(py), view.wx(px));
        if (hd === 'k') st.ka = a;
        else st.qa = a;
        dirty = true;
        redrawAll();
      },
    });

    // ---------- estimate histogram ----------
    const vHist = viz('short', 'many independent draws of $S$');
    const hist = createCanvas(vHist, (g, w, h) => {
      if (!est) return;
      const k = kv();
      const q = qv();
      const truth = q[0] * k[0] + q[1] * k[1];
      const nv = naiveVec(k);
      const naive = q[0] * nv[0] + q[1] * nv[1];
      let mu = 0;
      for (const v of est) mu += v;
      mu /= est.length;
      let sd = 0;
      for (const v of est) sd += (v - mu) ** 2;
      sd = Math.sqrt(sd / est.length);
      const lo = Math.min(truth - 4 * sd, naive - 1);
      const hi = Math.max(truth + 4 * sd, naive + 1);
      const bins = 60;
      const hgt = new Float64Array(bins);
      const bw = (hi - lo) / bins;
      for (const v of est) {
        const b = Math.floor((v - lo) / bw);
        if (b >= 0 && b < bins) hgt[b]++;
      }
      for (let i = 0; i < bins; i++) hgt[i] /= est.length * bw;
      const ymax = Math.max(...hgt) * 1.18 || 1;
      const plot = new Plot(g, { x: 0, y: 22, w, h: h - 22, xmin: lo, xmax: hi, ymin: 0, ymax, yticks: 3, xlabel: 'estimate of ⟨q, k⟩', margin: { l: 44, r: 16, t: 10, b: 40 } });
      plot.frame();
      plot.bars(hgt, lo, hi, { color: 'teal', alpha: 0.7 });
      plot.vline(truth, { color: 'ink', width: 2, dash: null });
      plot.vline(mu, { color: 'teal', width: 2, dash: [3, 3] });
      plot.vline(naive, { color: 'red', width: 2.5, dash: [6, 4] });
      label(g, `true ${truth.toFixed(2)}`, plot.tx(truth) + 6, plot.py + 12, { color: 'ink', size: 13 });
      label(g, `raw signs ${naive.toFixed(2)}`, plot.tx(naive) + (plot.tx(naive) > w - 150 ? -8 : 6), plot.py + 12, {
        color: 'red',
        size: 13,
        align: plot.tx(naive) > w - 150 ? 'right' : 'left',
      });
      const theory = Math.sqrt(Math.max(0, ((Math.PI / 2) * st.kn * st.kn - truth * truth) / m()));
      rTrue.set(truth.toFixed(3), 'ink');
      rNaive.set(naive.toFixed(2), 'red');
      rQjl.set(`${mu.toFixed(2)} ± ${sd.toFixed(2)}`, 'teal');
      rTheory.set(theory.toFixed(2));
    });

    // ---------- three mini cards ----------
    const mini = () => el('div.viz', { style: { height: '96px', minHeight: '0' } });

    // 1. lost magnitude becomes known
    const v1 = mini();
    const c1 = createCanvas(v1, (g, w, h) => {
      const k = kv();
      const half = w * 0.36;
      // raw coordinates vs what raw signs reconstruct
      const kmax = Math.max(Math.abs(k[0]), Math.abs(k[1]), st.kn / Math.SQRT2);
      const base = h - 18;
      const sc = (h - 40) / kmax;
      const nv = naiveVec(k);
      [0, 1].forEach((i) => {
        const x = 18 + i * (half / 2);
        const bwid = half / 2 - 22;
        g.fillStyle = alpha('accent', 0.85);
        g.fillRect(x, base - Math.abs(k[i]) * sc, bwid / 2, Math.max(1, Math.abs(k[i]) * sc));
        g.strokeStyle = color('red');
        g.setLineDash([4, 3]);
        g.strokeRect(x + bwid / 2 + 2, base - Math.abs(nv[i]) * sc, bwid / 2, Math.abs(nv[i]) * sc);
        g.setLineDash([]);
        label(g, `|k${i + 1}|=${Math.abs(k[i]).toFixed(1)}`, x, base + 10, { size: 11, color: 'ink-2' });
      });
      label(g, 'raw k  →  signs', 18, 10, { size: 11, color: 'ink-3' });
      // projected coordinates, normalized: always N(0,1)
      const x0 = half + 24;
      const pw = w - x0 - 10;
      const vals = [];
      for (let i = 0; i < m(); i++) vals.push((S[i][0] * k[0] + S[i][1] * k[1]) / st.kn);
      const bins = 28;
      const hh = new Float64Array(bins);
      for (const v of vals) {
        const b = Math.floor(((v + 3.5) / 7) * bins);
        if (b >= 0 && b < bins) hh[b]++;
      }
      const peak = Math.max(...hh) || 1;
      const scale = Math.max(peak / (vals.length * (7 / bins)), normalPdf(0));
      for (let i = 0; i < bins; i++) {
        const hy = ((hh[i] / (vals.length * (7 / bins))) / scale) * (h - 40);
        g.fillStyle = alpha('teal', 0.55);
        g.fillRect(x0 + (i / bins) * pw, base - hy, pw / bins - 1, hy);
      }
      g.strokeStyle = color('ink-2');
      g.lineWidth = 1.5;
      g.beginPath();
      for (let i = 0; i <= 80; i++) {
        const t = -3.5 + (7 * i) / 80;
        const X = x0 + ((t + 3.5) / 7) * pw;
        const Y = base - (normalPdf(t) / scale) * (h - 40);
        if (i) g.lineTo(X, Y);
        else g.moveTo(X, Y);
      }
      g.stroke();
      const mAbs = Math.sqrt(2 / Math.PI);
      for (const sgnv of [-1, 1]) line(g, x0 + ((sgnv * mAbs + 3.5) / 7) * pw, base, x0 + ((sgnv * mAbs + 3.5) / 7) * pw, 18, { color: 'yellow', dash: [3, 3] });
      label(g, 'Sk / ‖k‖ ~ N(0,1), always', x0, 10, { size: 11, color: 'ink-3' });
      label(g, '±√(2/π)', x0 + ((mAbs + 3.5) / 7) * pw + 4, 26, { size: 11, color: 'yellow' });
    });

    // 2. systematic error → random error
    const v2 = mini();
    const c2 = createCanvas(v2, (g, w, h) => {
      const k = kv();
      const nv = naiveVec(k);
      const N = 120;
      const pad = 8;
      const yr = st.kn * 1.25;
      const X = (i) => pad + (i / N) * (w - 2 * pad);
      const Y = (v) => h / 2 - (v / yr) * (h / 2 - 10);
      line(g, pad, Y(0), w - pad, Y(0), { color: 'line-2' });
      const curve = (f, col, width, dash) => {
        g.strokeStyle = color(col);
        g.lineWidth = width;
        g.setLineDash(dash || []);
        g.beginPath();
        for (let i = 0; i <= N; i++) {
          const phi = (2 * Math.PI * i) / N;
          const q = [Math.cos(phi), Math.sin(phi)];
          const yy = Y(f(q));
          if (i) g.lineTo(X(i), yy);
          else g.moveTo(X(i), yy);
        }
        g.stroke();
        g.setLineDash([]);
      };
      // one draw of S: noisy but centered on the truth
      for (let i = 0; i <= N; i += 2) {
        const phi = (2 * Math.PI * i) / N;
        dot(g, X(i), Y(qjlOnce([Math.cos(phi), Math.sin(phi)], k, m())), 2.2, 'teal');
      }
      curve((q) => q[0] * k[0] + q[1] * k[1], 'ink', 2);
      curve((q) => q[0] * nv[0] + q[1] * nv[1], 'red', 2, [6, 4]);
      const xi = (((st.qa % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) / (2 * Math.PI);
      line(g, pad + xi * (w - 2 * pad), 4, pad + xi * (w - 2 * pad), h - 4, { color: 'blue', dash: [3, 3] });
      label(g, 'query angle →', w - pad, h - 8, { size: 11, color: 'ink-3', align: 'right' });
    });

    // 3. energy spread evenly
    const v3 = mini();
    const c3 = createCanvas(v3, (g, w, h) => {
      const k = kv();
      const base = h - 16;
      const top = 20;
      const raw = [k[0] ** 2, k[1] ** 2].map((v) => v / (st.kn * st.kn));
      const half = w * 0.28;
      raw.forEach((v, i) => {
        const bw = half / 2 - 12;
        const x = 14 + i * (half / 2);
        g.fillStyle = alpha('accent', 0.85);
        g.fillRect(x, base - v * (base - top), bw, Math.max(1, v * (base - top)));
        label(g, `${(100 * v).toFixed(1)}%`, x, base + 9, { size: 10, color: 'ink-2' });
      });
      label(g, 'raw k: share of ‖k‖²', 14, 9, { size: 11, color: 'ink-3' });
      const nb = 32;
      const proj = [];
      let tot = 0;
      for (let i = 0; i < nb; i++) {
        const v = (S[i][0] * k[0] + S[i][1] * k[1]) ** 2;
        proj.push(v);
        tot += v;
      }
      const x0 = half + 26;
      const pw = w - x0 - 10;
      const maxShare = Math.max(...proj) / tot;
      proj.forEach((v, i) => {
        const share = v / tot;
        const hh = (share / Math.max(maxShare, 0.15)) * (base - top);
        g.fillStyle = alpha('teal', 0.75);
        g.fillRect(x0 + (i / nb) * pw, base - hh, pw / nb - 1.5, hh);
      });
      line(g, x0, base - ((1 / nb) / Math.max(maxShare, 0.15)) * (base - top), x0 + pw, base - ((1 / nb) / Math.max(maxShare, 0.15)) * (base - top), { color: 'yellow', dash: [3, 3] });
      label(g, `first ${nb} coords of Sk (dashed: 1/${nb} each)`, x0, 9, { size: 11, color: 'ink-3' });
    });

    function redrawAll() {
      sand.request();
      c1.request();
      c2.request();
      c3.request();
      hist.request();
    }

    const mSlider = slider({
      label: '$m$ rows',
      min: 3,
      max: 10,
      step: 1,
      value: st.mExp,
      format: (v) => 2 ** v,
      onInput: (v) => {
        st.mExp = v;
        dirty = true;
        redrawAll();
      },
    });
    const btnS = button('re-draw $S$', () => {
      drawS();
      redrawAll();
    });
    const btnReset = button('reset example', () => {
      st.ka = Math.atan2(K0[1], K0[0]);
      st.qa = Math.PI / 2;
      dirty = true;
      redrawAll();
    });

    const card = (title, v, caption) => el('div.card', { style: { gap: '4px', padding: '10px 14px' } }, rich('h3', title), v, rich('p.note', caption, { style: { fontSize: '0.85em' } }));

    // Left plots share whatever height the right column leaves.
    for (const v of [vSand, vHist]) Object.assign(v.style, { height: 'auto', flex: '1', minHeight: '180px' });
    const left = el(
      'div.col',
      {},
      vSand,
      vHist,
      el('div.row', {}, el('div', { style: { flex: '1', minWidth: '220px' } }, mSlider.el), btnS, btnReset),
    );
    const right = el(
      'div.col',
      {},
      card('1 · lost magnitude becomes known', v1, '$\\E\\,|s^\\top k| = \\sqrt{2/\\pi}\\,\\|k\\|$ for every $k$ ⇒ rescale by $\\sqrt{\\pi/2}\\,\\|k\\|$, no per-block constants'),
      card('2 · systematic → random error', v2, '<span class="c-red">raw signs</span>: same wrong answer every time · <span class="c-teal">one draw of $S$</span>: noise around the <b>truth</b>, shrinking with $m$'),
      card('3 · energy spread evenly', v3, 'every projected coordinate is equally informative, so every sign bit counts'),
      el('div.readouts', {}, rTrue.el, rNaive.el, rQjl.el, rTheory.el),
      rich('p.note', 'Known distribution · random (not systematic) error · energy spread evenly. <b class="c-yellow">Same trick, next: rotate the input →</b>', { style: { margin: '0', fontSize: '0.9em' } }),
      legend([
        { label: '$k$', color: 'accent' },
        { label: '$q$', color: 'blue' },
        { label: 'raw signs', color: 'red', dash: true },
        { label: 'project, then sign', color: 'teal' },
      ]),
    );
    body.append(el('div.stage.even', {}, left, right));

    ctx.loop(() => {
      if (dirty) {
        simulate();
        hist.request();
      }
    });
    ctx.onEnter(redrawAll);
  },
};

import { el, rich, segmented, button, readout, legend, viz } from '../lib/ui.js';
import { createCanvas, text, label, alpha, color } from '../lib/canvas.js';
import { Plot } from '../lib/plot.js';
import * as M from '../lib/math.js';

// ---------- Paper data (docs/paper-notes-turboquant.md) ----------

// TurboQuant Table 2 — time to quantize the dataset (seconds), 4-bit.
const DIMS = [200, 1536, 3072];
const INDEX_TIME = [
  { m: 'Product Quantization', short: 'PQ', t: [37.04, 239.75, 494.42], c: 'violet' },
  { m: 'RabitQ', short: 'RabitQ', t: [597.25, 2267.59, 3957.19], c: 'blue' },
  { m: 'TurboQuant', short: 'TurboQuant', t: [0.0007, 0.0013, 0.0021], c: 'teal', hot: true },
];

// TurboQuant Fig. 5 — recall@1@1, READ OFF THE PLOT (approximate).
const RECALL_DATASETS = ['GloVe · d=200', 'OpenAI3 · d=1536', 'OpenAI3 · d=3072'];
const RECALL = {
  4: { TurboQuant: [0.86, 0.97, 0.972], PQ: [0.82, 0.963, 0.962], RabitQ: [0.8, 0.928, 0.94] },
  2: { TurboQuant: [0.55, 0.896, 0.903], PQ: [0.49, 0.876, 0.888], RabitQ: [0.49, 0.854, 0.866] },
};
const RECALL_METHODS = [
  { m: 'TurboQuant', c: 'teal', hot: true },
  { m: 'PQ', c: 'violet' },
  { m: 'RabitQ', c: 'blue' },
];

// ---------- Live demo parameters (synthetic) ----------
const D = 128;
const N_PTS = 2000;
const N_Q = 100;
const N_CLUSTERS = 40;
const SIGMA = 1.0;
const K_MAX = 64;
const BITS = [2, 4];

const fmtSec = (t) => (t >= 1 ? `${t.toFixed(2)} s` : t >= 0.01 ? `${t.toFixed(3)} s` : `${t} s`);
const supers = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const pow10 = (e) => `10${String(e).split('').map((ch) => supers[ch]).join('')}`;

export default {
  id: 'results-nn',
  chapter: 'Practice',
  kicker: 'Experiments',
  title: 'Nearest-neighbor search with zero indexing time',
  notes: `Same quantizer, different job: compress a database of embeddings, then answer nearest-neighbor queries using inner products with the compressed vectors.
Left: TurboQuant Table 2. Time to quantize (index) the dataset at 4 bits for d = 200 (GloVe), 1536 and 3072 (DBpedia OpenAI3 embeddings, 100k points). Log axis. Pick d: Product Quantization 37.04 s / 239.75 s / 494.42 s; RabitQ 597.25 s / 2267.59 s / 3957.19 s; TurboQuant 0.0007 s / 0.0013 s / 0.0021 s. The TurboQuant-vs-PQ gap is about 5 orders of magnitude (≈ 53,000× at d = 200) and TurboQuant-vs-RabitQ about 6 (≈ 850,000× at d = 200, ≈ 1.9 million× at d = 3072). The log axis spans 8 decades. Why: TurboQuant is data-oblivious. There's no codebook to train (PQ runs k-means per subspace) and no per-dataset tuning, so indexing is one rotation (a matrix multiply) plus a per-coordinate lookup into a precomputed Lloyd–Max codebook. RabitQ isn't vectorizable on GPU, per the paper. Same hardware: a single A100.
Middle: recall@1@1 (is the true nearest neighbor ranked first?) from TurboQuant Fig. 5. These values are READ OFF THE PLOT by us, so they're approximate. Toggle 2 vs 4 bits. TurboQuant is on top in every setting even though it does no training. In the paper, all methods reach 1.0 by k ≈ 4–8 on OpenAI3, and GloVe at 2 bits gets to about 0.98–0.99 by k = 64.
Right: a LIVE, SYNTHETIC demo running in this browser. 2000 unit vectors in d = 128 drawn from 40 Gaussian clusters, plus 100 held-out queries from the same distribution. Each database vector is quantized with math.js turboMse (one random rotation, the Lloyd–Max codebook for d = 128) at 2 and 4 bits. Then recall@1@k = the fraction of queries whose exact top-1 inner-product neighbor is in the top k of the quantized ranking, for k = 1..64 (log axis). The curves climb to 1. 4 bits is near-perfect by small k; 2 bits needs a larger candidate list, the same qualitative picture as Fig. 5. The "index (JS)" readout is the wall-clock time this browser spent quantizing all 2000 vectors at one bit-width (unoptimized JavaScript, single thread, includes the inverse rotation for reconstruction). The "R@1@1" readouts are recall@1@1 at 2 and 4 bits. "New rotation" redraws Π and recomputes.
Caveat: the synthetic data isn't GloVe or OpenAI embeddings, and the absolute recall depends on how tightly clustered the data is. Only the shape of the curves carries over.`,
  mount(body, ctx) {
    // ---------- (a) indexing time ----------
    const state = { di: 1, bits: 4, grow: 0 };
    const vTime = viz('', '');
    vTime.style.height = 'clamp(240px, 44vh, 700px)';
    const rSpeed = el('div', { style: { display: 'flex', alignItems: 'baseline', gap: '10px', flexWrap: 'wrap' } });

    const cvTime = createCanvas(vTime, (g, w, h) => {
      const L = 16;
      const R = 18;
      const T = 18;
      const B = 44;
      const lo = -4;
      const hi = 4;
      const X = (t) => L + ((Math.log10(t) - lo) / (hi - lo)) * (w - L - R);
      for (let e = lo; e <= hi; e++) {
        g.strokeStyle = color(e === lo ? 'line-2' : 'grid');
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(X(10 ** e), T);
        g.lineTo(X(10 ** e), h - B);
        g.stroke();
        if ((e - lo) % 2 === 0) text(g, pow10(e), X(10 ** e), h - B + 14, { align: 'center', size: 11, font: 'mono', color: 'ink-3' });
      }
      text(g, 'seconds to quantize the dataset (log scale)', (L + w - R) / 2, h - 10, { align: 'center', size: 12, color: 'ink-3' });
      const rowH = (h - T - B) / INDEX_TIME.length;
      const s = M.smoothstep(M.clamp(state.grow, 0, 1));
      INDEX_TIME.forEach((r, i) => {
        const t = r.t[state.di];
        const cy = T + (i + 0.5) * rowH;
        const bh = Math.min(34, rowH * 0.42);
        // grow on a log scale from 10^lo to t
        const shown = 10 ** (lo + (Math.log10(t) - lo) * s);
        const x1 = X(shown);
        g.fillStyle = alpha(r.c, r.hot ? 1 : 0.8);
        g.beginPath();
        g.roundRect(X(10 ** lo), cy - bh / 2 + 8, Math.max(2, x1 - X(10 ** lo)), bh, [0, 4, 4, 0]);
        g.fill();
        text(g, r.m, L + 4, cy - bh / 2 - 6, { size: 13, color: r.hot ? 'teal' : 'ink-2', weight: r.hot ? 600 : 400 });
        const txt = fmtSec(t);
        const inside = x1 - X(10 ** lo) > 90;
        if (inside) text(g, txt, x1 - 8, cy + 8, { align: 'right', size: 12, font: 'mono', color: 'bg', weight: 600 });
        else label(g, txt, x1 + 8, cy + 8, { size: 12, font: 'mono', color: 'ink' });
      });
    });

    function updateSpeed() {
      const tq = INDEX_TIME[2].t[state.di];
      const pq = INDEX_TIME[0].t[state.di];
      const rb = INDEX_TIME[1].t[state.di];
      rSpeed.innerHTML = '';
      rSpeed.append(
        el('span.big-number.c-teal', { style: { fontSize: 'clamp(26px, 2.6vw, 46px)' } }, `${Math.round(pq / tq).toLocaleString('en-US')}×`),
        el('span.muted', { style: { fontSize: '0.85em' } }, `faster than PQ · ${Math.round(rb / tq).toLocaleString('en-US')}× vs RabitQ`),
      );
    }

    const dSeg = segmented({
      options: DIMS.map((d, i) => ({ label: i === 0 ? `d = ${d}` : String(d), value: i })),
      value: state.di,
      onChange: (v) => {
        state.di = v;
        state.grow = 0;
        updateSpeed();
      },
    });

    // ---------- (b) recall@1@1 read off Fig. 5 ----------
    const vRec = viz('', '');
    vRec.style.height = 'clamp(240px, 44vh, 700px)';
    const recTween = { cur: {}, k: 0 };
    const cvRec = createCanvas(vRec, (g, w, h) => {
      const L = 40;
      const R = 10;
      const T = 18;
      const B = 52;
      const ph = h - T - B;
      const Y = (v) => T + ph * (1 - v);
      for (const v of [0, 0.25, 0.5, 0.75, 1]) {
        g.strokeStyle = color(v === 0 ? 'line-2' : 'grid');
        g.beginPath();
        g.moveTo(L, Y(v));
        g.lineTo(w - R, Y(v));
        g.stroke();
        text(g, String(v), L - 6, Y(v), { align: 'right', size: 11, font: 'mono', color: 'ink-3' });
      }
      const gw = (w - L - R) / RECALL_DATASETS.length;
      const bw = Math.min(30, (gw - 18) / 3);
      const s = M.smoothstep(M.clamp(recTween.k, 0, 1));
      RECALL_DATASETS.forEach((name, di) => {
        const gx = L + di * gw + gw / 2;
        RECALL_METHODS.forEach((mm, j) => {
          const v = RECALL[state.bits][mm.m][di];
          const prev = recTween.cur[`${mm.m}${di}`] ?? 0;
          const vv = prev + (v - prev) * s;
          const x = gx + (j - 1) * (bw + 3) - bw / 2;
          g.fillStyle = alpha(mm.c, mm.hot ? 1 : 0.8);
          g.beginPath();
          g.roundRect(x, Y(vv), bw, Y(0) - Y(vv), [4, 4, 0, 0]);
          g.fill();
          if (mm.hot) label(g, String(v), x + bw / 2, Y(vv) - 10, { align: 'center', size: 11, font: 'mono', color: 'ink' });
        });
        const [a, b] = name.split(' · ');
        text(g, a, gx, h - B + 16, { align: 'center', size: 12, color: 'ink', weight: 600 });
        text(g, b, gx, h - B + 33, { align: 'center', size: 11, font: 'mono', color: 'ink-3' });
      });
    });
    const snapshotRecall = () => {
      // freeze the currently displayed values as the tween start
      const s = M.smoothstep(M.clamp(recTween.k, 0, 1));
      for (const mm of RECALL_METHODS)
        RECALL_DATASETS.forEach((_, di) => {
          const key = `${mm.m}${di}`;
          const prev = recTween.cur[key] ?? 0;
          recTween.cur[key] = prev + (RECALL[recTween.lastBits ?? state.bits][mm.m][di] - prev) * s;
        });
    };
    const bitSeg = segmented({
      options: [
        { label: '2 bits', value: 2 },
        { label: '4 bits', value: 4 },
      ],
      value: state.bits,
      onChange: (v) => {
        snapshotRecall();
        recTween.lastBits = v;
        state.bits = v;
        recTween.k = 0;
      },
    });
    recTween.lastBits = state.bits;

    // ---------- (c) live demo ----------
    const vLive = viz('', '');
    vLive.style.height = 'clamp(240px, 44vh, 700px)';
    const rIdx = readout('index (JS)');
    const r2 = readout('R@1@1 · 2b');
    const r4 = readout('R@1@1 · 4b');
    let live = null; // {X, Qs, top1, Pi, phase, ...}
    let hoverK = null;

    function makeData() {
      const rng = new M.RNG(2406);
      const centers = Array.from({ length: N_CLUSTERS }, () => M.randomUnit(D, rng));
      const mk = () => {
        const c = centers[rng.int(N_CLUSTERS)];
        const v = M.randn(D, rng);
        for (let i = 0; i < D; i++) v[i] = c[i] + (v[i] * SIGMA) / Math.sqrt(D);
        return M.normalize(v);
      };
      const X = Array.from({ length: N_PTS }, mk);
      const Qs = Array.from({ length: N_Q }, mk);
      // exact top-1 by inner product
      const top1 = Qs.map((q) => {
        let best = -Infinity;
        let bi = -1;
        for (let i = 0; i < N_PTS; i++) {
          const t = M.dot(q, X[i]);
          if (t > best) {
            best = t;
            bi = i;
          }
        }
        return bi;
      });
      return { X, Qs, top1 };
    }

    function startLive(seed) {
      if (!live) live = makeData();
      const rng = new M.RNG(seed);
      live.Pi = M.randomRotation(D, rng);
      live.Xq = BITS.map(() => new Array(N_PTS));
      live.qi = 0; // points quantized so far (both bit widths)
      live.qq = 0; // queries evaluated so far
      live.hits = BITS.map(() => new Float64Array(K_MAX + 1)); // hits[k] = #queries with rank < k
      live.quantMs = 0;
      live.done = false;
      rIdx.set('…');
      r2.set('…');
      r4.set('…');
    }

    function workLive(budgetMs) {
      if (!live || live.done) return false;
      const t0 = performance.now();
      while (performance.now() - t0 < budgetMs) {
        if (live.qi < N_PTS) {
          const s = performance.now();
          const end = Math.min(N_PTS, live.qi + 50);
          for (let i = live.qi; i < end; i++) BITS.forEach((b, j) => (live.Xq[j][i] = M.turboMse(live.X[i], live.Pi, b)));
          live.quantMs += performance.now() - s;
          live.qi = end;
          if (live.qi === N_PTS) rIdx.set(`${Math.round(live.quantMs / BITS.length)} ms`, 'teal');
        } else if (live.qq < N_Q) {
          const q = live.Qs[live.qq];
          const t = live.top1[live.qq];
          BITS.forEach((b, j) => {
            const Xq = live.Xq[j];
            const st = M.dot(q, Xq[t]);
            let rank = 0;
            for (let i = 0; i < N_PTS; i++) if (M.dot(q, Xq[i]) > st) rank++;
            for (let k = 1; k <= K_MAX; k++) if (rank < k) live.hits[j][k]++;
          });
          live.qq++;
        } else {
          live.done = true;
          r2.set((live.hits[0][1] / N_Q).toFixed(2), 'teal');
          r4.set((live.hits[1][1] / N_Q).toFixed(2), 'teal');
          break;
        }
      }
      return true;
    }

    let plot = null;
    const cvLive = createCanvas(vLive, (g, w, h) => {
      plot = new Plot(g, {
        x: 0,
        y: 0,
        w,
        h,
        xmin: 1,
        xmax: K_MAX,
        ymin: 0,
        ymax: 1.02,
        xlog: true,
        xticks: [1, 2, 4, 8, 16, 32, 64],
        yticks: [0, 0.25, 0.5, 0.75, 1],
        xfmt: (v) => String(v),
        xlabel: 'k (candidates returned)',
        ylabel: 'recall@1@k',
        margin: { l: 58, r: 16, t: 18, b: 44 },
      }).frame();
      if (!live) return;
      if (live.qi < N_PTS) {
        text(g, `quantizing… ${Math.round((100 * live.qi) / N_PTS)}%`, w / 2, h / 2, { align: 'center', color: 'ink-3' });
        return;
      }
      if (live.qq === 0) return;
      const ks = Array.from({ length: K_MAX }, (_, i) => i + 1);
      BITS.forEach((b, j) => {
        const ys = ks.map((k) => live.hits[j][k] / live.qq);
        plot.line(ks, ys, { color: 'teal', width: b === 4 ? 3 : 2.5, dash: b === 2 ? [7, 5] : null, alpha: b === 4 ? 1 : 0.85 });
        const yEnd = ys[ys.length - 1];
        label(g, `${b} bits`, plot.tx(K_MAX) - 4, plot.ty(yEnd) + (b === 4 ? -12 : 14), { align: 'right', size: 12, color: 'ink' });
      });
      if (live.qq < N_Q) text(g, `queries ${live.qq}/${N_Q}`, w - 20, h - 64, { align: 'right', size: 12, color: 'ink-3' });
      if (hoverK) {
        plot.vline(hoverK, { color: 'ink-2' });
        const vals = BITS.map((b, j) => `${b}b: ${(live.hits[j][hoverK] / live.qq).toFixed(2)}`).join('  ');
        label(g, `k = ${hoverK}   ${vals}`, plot.px + 10, plot.ty(0.08), { size: 12, color: 'ink' });
      }
    });
    cvLive.canvas.addEventListener('pointermove', (e) => {
      if (!plot) return;
      const r = cvLive.canvas.getBoundingClientRect();
      const k = Math.round(plot.ix(e.clientX - r.left));
      hoverK = k >= 1 && k <= K_MAX ? k : null;
      cvLive.request();
    });
    cvLive.canvas.addEventListener('pointerleave', () => {
      hoverK = null;
      cvLive.request();
    });

    let seed = 5;
    const reroll = button('new rotation $\\Pi$', () => {
      seed += 1;
      startLive(seed);
    });

    // ---------- Layout ----------
    const colA = el(
      'div.panel',
      { style: { gap: '10px', padding: '14px 16px' } },
      el('div.row', { style: { justifyContent: 'space-between' } }, el('h3', {}, 'Indexing time'), dSeg.el),
      el('div.muted', { style: { fontSize: '0.78em' } }, 'TurboQuant Table 2 · 4-bit · single A100'),
      vTime,
      rSpeed,
    );
    const colB = el(
      'div.panel',
      { style: { gap: '10px', padding: '14px 16px' } },
      el('div.row', { style: { justifyContent: 'space-between' } }, el('h3', {}, 'Recall@1@1'), bitSeg.el),
      el('div.muted', { style: { fontSize: '0.78em' } }, 'TurboQuant Fig. 5 · values read off the plot (≈)'),
      vRec,
      legend(RECALL_METHODS.map((m) => ({ label: m.m, color: m.c }))),
    );
    const colC = el(
      'div.panel',
      { style: { gap: '10px', padding: '14px 16px' } },
      el('div.row', { style: { justifyContent: 'space-between' } }, el('h3', {}, 'Live · synthetic'), reroll),
      rich('div.muted', `${N_PTS} clustered points, ${N_Q} queries, $d=${D}$ · TurboQuant$_{\\text{mse}}$`, { style: { fontSize: '0.78em' } }),
      vLive,
      el('div.readouts', { style: { gridTemplateColumns: 'repeat(3, minmax(0, 1fr))' } }, rIdx.el, r2.el, r4.el),
    );
    for (const v of [vTime, vRec, vLive]) {
      v.style.border = 'none';
      v.style.minHeight = '220px';
      v.style.height = 'auto';
      v.style.flex = '1 1 0';
    }
    body.append(el('div.stage.three', {}, colA, colB, colC));
    body.append(
      rich(
        'p.note.muted',
        'No codebook training: indexing is one rotation and a per-coordinate lookup, $\\tilde x = \\Pi^{\\top} c_{\\mathrm{idx}(\\Pi x)}$.',
        { style: { textAlign: 'center', fontSize: '0.9em', margin: '0' } },
      ),
    );
    updateSpeed();

    ctx.onEnter(() => {
      state.grow = 0;
      recTween.cur = {};
      recTween.k = 0;
      if (!live) startLive(seed);
    });
    ctx.loop((dt) => {
      if (state.grow < 1) {
        state.grow += dt / 1.1;
        cvTime.redraw();
      }
      if (recTween.k < 1) {
        recTween.k += dt / 0.6;
        cvRec.redraw();
      }
      if (workLive(7)) cvLive.redraw();
    });
  },
};

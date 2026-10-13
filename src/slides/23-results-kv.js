import { el, segmented, viz, legend } from '../lib/ui.js';
import { createCanvas, text, label, alpha, color, FONT } from '../lib/canvas.js';

// ---------- Data (copied from docs/paper-notes-*.md; do not edit without the source) ----------

// TurboQuant Table 1 — LongBench-E. KV = bits per channel.
const TQ_TASKS = ['SingleQA', 'MultiQA', 'Summarization', 'Few shot', 'Synthetic', 'Code', 'Average'];
const TQ_T1 = [
  {
    model: 'Llama-3.1-8B-Instruct',
    rows: [
      { m: 'Full', kv: 16, v: [45.29, 45.16, 26.55, 68.38, 59.54, 46.28, 50.06] },
      { m: 'KIVI', kv: 3, v: [43.38, 37.99, 27.16, 68.38, 59.5, 44.68, 48.5] },
      { m: 'KIVI', kv: 5, v: [45.04, 45.7, 26.47, 68.57, 59.55, 46.41, 50.16] },
      { m: 'PolarQuant', kv: 3.9, v: [45.18, 44.48, 26.23, 68.25, 60.07, 45.24, 49.78] },
      { m: 'TurboQuant', kv: 2.5, v: [44.16, 44.96, 24.8, 68.01, 59.65, 45.76, 49.44] },
      { m: 'TurboQuant', kv: 3.5, v: [45.01, 45.31, 26.0, 68.63, 59.95, 46.17, 50.06] },
    ],
  },
  {
    model: 'Ministral-7B',
    rows: [
      { m: 'Full', kv: 16, v: [47.53, 49.06, 26.09, 66.83, 53.5, 47.9, 49.89] },
      { m: 'TurboQuant', kv: 2.5, v: [48.38, 49.22, 24.91, 66.69, 53.17, 46.83, 49.62] },
    ],
  },
];

// TurboQuant Fig. 4 — needle-in-a-haystack, Llama-3.1-8B-Instruct, 4k–104k tokens, memory ratio 0.25.
const NIAH = [
  { m: 'SnapKV', v: 0.858 },
  { m: 'PyramidKV', v: 0.895 },
  { m: 'KIVI', v: 0.981 },
  { m: 'PolarQuant', v: 0.995 },
  { m: 'Full precision', v: 0.997 },
  { m: 'TurboQuant', v: 0.997 },
];

// QJL Table 1 — LongBench F1, longchat-7b-v1.5-32k.
const QJL_DATASETS = ['NarrativeQA', 'Qasper', 'MultiQA-en', 'MultiQA-zh', 'HotpotQA', '2WikiMultiQA'];
const QJL_T1 = [
  { m: 'FP16', kv: 16, v: [20.79, 29.42, 42.83, 34.33, 33.05, 24.14] },
  { m: 'KIVI', kv: 3, v: [20.96, 29.01, 40.93, 34.75, 32.79, 23.01] },
  { m: 'KVQuant', kv: 4.3, v: [20.14, 28.77, 44.22, 34.44, 34.06, 23.05] },
  { m: 'QJL', kv: 3, v: [21.83, 29.44, 41.52, 34.42, 35.62, 23.6] },
];

// QJL Table 2 — LM-eval accuracy.
const QJL_T2_TASKS = ['Lambada', 'HellaSwag', 'PIQA', 'MathQA', 'MMLU'];
const QJL_T2 = [
  { model: 'Llama-2-7B', m: 'FP16', kv: 16, v: [73.9, 57.18, 78.07, 28.11, 41.85] },
  { model: '', m: 'KIVI', kv: 3, v: [73.88, 57.13, 78.07, 28.11, 41.81] },
  { model: '', m: 'QJL', kv: 3, v: [73.88, 57.14, 78.07, 28.17, 41.78] },
  { model: 'Llama-3-8B', m: 'BF16', kv: 16, v: [75.59, 60.17, 79.65, 40.64, 62.09] },
  { model: '', m: 'QJL', kv: 3, v: [75.61, 60.13, 79.87, 40.6, 62.12] },
];

// Method → color (TurboQuant/QJL = teal, the hero; full precision = neutral).
function methodStyle(m, kv) {
  switch (m) {
    case 'Full':
    case 'FP16':
    case 'BF16':
    case 'Full precision':
      return { c: 'ink-3', a: 0.9 };
    case 'KIVI':
      return { c: 'violet', a: kv === 5 ? 0.95 : 0.6 };
    case 'PolarQuant':
      return { c: 'blue', a: 0.85 };
    case 'KVQuant':
      return { c: 'blue', a: 0.85 };
    case 'SnapKV':
    case 'PyramidKV':
      return { c: 'red', a: 0.75 };
    case 'TurboQuant':
      return { c: 'teal', a: kv === 2.5 ? 0.65 : 1, hot: true };
    case 'QJL':
      return { c: 'teal', a: 1, hot: true };
    default:
      return { c: 'ink-3', a: 0.8 };
  }
}

const fmtKv = (kv) => `${kv} bit`;

// ---------- Small charting helpers ----------

function niceStep(span, n = 5) {
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
}

/** Tweened values: each key eases toward its target inside ctx.loop. */
class Tween {
  constructor() {
    this.cur = new Map();
    this.tgt = new Map();
  }
  set(key, v) {
    this.tgt.set(key, v);
    if (!this.cur.has(key)) this.cur.set(key, 0);
  }
  get(key) {
    return this.cur.get(key) ?? 0;
  }
  step(dt) {
    let moving = false;
    const k = 1 - Math.exp(-dt * 9);
    for (const [key, t] of this.tgt) {
      const c = this.cur.get(key);
      if (Math.abs(c - t) > 1e-4) {
        this.cur.set(key, c + (t - c) * k);
        moving = true;
      } else this.cur.set(key, t);
    }
    return moving;
  }
}

export default {
  id: 'results-kv',
  chapter: 'Practice',
  kicker: 'Experiments',
  title: 'Quality-neutral KV cache at 3.5 bits',
  notes: `Left: TurboQuant Table 1, LongBench-E. Scores per task category for Llama-3.1-8B-Instruct (all six methods) and Ministral-7B (only Full and TurboQuant 2.5 are reported). The number under each bar is the KV cache size in bits per channel (16 = full precision).
Pick a task. Use "Δ vs full" to zoom in honestly: the bars then start at the full-precision score, so the tiny differences are visible without a truncated axis.
Headline: TurboQuant at 3.5 bits averages 50.06, EXACTLY the full-precision average (50.06). At 2.5 bits it loses about 0.6 points on average (49.44). KIVI at 3 bits loses about 1.6 points (48.50); most of that comes from MultiQA (37.99 vs 45.16). PolarQuant at 3.9 bits gets 49.78.
Caveats, from our reading of the paper:
 • KIVI at 5 bits averages 50.16, slightly ABOVE TurboQuant 3.5's 50.06 (= Full). So "best" is about bits, not raw score. KIVI-5 spends 1.5 more bits per channel.
 • The paper states the compression three ways: "a factor exceeding 5×" (§1.3), "at least a factor of 4.5×" (§4.3), and "4× compression" / "more than 4× quantized" (§4.2 and the Fig. 4 caption). For reference, 16/3.5 ≈ 4.57× and 16/2.5 = 6.4×. Norm storage and the outlier split's details aren't fully specified, so treat all these ratios as approximate.
 • The 2.5-bit split as printed, (32×3 + 96×2)/128, actually evaluates to 2.25, not 2.5 (see the previous slide).
 • The 3.5-bit outlier split isn't given.
Right: needle-in-a-haystack (TurboQuant Fig. 4): Llama-3.1-8B-Instruct, 4k to 104k tokens, memory ratio 0.25 (compressed KV is 25% of full). Scores: SnapKV 0.858, PyramidKV 0.895 (token-eviction methods drop context), KIVI 0.981, PolarQuant 0.995, Full precision 0.997, TurboQuant 0.997. The paper does NOT state TurboQuant's bit-width for this experiment. It only says "4× compression" and, in the caption, "more than 4× quantized". A 0.25 memory ratio would be about 4 bits per channel, but that is our inference.
Toggle "QJL (2024)" to show the earlier paper's results. QJL Table 1: LongBench F1 on longchat-7b-v1.5-32k at 3 bits vs KIVI (3 bits), KVQuant (4.3 bits) and FP16. QJL is best on NarrativeQA, Qasper and HotpotQA, and at or near FP16 elsewhere. QJL Table 2: LM-eval accuracy on Llama-2-7B (FP16 / KIVI-3 / QJL-3) and Llama-3-8B (BF16 / QJL-3). Every entry is within about 0.2 points of full precision. KIVI can't run in BF16, so there is no KIVI row for Llama-3.
Takeaway: a data-oblivious, online quantizer with no calibration matches full precision at about 3.5 bits per channel.`,
  mount(body, ctx) {
    const state = { src: 'tq', task: 6, mode: 'score', hover: null };
    const tw = new Tween();
    let bars = []; // hit boxes for hover

    // ---------- Left chart ----------
    const vMain = viz('', '');
    vMain.style.height = 'clamp(300px, 54vh, 820px)';
    const caption = el('div.muted', { style: { fontSize: '0.8em', letterSpacing: '0.04em' } });

    function spec() {
      if (state.src === 'tq') {
        const t = state.task;
        return {
          groups: TQ_T1.map((g) => ({
            name: g.model,
            ref: g.rows[0].v[t],
            bars: g.rows.map((r) => ({ key: `${g.model}|${r.m}|${r.kv}`, m: r.m, kv: r.kv, v: r.v[t] })),
          })),
          perBarLabels: true,
          unit: 'LongBench-E score',
        };
      }
      return {
        groups: QJL_DATASETS.map((name, i) => ({
          name,
          ref: QJL_T1[0].v[i],
          bars: QJL_T1.map((r) => ({ key: `${name}|${r.m}`, m: r.m, kv: r.kv, v: r.v[i] })),
        })),
        perBarLabels: false,
        unit: 'LongBench F1',
      };
    }

    function setTargets() {
      const s = spec();
      for (const g of s.groups) for (const b of g.bars) tw.set(b.key, state.mode === 'delta' ? b.v - g.ref : b.v);
      caption.textContent =
        state.src === 'tq'
          ? `TurboQuant Table 1 · LongBench-E · ${TQ_TASKS[state.task]}`
          : 'QJL Table 1 · LongBench · longchat-7b-v1.5-32k';
    }

    const cvMain = createCanvas(vMain, (g, w, h) => {
      const s = spec();
      const delta = state.mode === 'delta';
      const L = 54;
      const R = 14;
      const T = 26;
      const B = s.perBarLabels ? 74 : 44;
      const ph = h - T - B;
      // y range from the TARGET values so the axis doesn't wobble while tweening.
      const vals = s.groups.flatMap((gr) => gr.bars.map((b) => (delta ? b.v - gr.ref : b.v)));
      let ymin;
      let ymax;
      if (delta) {
        const m = Math.max(1, ...vals.map(Math.abs)) * 1.2;
        const st = niceStep(2 * m, 6);
        ymax = Math.ceil(m / st) * st;
        ymin = -ymax;
      } else {
        ymin = 0;
        const st = niceStep(Math.max(...vals), 5);
        ymax = Math.ceil((Math.max(...vals) * 1.08) / st) * st;
      }
      const Y = (v) => T + ph * (1 - (v - ymin) / (ymax - ymin));
      // grid + ticks
      const st = niceStep(ymax - ymin, delta ? 6 : 5);
      g.font = `12px ${FONT.mono}`;
      for (let v = Math.ceil(ymin / st) * st; v <= ymax + 1e-9; v += st) {
        const y = Y(v);
        g.strokeStyle = color(Math.abs(v) < 1e-9 ? 'line-2' : 'grid');
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(L, y);
        g.lineTo(w - R, y);
        g.stroke();
        const lab = delta && v > 0 ? `+${+v.toFixed(2)}` : `${+v.toFixed(2)}`;
        text(g, lab, L - 8, y, { align: 'right', size: 11, font: 'mono', color: 'ink-3' });
      }
      g.save();
      g.translate(14, T + ph / 2);
      g.rotate(-Math.PI / 2);
      text(g, delta ? 'Δ vs full precision (points)' : s.unit, 0, 0, { align: 'center', size: 12, color: 'ink-3' });
      g.restore();

      // layout: groups proportional to bar count
      const nb = s.groups.reduce((a, gr) => a + gr.bars.length, 0);
      const gapG = s.perBarLabels ? 40 : 18;
      const pw = w - L - R - gapG * (s.groups.length - 1) - 12;
      const slot = pw / nb;
      const bw = Math.min(64, slot * (s.perBarLabels ? 0.72 : 0.8));
      let x = L + 6;
      bars = [];
      const labels = [];
      for (const gr of s.groups) {
        const gx0 = x;
        for (const b of gr.bars) {
          const st2 = methodStyle(b.m, b.kv);
          const cx = x + slot / 2;
          const v = tw.get(b.key);
          const y0 = Y(delta ? 0 : 0);
          const y1 = Y(v);
          const top = Math.min(y0, y1);
          const hh = Math.max(1, Math.abs(y1 - y0));
          const hov = state.hover === b.key;
          g.fillStyle = alpha(st2.c, hov ? 1 : st2.a);
          g.beginPath();
          const rr = Math.min(4, hh);
          g.roundRect(cx - bw / 2, top, bw, hh, y1 <= y0 ? [rr, rr, 0, 0] : [0, 0, rr, rr]);
          g.fill();
          if (st2.hot) {
            g.strokeStyle = color('teal');
            g.lineWidth = 1.5;
            g.stroke();
          }
          bars.push({ key: b.key, x0: cx - slot / 2, x1: cx + slot / 2, b, ref: gr.ref, group: gr.name });
          // direct value label (target value, so it reads cleanly during tween)
          const tv = delta ? b.v - gr.ref : b.v;
          if (s.perBarLabels || hov) {
            const txt = delta ? `${tv >= 0 ? '+' : ''}${tv.toFixed(2)}` : b.v.toFixed(2);
            const ly = tv >= 0 || !delta ? top - 9 : top + hh + 10;
            labels.push([txt, cx, ly, st2.hot]);
          }
          if (s.perBarLabels) {
            const short = slot < 78 ? { TurboQuant: 'TQ', PolarQuant: 'Polar' }[b.m] || b.m : b.m;
            text(g, short, cx, T + ph + 16, { align: 'center', size: 12, color: st2.hot ? 'teal' : 'ink-2', weight: st2.hot ? 600 : 400 });
            text(g, fmtKv(b.kv), cx, T + ph + 33, { align: 'center', size: slot < 60 ? 10 : 11, font: 'mono', color: 'ink-3' });
          }
          x += slot;
        }
        // full-precision reference line in score mode
        if (!delta) {
          const yr = Y(gr.ref);
          g.save();
          g.setLineDash([4, 4]);
          g.strokeStyle = alpha('ink', 0.45);
          g.beginPath();
          g.moveTo(gx0, yr);
          g.lineTo(x, yr);
          g.stroke();
          g.restore();
        }
        for (const [txt, lx, ly, hot] of labels.splice(0)) {
          label(g, txt, lx, ly, { align: 'center', size: slot < 60 ? 10 : 12, font: 'mono', color: hot ? 'ink' : 'ink-2', weight: hot ? 600 : 400 });
        }
        text(g, gr.name, (gx0 + x) / 2, h - (s.perBarLabels ? 12 : 16), { align: 'center', size: 13, color: 'ink', weight: 600 });
        x += gapG;
      }

      if (state.hover) {
        const hb = bars.find((b) => b.key === state.hover);
        if (hb) {
          const d = hb.b.v - hb.ref;
          const msg = `${hb.group} · ${hb.b.m} ${hb.b.kv}-bit: ${hb.b.v.toFixed(2)} (${d >= 0 ? '+' : ''}${d.toFixed(2)} vs full)`;
          label(g, msg, L + 8, 12, { size: 12, color: 'ink' });
        }
      }
    });

    cvMain.canvas.addEventListener('pointermove', (e) => {
      const r = cvMain.canvas.getBoundingClientRect();
      const x = e.clientX - r.left;
      const hit = bars.find((b) => x >= b.x0 && x < b.x1);
      const key = hit ? hit.key : null;
      if (key !== state.hover) {
        state.hover = key;
        cvMain.request();
      }
    });
    cvMain.canvas.addEventListener('pointerleave', () => {
      state.hover = null;
      cvMain.request();
    });

    // ---------- Right: NIAH bars or QJL Table 2 ----------
    const vNiah = viz('', '');
    vNiah.style.height = 'clamp(300px, 54vh, 820px)';
    const niahTw = new Tween();
    NIAH.forEach((r) => niahTw.set(r.m, r.v));
    const cvNiah = createCanvas(vNiah, (g, w, h) => {
      const L = 118;
      const R = 58;
      const T = 20;
      const B = 40;
      const rowH = (h - T - B) / NIAH.length;
      const X = (v) => L + (w - L - R) * v;
      for (const v of [0, 0.25, 0.5, 0.75, 1]) {
        g.strokeStyle = color(v === 0 ? 'line-2' : 'grid');
        g.beginPath();
        g.moveTo(X(v), T);
        g.lineTo(X(v), h - B);
        g.stroke();
        text(g, String(v), X(v), h - B + 14, { align: 'center', size: 11, font: 'mono', color: 'ink-3' });
      }
      text(g, 'score', X(0.5), h - 10, { align: 'center', size: 12, color: 'ink-3' });
      NIAH.forEach((r, i) => {
        const st2 = methodStyle(r.m);
        const cy = T + (i + 0.5) * rowH;
        const bh = Math.min(30, rowH * 0.62);
        g.fillStyle = alpha(st2.c, st2.a);
        g.beginPath();
        g.roundRect(X(0), cy - bh / 2, X(niahTw.get(r.m)) - X(0), bh, [0, 4, 4, 0]);
        g.fill();
        text(g, r.m, L - 10, cy, { align: 'right', size: 13, color: st2.hot ? 'teal' : 'ink-2', weight: st2.hot ? 600 : 400 });
        text(g, r.v.toFixed(3), X(niahTw.get(r.m)) + 8, cy, { align: 'left', size: 12, font: 'mono', color: st2.hot ? 'ink' : 'ink-2', weight: st2.hot ? 600 : 400 });
      });
    });

    // QJL Table 2, transposed so it fits: rows = tasks, columns = model/method.
    const t2 = el(
      'table.data',
      { style: { fontSize: '0.84em' } },
      el(
        'thead',
        {},
        el('tr', {}, el('th', {}, ''), el('th', { colspan: 3, style: { textAlign: 'center' } }, 'Llama-2-7B'), el('th', { colspan: 2, style: { textAlign: 'center' } }, 'Llama-3-8B')),
        el('tr', {}, el('th', {}, 'task'), ...QJL_T2.map((r) => el('th', { style: r.m === 'QJL' ? { color: 'var(--teal)' } : {} }, `${r.m} · ${r.kv}b`))),
      ),
      el(
        'tbody',
        {},
        QJL_T2_TASKS.map((t, i) =>
          el(
            'tr',
            {},
            el('td', {}, t),
            ...QJL_T2.map((r) => el('td', { style: r.m === 'QJL' ? { color: 'var(--teal)', fontWeight: '600' } : {} }, r.v[i].toFixed(2))),
          ),
        ),
      ),
    );
    t2.querySelectorAll('th, td').forEach((c) => (c.style.padding = '6px 7px'));
    const t2Wrap = el(
      'div.col',
      { style: { display: 'none', gap: '14px' } },
      t2,
      el('p.note.muted', { style: { fontSize: '0.82em' } }, 'Every QJL entry is within 0.22 points of full precision. No KIVI on Llama-3: it can\u2019t run BF16.'),
      el(
        'div.card',
        { style: { alignItems: 'flex-start' } },
        el('div.big-number.c-teal', {}, '≈ 81%'),
        el('div.muted', { style: { fontSize: '0.85em' } }, 'less KV memory at 3 bits: 1 − 3/16 (QJL, "more than fivefold")'),
      ),
    );

    const rightTitle = el('h3', {});
    const rightCaption = el('div.muted', { style: { fontSize: '0.8em' } });

    // ---------- Controls ----------
    const taskSeg = segmented({
      options: TQ_TASKS.map((t, i) => ({ label: t, value: i })),
      value: state.task,
      onChange: (v) => {
        state.task = v;
        setTargets();
      },
    });
    const modeSeg = segmented({
      options: [
        { label: 'score', value: 'score' },
        { label: 'Δ vs full', value: 'delta' },
      ],
      value: state.mode,
      onChange: (v) => {
        state.mode = v;
        setTargets();
      },
    });
    const srcSeg = segmented({
      options: [
        { label: 'TurboQuant (2025)', value: 'tq' },
        { label: 'QJL (2024)', value: 'qjl' },
      ],
      value: state.src,
      onChange: (v) => {
        state.src = v;
        state.hover = null;
        taskSeg.el.style.display = v === 'tq' ? '' : 'none';
        vNiah.style.display = v === 'tq' ? '' : 'none';
        t2Wrap.style.display = v === 'tq' ? 'none' : '';
        legendTq.style.display = v === 'tq' ? '' : 'none';
        legendQjl.style.display = v === 'tq' ? 'none' : '';
        rightTitle.textContent = v === 'tq' ? 'Needle in a haystack' : 'LM-eval accuracy';
        rightCaption.textContent =
          v === 'tq'
            ? 'TurboQuant Fig. 4 · Llama-3.1-8B-Instruct · 4k–104k tokens · memory ratio 0.25 · bit-width not stated'
            : 'QJL Table 2 · KV bits per channel';
        setTargets();
        if (v === 'tq') {
          NIAH.forEach((r) => niahTw.cur.set(r.m, 0));
        }
      },
    });

    const legendTq = legend([
      { label: 'Full precision', color: 'ink-3' },
      { label: 'KIVI', color: 'violet' },
      { label: 'PolarQuant', color: 'blue' },
      { label: 'TurboQuant', color: 'teal' },
      { label: 'full-precision level', color: 'ink-2', dash: true },
    ]);
    const legendQjl = legend([
      { label: 'FP16 · 16 bit', color: 'ink-3' },
      { label: 'KIVI · 3 bit', color: 'violet' },
      { label: 'KVQuant · 4.3 bit', color: 'blue' },
      { label: 'QJL · 3 bit', color: 'teal' },
    ]);
    legendQjl.style.display = 'none';

    const left = el(
      'div.panel',
      { style: { gap: '10px', padding: '14px 16px' } },
      el('div.row', { style: { justifyContent: 'space-between' } }, el('h3', {}, 'LongBench'), modeSeg.el),
      taskSeg.el,
      vMain,
      el('div.row', { style: { justifyContent: 'space-between' } }, caption, legendTq, legendQjl),
    );
    const right = el('div.panel', { style: { gap: '10px', padding: '14px 16px' } }, rightTitle, rightCaption, vNiah, t2Wrap);
    rightTitle.textContent = 'Needle in a haystack';
    rightCaption.textContent = 'TurboQuant Fig. 4 · Llama-3.1-8B-Instruct · 4k–104k tokens · memory ratio 0.25 · bit-width not stated';

    body.append(el('div.row', { style: { justifyContent: 'flex-end', marginTop: '-8px' } }, el('span.muted', { style: { fontSize: '0.85em' } }, 'results from'), srcSeg.el));
    body.append(el('div.stage', {}, left, right));
    vMain.style.border = 'none';
    vNiah.style.border = 'none';
    vMain.style.flex = '1';
    setTargets();

    ctx.onEnter(() => {
      // Grow bars from zero each time the slide comes into view.
      for (const k of tw.cur.keys()) tw.cur.set(k, 0);
      for (const r of NIAH) niahTw.cur.set(r.m, 0);
    });
    ctx.loop((dt) => {
      if (tw.step(dt)) cvMain.redraw();
      if (niahTw.step(dt)) cvNiah.redraw();
    });
  },
};

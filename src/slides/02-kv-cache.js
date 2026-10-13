import { el, eq, slider, segmented, readout, viz, legend } from '../lib/ui.js';
import { createCanvas, line, text, label, alpha, color, FONT } from '../lib/canvas.js';
import { RNG, clamp, smoothstep } from '../lib/math.js';

const MODELS = {
  'llama31-8b': { name: 'Llama-3.1-8B', layers: 32, kvHeads: 8, headDim: 128, params: 8e9 },
  'llama2-7b': { name: 'Llama-2-7B', layers: 32, kvHeads: 32, headDim: 128, params: 7e9 },
};
const BITS = [16, 8, 4, 3.5, 3, 2.5, 2];

// Illustrative decoding text for the animation (not model output).
const WORDS = (
  'Every new token appends one key and one value to the cache and the next query ' +
  'must read all of them again before the model can emit the following token so ' +
  'the memory traffic per step keeps growing with the context length'
).split(' ');
const PROMPT = 4; // tokens already in the cache when the animation restarts
const N_MAX = 44; // cache rows before the animation restarts
const DH = 12; // head-dimension cells drawn per row (the real head dim is 128)
const STEP = 1.15; // seconds per decoding step

function bytesStr(b) {
  if (b >= 1e12) return `${(b / 1e12).toFixed(b >= 1e13 ? 1 : 2)} TB`;
  if (b >= 1e9) return `${(b / 1e9).toFixed(b >= 1e11 ? 0 : 1)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(0)} MB`;
  return `${(b / 1e3).toFixed(0)} kB`;
}

function tokensStr(n) {
  if (n >= 2 ** 20) return `${(n / 2 ** 20).toFixed(n % 2 ** 20 ? 2 : 0)}M`;
  return `${Math.round(n / 1024)}k`;
}

export default {
  id: 'kv-cache',
  chapter: 'Why',
  kicker: 'Motivation',
  title: 'The KV cache is the bottleneck',
  notes: `Left: autoregressive decoding, one token per step (the sentence is illustrative). Each step, the newest token produces a query q (blue), a key k (orange) and a value v (teal). k and v are appended as one new row to the cache. Then q is compared with every cached key, the scores go through a softmax (blue bars = attention weights p_i), and the output is the p-weighted average of the value rows.
The yellow sweep is the point of the slide: to produce ONE token, the GPU streams the ENTIRE cache (every K row and every V row, in every layer and head) from HBM. The QJL paper's framing: decoding is memory-bandwidth bound, the work per step is O(n·d) and the whole KV cache is read for every generated token. So the cache size is both a memory-capacity problem and a speed problem.

Right: the cache-size calculator. bytes = 2 (K and V) × layers × KV heads × head dim × tokens × batch × bits/8.
• Llama-3.1-8B (32 layers, 8 KV heads with grouped-query attention, head dim 128): 128 KiB per token at 16 bits. At a 128k context with batch 1 that is about 17 GB, already more than the roughly 16 GB of FP16 weights (8B params × 2 bytes).
• Llama-2-7B has no GQA (32 KV heads), so its cache is 4× bigger per token: 512 KiB/token.
• Drag the context to 1M, or raise the batch: the cache dwarfs the model.
Bits per number: the compression factor is just 16/b. QJL reports "more than fivefold" KV memory reduction at 3 bits (16/3 = 5.3×, i.e. about 81% less). TurboQuant reports quality neutrality at 3.5 bits per channel on LongBench-E for Llama-3.1-8B (16/3.5 ≈ 4.6×) and only marginal degradation at 2.5 bits (6.4×).
Caveat for the honest version: real systems also store a few overhead numbers (scales, norms) and keep some tokens in full precision; the calculator counts only the payload bits. That overhead is exactly what the next slide is about.`,
  mount(body, ctx) {
    // ---------------- Left: decoding animation ----------------
    const rng = new RNG(11);
    const tanhRow = () => Float64Array.from({ length: DH }, () => Math.tanh(0.9 * rng.normal()));
    const rows = []; // {k, v}
    let words = [];
    const reset = () => {
      rows.length = 0;
      words = WORDS.slice(0, PROMPT);
      for (let i = 0; i < PROMPT - 1; i++) rows.push({ k: tanhRow(), v: tanhRow() });
    };
    reset();
    const st = { phase: 0, q: tanhRow(), p: [], hold: 0 };

    const softmaxScores = () => {
      const s = rows.map((r) => {
        let a = 0;
        for (let j = 0; j < DH; j++) a += st.q[j] * r.k[j];
        return (2.2 * a) / Math.sqrt(DH);
      });
      const mx = Math.max(...s);
      const e = s.map((v) => Math.exp(v - mx));
      const z = e.reduce((a, b) => a + b, 0);
      st.p = e.map((v) => v / z);
    };

    const beginStep = () => {
      rows.push({ k: tanhRow(), v: tanhRow() });
      st.q = tanhRow();
      softmaxScores();
    };
    beginStep();

    const anim = viz('', 'decoding · illustrative');
    const cellRow = (g, x, y, w, h, vals, col, a = 1) => {
      const cw = w / vals.length;
      for (let j = 0; j < vals.length; j++) {
        g.fillStyle = alpha(col, a * (0.14 + 0.75 * Math.abs(vals[j])));
        g.fillRect(x + j * cw + 0.5, y + 0.5, Math.max(0.5, cw - 1), Math.max(0.5, h - 1));
      }
    };

    const av = createCanvas(anim, (g, w, h) => {
      const pad = 18;
      const availW = w - 2 * pad;
      const gw = availW * 0.17;
      const g1 = availW * 0.13;
      const g2 = availW * 0.12;
      const xQ = pad;
      const xK = xQ + gw + g1;
      const xP = xK + gw + 10;
      const xV = xK + gw + g2;
      const xO = xV + gw + availW * 0.08 - 6;
      const y0 = 108;
      const y1 = h - 44;
      const n = rows.length;
      const rh = Math.min(20, (y1 - y0) / Math.max(n, 12));
      const cacheH = n * rh;
      const yMid = y0 + Math.max(cacheH, rh * 3) / 2;
      const ph = st.phase; // 0..1 within a step
      const appear = smoothstep(clamp(ph / 0.22, 0, 1));
      const attn = smoothstep(clamp((ph - 0.22) / 0.4, 0, 1));
      const outp = smoothstep(clamp((ph - 0.62) / 0.25, 0, 1));
      const sweepY = y0 + cacheH * clamp((ph - 0.22) / 0.4, 0, 1);

      // ---- generated text strip ----
      g.save();
      g.font = `500 15px ${FONT.sans}`;
      const shown = [];
      let used = 0;
      for (let i = words.length - 1; i >= 0; i--) {
        const wpx = g.measureText(words[i]).width + 14;
        if (used + wpx > availW - 30) break;
        shown.unshift({ s: words[i], wpx, newest: i === words.length - 1 });
        used += wpx;
      }
      let x = pad;
      if (shown.length < words.length) {
        text(g, '…', x, 52, { color: 'ink-3', size: 15 });
        x += 18;
      }
      for (const wd of shown) {
        if (wd.newest) {
          g.strokeStyle = alpha('blue', 0.9);
          g.lineWidth = 1.5;
          g.beginPath();
          g.roundRect(x - 4, 40, wd.wpx - 6, 24, 6);
          g.stroke();
        }
        text(g, wd.s, x + 2, 52, { color: wd.newest ? 'blue' : 'ink-2', size: 15, weight: 500 });
        x += wd.wpx;
      }
      // the next token fades in as the output completes
      if (outp > 0.6 && words.length < WORDS.length && rows.length < N_MAX && st.hold === 0) {
        const nx = WORDS[words.length];
        text(g, nx, x + 2, 52, { color: 'teal', size: 15, weight: 500, alpha: (outp - 0.6) / 0.4 });
      }
      g.restore();

      // ---- column headers ----
      const hy = y0 - 18;
      text(g, 'q', xQ + gw / 2, hy, { color: 'blue', size: 17, weight: 600, align: 'center', font: 'serif' });
      text(g, 'K cache', xK + gw / 2, hy, { color: 'accent', size: 15, weight: 600, align: 'center' });
      text(g, 'p', xP + (xV - xP - 14) / 2, hy, { color: 'blue', size: 17, weight: 600, align: 'center', font: 'serif' });
      text(g, 'V cache', xV + gw / 2, hy, { color: 'teal', size: 15, weight: 600, align: 'center' });
      text(g, 'o', xO + gw / 2, hy, { color: 'teal', size: 17, weight: 600, align: 'center', font: 'serif' });

      // ---- lines q → every key, weighted ----
      const pmax = Math.max(...st.p);
      const qy = yMid - rh / 2;
      for (let i = 0; i < n; i++) {
        const yy = y0 + i * rh + rh / 2;
        const wgt = st.p[i] / pmax;
        line(g, xQ + gw, qy + rh / 2, xK - 2, yy, { color: 'blue', width: 0.6 + 1.6 * wgt, alpha: attn * (0.06 + 0.7 * wgt) });
        line(g, xV + gw + 2, yy, xO, qy + rh / 2, { color: 'teal', width: 0.6 + 1.6 * wgt, alpha: outp * (0.05 + 0.6 * wgt) });
      }

      // ---- cache rows ----
      for (let i = 0; i < n; i++) {
        const isNew = i === n - 1;
        const yy = y0 + i * rh;
        const a = isNew ? appear : 1;
        const slide = isNew ? (1 - appear) * 30 : 0;
        cellRow(g, xK - slide, yy, gw, rh, rows[i].k, 'accent', a);
        cellRow(g, xV - slide, yy, gw, rh, rows[i].v, 'teal', a);
        // weight bars
        const bw = (xV - xP - 22) * (st.p[i] / pmax) * attn;
        g.fillStyle = alpha('blue', 0.75);
        g.fillRect(xP, yy + rh * 0.2, bw, Math.max(1, rh * 0.6));
      }
      if (n) {
        const yNew = y0 + (n - 1) * rh;
        g.strokeStyle = alpha('ink', 0.55 * appear);
        g.lineWidth = 1;
        g.strokeRect(xK - 2, yNew - 1, gw + 4, rh + 2);
        g.strokeRect(xV - 2, yNew - 1, gw + 4, rh + 2);
      }

      // ---- the read sweep: every row is streamed for this one token ----
      if (ph > 0.22 && ph < 0.66) {
        for (const xx of [xK, xV]) {
          g.fillStyle = alpha('yellow', 0.1);
          g.fillRect(xx - 3, y0, gw + 6, sweepY - y0);
          line(g, xx - 6, sweepY, xx + gw + 6, sweepY, { color: 'yellow', width: 2, alpha: 0.9 });
        }
      }

      // ---- q and o ----
      cellRow(g, xQ, qy, gw, rh, st.q, 'blue', appear);
      const o = new Float64Array(DH);
      for (let i = 0; i < n; i++) for (let j = 0; j < DH; j++) o[j] += st.p[i] * rows[i].v[j];
      const om = Math.max(...o.map(Math.abs)) || 1;
      cellRow(
        g,
        xO,
        qy,
        gw,
        rh,
        o.map((v) => v / om),
        'teal',
        outp,
      );

      // ---- footer ----
      const fy = h - 18;
      label(g, `n = ${n} tokens`, pad, fy, { color: 'ink-2', size: 14, font: 'mono', weight: 400 });
      label(g, `this step reads all ${n} K rows + ${n} V rows`, xK - 6, fy, {
        color: ph > 0.22 && ph < 0.7 ? 'yellow' : 'ink-3',
        size: 14,
        weight: 400,
      });
      if (st.hold > 0) {
        label(g, '… and it keeps growing, in every layer and head', w / 2, y1 - 6, {
          color: 'ink',
          size: 18,
          align: 'center',
          weight: 600,
        });
      }
    });

    ctx.loop((dt) => {
      if (st.hold > 0) {
        st.hold -= dt;
        if (st.hold <= 0) {
          st.hold = 0;
          reset();
          st.phase = 0;
          beginStep();
        }
        av.redraw();
        return;
      }
      st.phase += dt / STEP;
      if (st.phase >= 1) {
        st.phase = 0;
        if (rows.length >= N_MAX || words.length >= WORDS.length) {
          st.hold = 2.6;
          st.phase = 0.99;
        } else {
          words.push(WORDS[words.length]);
          beginStep();
        }
      }
      av.redraw();
    });

    const left = el(
      'div.col',
      { style: { minHeight: '0' } },
      el('div', { 'data-reveal': '1' }, eq('\\mathrm{Attn}(q,K,V)=\\mathrm{softmax}\\!\\big(qK^\\top/\\sqrt{d}\\big)\\,V', 'big')),
      anim,
    );
    Object.assign(anim.style, { flex: '1 1 0', height: 'auto', minHeight: '300px' });

    // ---------------- Right: memory calculator ----------------
    const state = { model: 'llama31-8b', exp: 17, batch: 1, bits: 3 };
    const tokens = () => Math.round(2 ** state.exp);
    const kvBytes = (bits) => {
      const m = MODELS[state.model];
      return (2 * m.layers * m.kvHeads * m.headDim * tokens() * state.batch * bits) / 8;
    };

    const rPer = readout('KV / token @16b');
    const rComp = readout('compression $16/b$');
    const rSaved = readout('memory saved');

    const chart = viz('', '');
    Object.assign(chart.style, { flex: '1 1 0', height: 'auto', minHeight: '170px' });
    const cv = createCanvas(chart, (g, w, h) => {
      const m = MODELS[state.model];
      const items = [
        { name: 'weights · FP16', sub: `${m.params / 1e9}B params × 2 B`, v: m.params * 2, c: 'ink-3' },
        { name: 'KV cache · 16-bit', sub: '', v: kvBytes(16), c: 'accent' },
        { name: `KV cache · ${state.bits}-bit`, sub: '', v: kvBytes(state.bits), c: 'teal' },
      ];
      const vmax = Math.max(...items.map((i) => i.v));
      const lw = Math.min(150, w * 0.36);
      const x0 = 14 + lw;
      const x1 = w - 86;
      const top = 16;
      const bh = Math.min(34, (h - top - 14) / items.length - 14);
      const gap = (h - top - 10 - items.length * bh) / items.length;
      items.forEach((it, i) => {
        const y = top + i * (bh + gap) + gap / 2;
        text(g, it.name, 14, y + bh / 2 - (it.sub ? 7 : 0), { color: 'ink-2', size: 13.5, weight: 500 });
        if (it.sub) text(g, it.sub, 14, y + bh / 2 + 10, { color: 'ink-3', size: 11.5 });
        const bw = Math.max(2, ((x1 - x0) * it.v) / vmax);
        g.fillStyle = alpha(it.c, it.c === 'ink-3' ? 0.55 : 0.85);
        g.beginPath();
        g.roundRect(x0, y, bw, bh, 4);
        g.fill();
        text(g, bytesStr(it.v), x0 + bw + 8, y + bh / 2, { color: it.c === 'ink-3' ? 'ink-2' : it.c, size: 14, font: 'mono', weight: 600 });
      });
      // weights reference line
      const wx = x0 + ((x1 - x0) * items[0].v) / vmax;
      line(g, wx, top - 6, wx, h - 8, { color: 'ink-3', dash: [3, 4], alpha: 0.6 });

      rPer.set(bytesStr(kvBytes(16) / tokens() / state.batch));
      const f = 16 / state.bits;
      rComp.set(`${f.toFixed(f % 1 ? 2 : 0)}×`, 'teal');
      rSaved.set(bytesStr(kvBytes(16) - kvBytes(state.bits)), 'teal');
    });

    const segModel = segmented({
      options: Object.entries(MODELS).map(([value, m]) => ({ value, label: m.name })),
      value: state.model,
      onChange: (v) => {
        state.model = v;
        cv.request();
      },
    });
    const sCtx = slider({
      label: 'context',
      min: 10,
      max: 20,
      step: 0.05,
      value: state.exp,
      format: (v) => tokensStr(Math.round(2 ** v)),
      onInput: (v) => {
        state.exp = v;
        cv.request();
      },
    });
    const sBatch = slider({
      label: 'batch',
      min: 1,
      max: 64,
      step: 1,
      value: state.batch,
      onInput: (v) => {
        state.batch = v;
        cv.request();
      },
    });
    const segBits = segmented({
      options: BITS.map((b) => ({ value: b, label: String(b) })),
      value: state.bits,
      onChange: (v) => {
        state.bits = v;
        cv.request();
      },
    });

    const panel = el(
      'div.panel',
      { 'data-reveal': '2', style: { minHeight: '0' } },
      el('h3', {}, 'KV cache size'),
      segModel.el,
      sCtx.el,
      sBatch.el,
      el('div.row', { style: { gap: '10px' } }, el('span.muted', { style: { fontSize: '0.92em' } }, 'bits / number'), segBits.el),
      chart,
      el('div', { style: { fontSize: '0.86em' } }, eq('\\text{bytes}=2\\cdot L\\cdot H_{kv}\\cdot d_h\\cdot n\\cdot B\\cdot \\tfrac{b}{8}')),
      el('div.readouts', {}, rPer.el, rComp.el, rSaved.el),
      legend([
        { label: 'key $k$', color: 'accent' },
        { label: 'value $v$', color: 'teal' },
        { label: 'query $q$ / weights', color: 'blue' },
        { label: 'memory read', color: 'yellow' },
      ]),
    );

    body.append(el('div.stage', { style: { minHeight: '0' } }, left, panel));
    ctx.onEnter(() => cv.request());
  },
};

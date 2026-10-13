import { el, rich, slider, readout, viz } from '../lib/ui.js';
import { createCanvas, line, text, label, circle, draggable, alpha, color } from '../lib/canvas.js';
import { Plot } from '../lib/plot.js';
import { RNG, clamp } from '../lib/math.js';

/** A card with a letter, a 2–4 word label and a flexible mini canvas. */
function card(letter, title, extra = [], reveal = '1') {
  const v = viz('', '');
  Object.assign(v.style, { flex: '1 1 0', height: 'auto', minHeight: '130px' });
  const node = el(
    'div.card',
    { 'data-reveal': reveal, style: { minHeight: '0', padding: '14px 16px', gap: '8px' } },
    el(
      'div.row',
      { style: { gap: '10px' } },
      el('span', { style: { fontFamily: 'var(--serif)', fontSize: '1.5em', color: 'var(--accent)', lineHeight: '1' } }, letter),
      rich('h3', title, { style: { fontSize: '0.92em', color: 'var(--ink)' } }),
    ),
    v,
    ...extra,
  );
  return { node, v };
}

export default {
  id: 'requirements',
  chapter: 'Why',
  kicker: 'The rules',
  title: 'What a quantizer must do',
  notes: `Four requirements. Every candidate in the rest of the talk gets scored against these.

(a) Online and data-oblivious. Keys and values are produced one token at a time during decoding; you must compress each one the moment it is produced, with no chance to look at the rest of the data. Left half: the same stream as the right, quantized on arrival against a codebook that was fixed in advance. Right half: k-means / product quantization must first collect a dataset, then run Lloyd iterations to fit centroids, and only then can it encode; when the distribution drifts it has to refit. TurboQuant's nearest-neighbor table makes the cost concrete: at 4 bits, quantization takes 37–494 s for PQ and 597–3957 s for RabitQ, versus 0.0007–0.0021 s for TurboQuant (d = 200 to 3072, single A100). The QJL paper also calls QJL data-oblivious: a sketch with no tuning.

(b) Accelerator-friendly. GPUs are fast when thousands of lanes execute the same instruction in lockstep on different data. A per-coordinate rule (rotate, then look up the nearest of a few scalar levels) is exactly that: every lane does the same thing. Search over a learned codebook, tree, or graph is branchy and sequential; lanes diverge and idle. TurboQuant's paper says RabitQ is not vectorizable on GPU, and PQ needs heavy preprocessing.

(c) Zero overhead. Standard KV quantizers work in small groups and store a scale and a zero-point per group in full precision. If both are FP16 and the group has g numbers, the overhead is 32/g bits per number. Drag g: 32 numbers per group costs +1 bit, 16 costs +2. QJL's intro puts it as "1 or 2 additional bits per quantized number". At b = 3, that is a 33–67% tax on the payload. The goal is a scheme whose only side information is one number per whole vector (its norm), which is negligible for d = 128.

(d) Worst-case guarantee near the limit. We want a bound that holds for EVERY input vector (the expectation is only over the quantizer's own randomness), and we want it close to what is information-theoretically possible. Teaser: later we prove no quantizer, randomized or not, can beat 4^(−b) relative MSE on its worst input. Drag the dot ("your quantizer") below the curve: impossible. TurboQuant lands within a factor of about 2.7 of this line for every input.`,
  mount(body, ctx) {
    // ===================== (a) online vs offline =====================
    const A = card('a', 'Online · data-oblivious', [], '1');
    const rngA = new RNG(5);
    const stA = { t: 0, spawn: 0, left: [], right: [], phase: 'collect', iter: 0, iterT: 0, hold: 0, cents: [], doneL: 0, doneR: 0, drift: 0 };
    const COLLECT = 48;
    const K = 5;
    const fixedGrid = [];
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) fixedGrid.push([(i + 0.5) / 4, (j + 0.5) / 4]);
    const nearest = (p, cs) => {
      let bi = 0;
      let bd = Infinity;
      cs.forEach((c, i) => {
        const dd = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2;
        if (dd < bd) {
          bd = dd;
          bi = i;
        }
      });
      return bi;
    };
    const sample = () => {
      // mixture of three blobs whose centers drift slowly over time
      const k = rngA.int(3);
      const ang = stA.drift + (k * 2 * Math.PI) / 3;
      const cx = 0.5 + 0.24 * Math.cos(ang);
      const cy = 0.5 + 0.24 * Math.sin(ang);
      return [clamp(cx + 0.08 * rngA.normal(), 0.03, 0.97), clamp(cy + 0.08 * rngA.normal(), 0.03, 0.97)];
    };
    const cvA = createCanvas(A.v, (g, w, h) => {
      const half = w / 2;
      const pad = 12;
      const top = 26;
      const box = (side) => {
        const x0 = side * half + pad;
        const s = Math.min(half - 2 * pad, h - top - 28);
        return { x0: x0 + (half - 2 * pad - s) / 2, y0: top, s };
      };
      const L = box(0);
      const R = box(1);
      const P = (b, p) => [b.x0 + p[0] * b.s, b.y0 + (1 - p[1]) * b.s];
      line(g, half, 8, half, h - 8, { color: 'line', width: 1 });
      text(g, 'fixed codebook', half / 2, 13, { color: 'teal', size: 12.5, align: 'center', weight: 600 });
      text(g, 'k-means / PQ', half + half / 2, 13, { color: 'red', size: 12.5, align: 'center', weight: 600 });
      // left: fixed grid, quantize on arrival
      for (const c of fixedGrid) {
        const [x, y] = P(L, c);
        circle(g, x, y, 2.2, { fill: alpha('teal', 0.45) });
      }
      for (const p of stA.left) {
        const [x, y] = P(L, p.p);
        const fade = p.age > 1.1 ? Math.max(0, 1 - (p.age - 1.1) / 0.4) : 1;
        if (p.age > 0.18) {
          const [cx, cy] = P(L, fixedGrid[p.c]);
          line(g, x, y, cx, cy, { color: 'teal', width: 1.4, alpha: 0.8 * fade });
          circle(g, cx, cy, 4.5, { stroke: alpha('teal', fade), width: 1.5 });
        }
        circle(g, x, y, 3, { fill: alpha('accent', 0.9 * fade) });
      }
      // right: buffer, then fit, then encode
      for (const p of stA.right) {
        const [x, y] = P(R, p.p);
        const enc = stA.phase === 'done';
        if (enc) {
          const [cx, cy] = P(R, stA.cents[p.c]);
          line(g, x, y, cx, cy, { color: 'teal', width: 1, alpha: 0.6 });
        } else if (stA.phase === 'fit') {
          const [cx, cy] = P(R, stA.cents[nearest(p.p, stA.cents)]);
          line(g, x, y, cx, cy, { color: 'ink-3', width: 1, alpha: 0.25 });
        }
        circle(g, x, y, 2.6, { fill: alpha(enc ? 'accent' : 'ink-3', enc ? 0.9 : 0.75) });
      }
      for (const c of stA.cents) {
        const [x, y] = P(R, c);
        line(g, x - 6, y - 6, x + 6, y + 6, { color: 'teal', width: 2.5 });
        line(g, x - 6, y + 6, x + 6, y - 6, { color: 'teal', width: 2.5 });
      }
      // status lines
      const by = h - 12;
      text(g, `encoded ${stA.doneL}`, L.x0, by, { color: 'teal', size: 12, font: 'mono' });
      if (stA.phase === 'collect') {
        const f = stA.right.length / COLLECT;
        g.fillStyle = alpha('ink-3', 0.25);
        g.fillRect(R.x0, by - 4, R.s * 0.55, 8);
        g.fillStyle = alpha('red', 0.8);
        g.fillRect(R.x0, by - 4, R.s * 0.55 * f, 8);
        text(g, 'waiting', R.x0 + R.s * 0.58, by, { color: 'red', size: 12 });
      } else if (stA.phase === 'fit') {
        text(g, `fitting… iter ${stA.iter}`, R.x0, by, { color: 'red', size: 12, font: 'mono' });
      } else {
        text(g, `encoded ${stA.doneR}`, R.x0, by, { color: 'teal', size: 12, font: 'mono' });
      }
    });
    const stepA = (dt) => {
      stA.t += dt;
      stA.drift += dt * 0.12;
      stA.spawn += dt;
      while (stA.spawn > 0.16) {
        stA.spawn -= 0.16;
        const p = sample();
        stA.left.push({ p, age: 0, c: nearest(p, fixedGrid) });
        stA.doneL++;
        if (stA.phase === 'collect') stA.right.push({ p, c: 0 });
      }
      stA.left.forEach((p) => (p.age += dt));
      stA.left = stA.left.filter((p) => p.age < 1.5);
      if (stA.phase === 'collect' && stA.right.length >= COLLECT) {
        stA.phase = 'fit';
        stA.iter = 0;
        stA.iterT = 0;
        stA.cents = Array.from({ length: K }, () => [rngA.uniform(0.15, 0.85), rngA.uniform(0.15, 0.85)]);
      } else if (stA.phase === 'fit') {
        stA.iterT += dt;
        if (stA.iterT > 0.32) {
          stA.iterT = 0;
          stA.iter++;
          const sum = stA.cents.map(() => [0, 0, 0]);
          for (const p of stA.right) {
            const k = nearest(p.p, stA.cents);
            sum[k][0] += p.p[0];
            sum[k][1] += p.p[1];
            sum[k][2]++;
          }
          stA.cents = stA.cents.map((c, i) => (sum[i][2] ? [sum[i][0] / sum[i][2], sum[i][1] / sum[i][2]] : c));
          if (stA.iter >= 7) {
            stA.phase = 'done';
            stA.hold = 1.6;
            stA.right.forEach((p) => (p.c = nearest(p.p, stA.cents)));
            stA.doneR += stA.right.length;
          }
        }
      } else if (stA.phase === 'done') {
        stA.hold -= dt;
        if (stA.hold <= 0) {
          stA.phase = 'collect';
          stA.right = [];
          stA.cents = [];
        }
      }
      cvA.redraw();
    };

    // ===================== (b) SIMD lockstep vs branchy search =====================
    const B = card('b', 'Accelerator-friendly', [], '2');
    const rngB = new RNG(9);
    const stB = { tick: 0, pulse: 0, op: 0, vecs: 0, node: 0, path: [0], wait: 0, found: 0 };
    const OPS = ['rotate', 'lookup', 'store'];
    const cvB = createCanvas(B.v, (g, w, h) => {
      const half = w / 2;
      line(g, half, 8, half, h - 8, { color: 'line', width: 1 });
      text(g, 'same op, every lane', half / 2, 13, { color: 'teal', size: 12.5, align: 'center', weight: 600 });
      text(g, 'branchy search', half + half / 2, 13, { color: 'red', size: 12.5, align: 'center', weight: 600 });
      // lanes
      const n = 8;
      const s = Math.min(half - 30, h - 62) / n;
      const gx = (half - s * n) / 2;
      const gy = 28;
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
          g.fillStyle = alpha('teal', 0.18 + 0.7 * stB.pulse);
          g.fillRect(gx + j * s + 1, gy + i * s + 1, s - 2, s - 2);
        }
      text(g, OPS[stB.op], half / 2, gy + n * s + 12, { color: 'ink-2', size: 12, align: 'center', font: 'mono' });
      text(g, `vectors ${stB.vecs}`, 12, h - 12, { color: 'teal', size: 12, font: 'mono' });
      // tree
      const depth = 4;
      const pos = (k) => {
        const lvl = Math.floor(Math.log2(k + 1));
        const idx = k + 1 - 2 ** lvl;
        const cnt = 2 ** lvl;
        return [half + ((idx + 0.5) / cnt) * half, 34 + (lvl / (depth - 1)) * (h - 80)];
      };
      for (let k = 0; k < 2 ** depth - 1; k++) {
        for (const c of [2 * k + 1, 2 * k + 2]) {
          if (c >= 2 ** depth - 1) continue;
          const [x0, y0] = pos(k);
          const [x1, y1] = pos(c);
          const on = stB.path.includes(k) && stB.path.includes(c);
          line(g, x0, y0, x1, y1, { color: on ? 'accent' : 'line-2', width: on ? 2 : 1 });
        }
      }
      for (let k = 0; k < 2 ** depth - 1; k++) {
        const [x, y] = pos(k);
        const cur = k === stB.node;
        circle(g, x, y, cur ? 6 : 4, { fill: cur ? 'accent' : stB.path.includes(k) ? alpha('accent', 0.5) : 'line-2' });
      }
      text(g, 'one lane busy · others idle', half + half / 2, h - 30, { color: 'ink-3', size: 11.5, align: 'center' });
      text(g, `vectors ${stB.found}`, half + 12, h - 12, { color: 'red', size: 12, font: 'mono' });
    });
    const stepB = (dt) => {
      stB.tick += dt;
      stB.pulse = Math.max(0, stB.pulse - dt * 4);
      if (stB.tick > 0.3) {
        stB.tick = 0;
        stB.pulse = 1;
        stB.op = (stB.op + 1) % OPS.length;
        if (stB.op === 0) stB.vecs += 64;
      }
      stB.wait -= dt;
      if (stB.wait <= 0) {
        // step one node at a time; sometimes backtrack (a failed comparison)
        const leafStart = 2 ** 3 - 1;
        if (stB.node >= leafStart) {
          stB.found++;
          stB.node = 0;
          stB.path = [0];
        } else if (stB.path.length > 1 && rngB.next() < 0.25) {
          stB.path.pop();
          stB.node = stB.path[stB.path.length - 1];
        } else {
          stB.node = 2 * stB.node + 1 + rngB.int(2);
          stB.path.push(stB.node);
        }
        stB.wait = 0.25 + 0.35 * rngB.next();
      }
      cvB.redraw();
    };

    // ===================== (c) zero overhead =====================
    const stC = { b: 3, lg: 5 };
    const rEff = readout('bits / number');
    const sB = slider({
      label: '$b$',
      min: 1,
      max: 4,
      step: 1,
      value: stC.b,
      onInput: (v) => {
        stC.b = v;
        cvC.request();
      },
    });
    const sG = slider({
      label: 'group $g$',
      min: 3,
      max: 8,
      step: 1,
      value: stC.lg,
      format: (v) => String(2 ** v),
      onInput: (v) => {
        stC.lg = v;
        cvC.request();
      },
    });
    const C = card(
      'c',
      'Zero overhead',
      [
        el('div', { style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 16px', alignItems: 'center' } }, sB.el, sG.el),
        el(
          'div.row',
          { style: { gap: '12px', flexWrap: 'nowrap' } },
          rEff.el,
          el('p.note.muted', { style: { fontSize: '0.8em', margin: '0' } }, 'QJL: “1 or 2 additional bits per quantized number”'),
        ),
      ],
      '3',
    );
    Object.assign(C.v.style, { minHeight: '100px' });
    const cvC = createCanvas(C.v, (g, w, h) => {
      const b = stC.b;
      const gsz = 2 ** stC.lg;
      const ov = 32 / gsz;
      const pad = 14;
      const lw = 78;
      const x0 = pad + lw;
      const x1 = w - pad;
      const maxBits = 8;
      const sx = (v) => x0 + ((x1 - x0) * v) / maxBits;
      const bh = Math.min(24, (h - 70) / 3);
      // axis
      for (let v = 0; v <= maxBits; v++) {
        line(g, sx(v), 10, sx(v), 10 + 2 * bh + 16, { color: 'grid', width: 1 });
        text(g, String(v), sx(v), 10 + 2 * bh + 24, { color: 'ink-3', size: 10.5, align: 'center', font: 'mono' });
      }
      const bar = (y, parts, name) => {
        text(g, name, pad, y + bh / 2, { color: 'ink-2', size: 12 });
        let acc = 0;
        for (const [v, c] of parts) {
          g.fillStyle = alpha(c, 0.85);
          g.fillRect(sx(acc), y, Math.max(0, sx(acc + v) - sx(acc) - 1), bh);
          acc += v;
        }
      };
      bar(12, [[b, 'teal'], [Math.min(ov, maxBits - b), 'red']], 'grouped');
      bar(12 + bh + 8, [[b, 'teal']], 'goal');
      // the group strip: g payloads of b bits + FP16 scale + FP16 zero
      const sy = Math.min(h - 22, 2 * bh + 78);
      const sh = 14;
      const total = gsz * b + 32;
      const W = x1 - pad;
      let x = pad;
      g.fillStyle = alpha('teal', 0.75);
      const pw = (W * gsz * b) / total;
      if (gsz <= 64) {
        for (let i = 0; i < gsz; i++) g.fillRect(x + (i * pw) / gsz, sy, Math.max(0.6, pw / gsz - (gsz <= 32 ? 1 : 0.4)), sh);
      } else g.fillRect(x, sy, pw, sh);
      x += pw;
      const ow = (W * 16) / total;
      g.fillStyle = alpha('red', 0.85);
      g.fillRect(x + 1, sy, ow - 1, sh);
      g.fillStyle = alpha('red', 0.6);
      g.fillRect(x + ow + 1, sy, ow - 1, sh);
      if (ow > 30) {
        text(g, 'scale', x + ow / 2, sy + sh / 2, { color: '#1b1208', size: 10, align: 'center', weight: 600 });
        text(g, 'zero', x + ow * 1.5, sy + sh / 2, { color: '#1b1208', size: 10, align: 'center', weight: 600 });
      }
      text(g, `one group: ${gsz} × ${b} bits + 2 × 16 bits`, pad, sy - 9, { color: 'ink-3', size: 11 });
      // ov is a power of two, so String() prints it exactly (4, 2, 1, 0.5, 0.25, 0.125)
      rEff.set(`${b} + ${ov} = ${b + ov}`, ov >= 1 ? 'red' : 'ink');
    });

    // ===================== (d) worst-case near the limit =====================
    const D = card('d', 'Worst case · near the limit', [], '4');
    const stD = { b: 2.4, e: 0.15 };
    let plotD;
    const cvD = createCanvas(D.v, (g, w, h) => {
      plotD = new Plot(g, {
        x: 0,
        y: 4,
        w,
        h: h - 4,
        xmin: 0.5,
        xmax: 4.5,
        ymin: 1e-3,
        ymax: 1.5,
        ylog: true,
        xticks: [1, 2, 3, 4],
        yticks: [1e-3, 1e-2, 1e-1, 1],
        yfmt: (v) => (v >= 1 ? '1' : v >= 0.1 ? '0.1' : v >= 0.01 ? '0.01' : '0.001'),
        xlabel: 'bits per coordinate  b',
        ylabel: 'worst-case MSE',
        margin: { l: 56, r: 12, t: 8, b: 36 },
      });
      plotD.frame();
      const lb = (x) => 4 ** -x;
      // impossible region below 4^-b
      plotD.clip(() => {
        g.fillStyle = alpha('red', 0.16);
        g.beginPath();
        g.moveTo(plotD.tx(0.5), plotD.ty(1e-3));
        for (let i = 0; i <= 120; i++) {
          const x = 0.5 + (4 * i) / 120;
          g.lineTo(plotD.tx(x), plotD.ty(Math.max(1e-3, lb(x))));
        }
        g.lineTo(plotD.tx(4.5), plotD.ty(1e-3));
        g.closePath();
        g.fill();
      });
      plotD.fn(lb, { color: 'red', width: 2.5 });
      label(g, 'impossible', plotD.tx(1.1), plotD.ty(0.004), { color: 'red', size: 15, weight: 600 });
      label(g, '4⁻ᵇ', plotD.tx(1.0) + 6, plotD.ty(lb(1.0)) - 12, { color: 'red', size: 14, weight: 600 });
      // the draggable candidate
      const bad = stD.e < lb(stD.b);
      const px = plotD.tx(stD.b);
      const py = plotD.ty(stD.e);
      circle(g, px, py, 9, { fill: bad ? 'red' : 'yellow' });
      circle(g, px, py, 13, { stroke: bad ? 'red' : 'yellow', width: 1.5 });
      const flip = px > plotD.px + plotD.pw - 170;
      label(g, bad ? '✗ no quantizer can' : 'your quantizer?', px + (flip ? -18 : 18), py, {
        color: bad ? 'red' : 'yellow',
        size: 13,
        weight: 600,
        align: flip ? 'right' : 'left',
      });
    });
    draggable(cvD.canvas, {
      pick(px, py) {
        if (!plotD) return null;
        return Math.hypot(px - plotD.tx(stD.b), py - plotD.ty(stD.e)) < 24 ? 'p' : null;
      },
      move(_, px, py) {
        stD.b = clamp(plotD.ix(px), 0.6, 4.4);
        const t = clamp((py - plotD.py) / plotD.ph, 0, 1);
        stD.e = Math.exp(Math.log(1.5) + t * (Math.log(1e-3) - Math.log(1.5)));
        cvD.request();
      },
    });

    const grid = el(
      'div.stage.even',
      { style: { gridAutoRows: '1fr', minHeight: '0' } },
      A.node,
      B.node,
      C.node,
      D.node,
    );
    body.append(grid);

    ctx.loop((dt) => {
      stepA(dt);
      stepB(dt);
    });
    ctx.onEnter(() => {
      cvC.request();
      cvD.request();
    });
  },
};

import { el, eq, button, readout, viz, toggle, segmented, texify } from '../lib/ui.js';
import { createCanvas, View2D, arrow, circle, dot, line, label, gridLines, alpha } from '../lib/canvas.js';
import { Plot } from '../lib/plot.js';
import { RNG, histogram } from '../lib/math.js';
import { dragQK, clipRect, dirLine, halfPlane, DirSampler, runningMeans, strip } from './_shared-qjl.js';

const SQ = Math.sqrt(Math.PI / 2);
const TRIALS = 1500;
const DOT_SCALE = 0.5; // s is drawn at half size so the Gaussian cloud fits

export default {
  id: 'qjl',
  chapter: 'QJL',
  kicker: 'QJL',
  title: 'Keep only the sign: more rows beat more precision',
  notes: `The JL slide ended with: float rows are expensive (16 bits each), and to save memory we'd have to drop rows, which costs accuracy. QJL's answer: quantize each row as hard as possible, one bit, the sign, and spend the saved bits on MANY more rows. Def. 3.1: H_S(k) = sign(Sk) ∈ {±1}^m, plus ‖k‖ as one float. The query is not quantized:
  Prod_QJL(q,k) = (√(π/2)/m) · ‖k‖ · ⟨Sq, sign(Sk)⟩.

LEFT, the geometry of one sign bit (2-D is exact: only the projection of s onto span(q,k) matters):
- The orange half-plane is the set of directions s with s·k > 0; every random s (dots, half size) is colored by its bit.
- For the newest s, the sample √(π/2)·‖k‖·sign(s·k)·(s·q) drops onto the number line. "+100" a few times: the running average converges to ⟨q,k⟩. One bit per row and still unbiased (the √(π/2) is explained on the next slide).
- Toggle "sign both sides (SimHash)": also taking sign(Sq) estimates the angle (fraction of disagreeing bits ≈ θ/π, the red wedges), and ‖q‖‖k‖cos(π·fraction) is biased (cos of an unbiased estimate; our exact binomial check: −0.073 at θ = 45°, m = 8). The query is in full precision at decode time anyway, so QJL keeps it: asymmetric.

RIGHT, the point of this slide: same bits per key, two ways to spend them.
- Floats (plain JL): B/16 rows of 16-bit floats. Var = (⟨q,k⟩² + ‖q‖²‖k‖²)/(B/16).
- Sign bits (QJL): B − 16 rows of 1 bit, plus 16 bits for ‖k‖. Var = ((π/2)‖q‖²‖k‖² − ⟨q,k⟩²)/(B − 16) (derived from the QJL paper's moment formula).
- Histograms (live simulation, 1500 trials each): teal = QJL up, violet = floats down, same x-axis. Both are centered on the truth (both unbiased); the sign version is several times narrower. Readout: measured sd ratio, ≈ 3–7× depending on the angle between q and k (≈ 3.2× at 90°, more when q is near k). Change "bits per key".
- Bottom-right curve (theory): error sd vs bits per key, log–log. Both fall like 1/√B (parallel lines: same rate, better constant for signs). Reference: storing k itself in FP16 at d = 128 costs 2048 bits and is exact.

Takeaway: a coarse value on many random rows beats a precise value on a few. And it needs no per-block scale or zero point, just one norm per key ("zero overhead" in the QJL title). The catch, for later: error still only falls like 1/√bits; TurboQuant will get 2^(−b).`,
  mount(body, ctx) {
    const rng = new RNG(1010);
    const trialRng = new RNG(10101);
    const state = { q: [1.0, 0.18], k: [0.62, 0.78], simhash: false, B: 256 };
    const smp = new DirSampler(rng, 5000);
    let contrib = [];
    let means = new Float64Array(0);
    let disagree = 0;

    const nq = () => Math.hypot(...state.q);
    const nk = () => Math.hypot(...state.k);
    const ipTrue = () => state.q[0] * state.k[0] + state.q[1] * state.k[1];
    const sgn = (v) => (v >= 0 ? 1 : -1);
    const sdot = (s, v) => s[0] * v[0] + s[1] * v[1];
    const one = (s) => SQ * nk() * sgn(sdot(s, state.k)) * sdot(s, state.q);
    const simhashEst = () => (smp.s.length ? nq() * nk() * Math.cos((Math.PI * disagree) / smp.s.length) : null);

    function recomputeAll() {
      contrib = smp.s.map(one);
      means = runningMeans(contrib);
      disagree = 0;
      for (const s of smp.s) if (sgn(sdot(s, state.q)) !== sgn(sdot(s, state.k))) disagree++;
    }

    // ---------- sandbox ----------
    const v = viz('', 'drag $q$ and $k$ · 2-D sandbox');
    let view;
    const cv = createCanvas(v, (g, w, h) => {
      const planeH = Math.round(h * 0.6);
      const plane = { x: 0, y: 0, w, h: planeH };
      view = new View2D({ w, h: planeH, xmin: -1.6, xmax: 1.6, ymin: -1.5, ymax: 1.5, pad: 14 });
      const { q, k } = state;
      clipRect(g, plane, () => {
        gridLines(g, view, 0.5);
        halfPlane(g, view, k, 'accent', 0.1);
        dirLine(g, view, [-k[1], k[0]], { color: 'accent', alpha: 0.45, width: 1.2, dash: [6, 5] });
        if (state.simhash) {
          halfPlane(g, view, q, 'blue', 0.07);
          dirLine(g, view, [-q[1], q[0]], { color: 'blue', alpha: 0.45, width: 1.2, dash: [6, 5] });
        }
        // the cloud of random directions, colored by their bit
        const S = smp.s;
        for (let i = 0; i < S.length; i++) {
          const s = S[i];
          const bk = sgn(sdot(s, k));
          let c = bk > 0 ? 'accent' : 'ink-3';
          if (state.simhash && sgn(sdot(s, q)) !== bk) c = 'red';
          g.fillStyle = alpha(c, 0.55);
          g.beginPath();
          g.arc(view.sx(DOT_SCALE * s[0]), view.sy(DOT_SCALE * s[1]), 2.2, 0, Math.PI * 2);
          g.fill();
        }
        if (S.length) {
          const s = S[S.length - 1];
          const ss = s[0] * s[0] + s[1] * s[1];
          dirLine(g, view, s, { color: 'ink', alpha: 0.35, width: 1.2 });
          const fq = sdot(s, q) / ss;
          line(g, view.sx(q[0]), view.sy(q[1]), view.sx(fq * s[0]), view.sy(fq * s[1]), { color: 'blue', dash: [4, 4], alpha: 0.8 });
          dot(g, view.sx(fq * s[0]), view.sy(fq * s[1]), 5, 'blue');
          const bk = sgn(sdot(s, k));
          arrow(g, view.sx(0), view.sy(0), view.sx(DOT_SCALE * s[0]), view.sy(DOT_SCALE * s[1]), { color: bk > 0 ? 'accent' : 'ink-2', width: 2.2, head: 9 });
          label(g, bk > 0 ? '+1' : '−1', view.sx(DOT_SCALE * s[0]) + 8, view.sy(DOT_SCALE * s[1]) - 10, { color: bk > 0 ? 'accent' : 'ink-2', size: 15, weight: 600, font: 'mono' });
          const c = contrib[contrib.length - 1];
          label(g, `√(π/2)‖k‖ · sign(s·k) · (s·q) = ${c.toFixed(2)}`, plane.x + 14, plane.h - 16, { color: 'ink-2', size: 13.5, font: 'mono' });
        }
        circle(g, view.sx(0), view.sy(0), 3, { fill: 'ink-3' });
        arrow(g, view.sx(0), view.sy(0), view.sx(k[0]), view.sy(k[1]), { color: 'accent', width: 3.5 });
        arrow(g, view.sx(0), view.sy(0), view.sx(q[0]), view.sy(q[1]), { color: 'blue', width: 3.5 });
        label(g, 'k', view.sx(k[0]) + 8, view.sy(k[1]) - 10, { color: 'accent', size: 18, weight: 600 });
        label(g, 'q', view.sx(q[0]) + 8, view.sy(q[1]) - 10, { color: 'blue', size: 18, weight: 600 });
        {
          const a = Math.atan2(k[1], k[0]) + 0.95;
          label(g, 's·k > 0', view.sx(1.25 * Math.cos(a)), view.sy(1.25 * Math.sin(a)), { color: 'accent', size: 13, align: 'center' });
          label(g, 's·k < 0', view.sx(-1.25 * Math.cos(a)), view.sy(-1.25 * Math.sin(a)), { color: 'ink-3', size: 13, align: 'center' });
        }
      });
      line(g, 0, planeH, w, planeH, { color: 'line' });

      const truth = ipTrue();
      const R = 3.4 * nq() * nk() + 0.3;
      const falling = smp.recent.filter((r) => r.age < 0.45);
      const drops = falling.map((r) => ({ v: contrib[r.i], t: r.age / 0.45 }));
      const skip = new Set(falling.map((r) => r.i));
      const landed = skip.size ? contrib.filter((_, i) => !skip.has(i)) : contrib;
      const markers = [
        { v: truth, color: 'ink', dash: [5, 4], label: '⟨q,k⟩' },
        { v: means.length ? means[means.length - 1] : null, color: 'teal', label: `QJL avg of ${means.length}` },
      ];
      if (state.simhash) markers.push({ v: simhashEst(), color: 'red', label: 'SimHash' });
      strip(g, { x: 22, y: planeH + 8, w: w - 44, h: h - planeH - 12 }, {
        lo: truth - R,
        hi: truth + R,
        values: landed,
        color: 'teal',
        drops,
        title: 'samples √(π/2)‖k‖ · sign(s·k) · (s·q)',
        markers,
      });
    });

    // ---------- same bits: m floats (JL) vs many sign bits (QJL) ----------
    // Bits per key B. Float JL: B/16 rows of 16-bit floats. QJL: B − 16 sign bits + a 16-bit norm.
    const rowsF = () => state.B / 16;
    const rowsS = () => state.B - 16;
    const theorySd = (B) => {
      const ip = ipTrue();
      const b2 = (nq() * nk()) ** 2;
      return {
        f: Math.sqrt((ip * ip + b2) / (B / 16)),
        s: Math.sqrt(Math.max(0, (Math.PI / 2) * b2 - ip * ip) / (B - 16)),
      };
    };
    const tF = [];
    const tQ = [];
    function resetTrials() {
      tF.length = 0;
      tQ.length = 0;
    }
    function runTrials(rowBudget) {
      const { q, k } = state;
      const mf = rowsF();
      const ms = rowsS();
      const a = nk();
      let used = 0;
      while (used < rowBudget && tQ.length < TRIALS) {
        let accF = 0;
        for (let r = 0; r < mf; r++) {
          const s0 = trialRng.normal();
          const s1 = trialRng.normal();
          accF += (s0 * k[0] + s1 * k[1]) * (s0 * q[0] + s1 * q[1]);
        }
        let accS = 0;
        for (let r = 0; r < ms; r++) {
          const s0 = trialRng.normal();
          const s1 = trialRng.normal();
          accS += sgn(s0 * k[0] + s1 * k[1]) * (s0 * q[0] + s1 * q[1]);
        }
        tF.push(accF / mf);
        tQ.push((SQ / ms) * a * accS);
        used += mf + ms;
      }
    }
    const meanOf = (a) => a.reduce((x, y) => x + y, 0) / (a.length || 1);
    const sdOf = (a) => {
      const mu = meanOf(a);
      return Math.sqrt(a.reduce((x, y) => x + (y - mu) ** 2, 0) / (a.length || 1));
    };

    const vT = viz('', 'same bits per key · live simulation');
    Object.assign(vT.style, { height: 'auto', flex: '1', minHeight: '170px' });
    const cvT = createCanvas(vT, (g, w, h) => {
      const truth = ipTrue();
      const th = theorySd(state.B);
      const R = 3.2 * th.f;
      const lo = truth - R;
      const hi = truth + R;
      const bins = 48;
      const hf = histogram(tF, lo, hi, bins);
      const hq = histogram(tQ, lo, hi, bins);
      const mx = Math.max(...hq, ...hf) || 1;
      const p = new Plot(g, { x: 0, y: 26, w, h: h - 26, xmin: lo, xmax: hi, ymin: -1.1, ymax: 1.1, yticks: [], margin: { l: 12, b: 24, t: 4, r: 10 } });
      p.frame();
      const drawBars = (hist, c, sgnY) => {
        const bw = (hi - lo) / bins;
        p.clip(() => {
          g.fillStyle = alpha(c, 0.6);
          for (let i = 0; i < bins; i++) {
            if (!hist[i]) continue;
            const x0 = p.tx(lo + i * bw);
            const x1 = p.tx(lo + (i + 1) * bw);
            const y0 = p.ty(0);
            const y1 = p.ty((sgnY * hist[i]) / mx);
            g.fillRect(x0 + 0.5, Math.min(y0, y1), Math.max(0.5, x1 - x0 - 1), Math.abs(y1 - y0));
          }
        });
      };
      drawBars(hq, 'teal', 1);
      drawBars(hf, 'violet', -1);
      p.hline(0, { color: 'line-2', dash: null });
      p.vline(truth, { color: 'ink', dash: [5, 4], width: 1.5 });
      label(g, `sign bits: ${rowsS()} rows × 1 bit + ‖k‖`, p.px + 6, p.py + 12, { size: 12, color: 'teal' });
      label(g, `floats: ${rowsF()} rows × 16 bits`, p.px + 6, p.py + p.ph - 12, { size: 12, color: 'violet' });
      label(g, `truth ${truth.toFixed(2)}`, p.px + p.pw - 6, p.py + 12, { size: 12, color: 'ink', align: 'right', font: 'mono' });
      const sF = sdOf(tF);
      const sQ = sdOf(tQ);
      rSdF.set(sF.toFixed(3), 'violet');
      rSdQ.set(sQ.toFixed(3), 'teal');
      rRatio.set(sQ > 0 ? `${(sF / sQ).toFixed(1)}×` : '–', 'yellow');
    });

    // error sd vs bits per key (theory), with FP16 storage of k as the reference
    const vC = viz('', 'error sd vs bits per key · theory');
    Object.assign(vC.style, { height: 'auto', flex: '1', minHeight: '170px' });
    const cvC = createCanvas(vC, (g, w, h) => {
      const lo = Math.max(1e-3, theorySd(4096).s * 0.6);
      const hi = theorySd(32).f * 1.6;
      const p = new Plot(g, {
        x: 0,
        y: 26,
        w,
        h: h - 26,
        xmin: 32,
        xmax: 4096,
        ymin: lo,
        ymax: hi,
        xlog: true,
        ylog: true,
        xticks: [32, 128, 512, 2048],
        xfmt: (v) => String(v),
        margin: { l: 44, b: 26, t: 4, r: 10 },
      });
      p.frame();
      p.fn((B) => theorySd(B).f, { color: 'violet', width: 2.2 });
      p.fn((B) => theorySd(B).s, { color: 'teal', width: 2.2, lo: 32 });
      p.vline(2048, { color: 'ink-3', dash: [4, 4] });
      label(g, 'FP16 k, d = 128: exact', p.tx(2048) - 6, p.py + 12, { size: 11.5, color: 'ink-2', align: 'right' });
      const th = theorySd(state.B);
      p.vline(state.B, { color: 'yellow', dash: [2, 3] });
      p.points([state.B, state.B], [th.f, th.s], { color: 'yellow', r: 4, alpha: 1 });
      label(g, 'floats', p.tx(48), p.ty(theorySd(48).f) - 10, { size: 12, color: 'violet' });
      label(g, 'sign bits', p.tx(48), p.ty(theorySd(48).s) + 12, { size: 12, color: 'teal' });
    });

    // ---------- pipeline diagram ----------
    const chip = (tex, c, extra = {}) =>
      el('span', {
        style: {
          display: 'inline-flex',
          alignItems: 'center',
          padding: '2px 9px',
          borderRadius: '8px',
          border: `1px solid var(--${c})`,
          color: `var(--${c})`,
          background: 'rgba(8,14,28,.45)',
          whiteSpace: 'nowrap',
          ...extra,
        },
        html: texify(tex),
      });
    const arr = () => el('span', { style: { color: 'var(--ink-3)' } }, '→');
    const qSign = chip('$\\sign$', 'red', { display: 'none' });
    const qSignArr = arr();
    qSignArr.style.display = 'none';
    const rowStyle = { display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap', fontSize: '0.9em' };
    const pipeline = el(
      'div',
      { 'data-reveal': '2', style: { display: 'flex', flexDirection: 'column', gap: '8px' } },
      el('div', { style: rowStyle }, chip('$k$', 'accent'), arr(), chip('$Sk$', 'accent'), arr(), chip('$\\sign$', 'accent'), arr(), chip('$\\{\\pm1\\}^m$', 'accent'), el('span.muted', {}, '+'), chip('$\\norm{k}$', 'accent')),
      el('div', { style: rowStyle }, chip('$q$', 'blue'), arr(), chip('$Sq$', 'blue'), qSignArr, qSign, arr(), el('span.muted', { html: texify('$\\langle\\cdot,\\cdot\\rangle$') })),
    );

    // ---------- readouts & controls ----------
    const rM = readout('$m$', '0');
    const rEst = readout('QJL est.');
    const rTrue = readout('$\\langle q,k\\rangle$');
    const rBits = readout('bits / key');
    const rSim = readout('SimHash');
    rSim.el.style.display = 'none';
    const rSdF = readout('sd · floats');
    const rSdQ = readout('sd · sign bits');
    const rRatio = readout('signs better by');

    function refresh() {
      const m = smp.s.length;
      rM.set(String(m));
      rTrue.set(ipTrue().toFixed(3), 'ink');
      rEst.set(m ? means[m - 1].toFixed(3) : '–', 'teal');
      rBits.set(String(m + 16), 'accent');
      const se = simhashEst();
      rSim.set(se == null ? '–' : se.toFixed(3), 'red');
      cv.request();
    }

    dragQK(cv, () => view, state, () => {
      recomputeAll();
      resetTrials();
      refresh();
      cvT.request();
      cvC.request();
    });

    const tSim = toggle({
      label: 'sign both sides (SimHash)',
      onChange: (on) => {
        state.simhash = on;
        qSign.style.display = on ? 'inline-flex' : 'none';
        qSignArr.style.display = on ? '' : 'none';
        rSim.el.style.display = on ? '' : 'none';
        refresh();
      },
    });
    const segB = segmented({
      label: 'bits per key',
      options: [64, 128, 256, 512, 1024],
      value: state.B,
      onChange: (B) => {
        state.B = B;
        resetTrials();
        cvT.request();
        cvC.request();
      },
    });

    ctx.loop((dt) => {
      if (tQ.length < TRIALS) {
        runTrials(60000);
        cvT.redraw();
      }
      if (!smp.busy) return;
      const n = smp.step(dt, 5);
      if (n) {
        for (let i = smp.s.length - n; i < smp.s.length; i++) {
          const s = smp.s[i];
          contrib.push(one(s));
          if (sgn(sdot(s, state.q)) !== sgn(sdot(s, state.k))) disagree++;
        }
        means = runningMeans(contrib);
        refresh();
      }
      cv.redraw();
    });
    let first = true;
    ctx.onEnter(() => {
      if (first) {
        first = false;
        smp.add(3);
      }
      refresh();
    });

    const panel = el(
      'div.col',
      {},
      el(
        'div.card',
        { 'data-reveal': '1', style: { gap: '4px', padding: '12px 16px' } },
        eq('\\mathrm{Prod}_{\\rm QJL}(q,k)=\\frac{\\sqrt{\\pi/2}}{m}\\,\\norm{k}\\,\\big\\langle Sq,\\ \\sign(Sk)\\big\\rangle'),
        pipeline,
      ),
      el('div.row', {}, segB.el),
      el('div.row', { style: { flex: '1', minHeight: '0', alignItems: 'stretch' } }, vT, vC),
      el('div.readouts', { style: { gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '8px' } }, rSdF.el, rSdQ.el, rRatio.el),
    );
    const leftCol = el(
      'div.col',
      {},
      v,
      el(
        'div.row',
        {},
        button('+1', () => smp.add(1), 'primary'),
        button('+100', () => smp.add(100)),
        button('reset', () => {
          smp.reset();
          recomputeAll();
          refresh();
        }),
        tSim.el,
      ),
      el('div.readouts', { style: { gridTemplateColumns: 'repeat(auto-fit, minmax(78px, 1fr))', gap: '8px' } }, rM.el, rEst.el, rTrue.el, rSim.el, rBits.el),
    );
    Object.assign(v.style, { height: 'auto', flex: '1', minHeight: '300px' });
    body.append(el('div.stage.even', {}, leftCol, panel));
    recomputeAll();
    refresh();
  },
};

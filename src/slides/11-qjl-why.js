import { el, eq, rich, button, readout, viz } from '../lib/ui.js';
import { createCanvas, View2D, arrow, circle, line, label, gridLines } from '../lib/canvas.js';
import { Plot } from '../lib/plot.js';
import { RNG, normalPdf } from '../lib/math.js';
import { dragQK, clipRect, dirLine, halfPlane, DirSampler, strip } from './_shared-qjl.js';

const SQ = Math.sqrt(Math.PI / 2);
const MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
const T_PER_M = 300;
const MAX_SAMPLES = 2500;

export default {
  id: 'qjl-why',
  chapter: 'QJL',
  kicker: 'QJL',
  title: 'Why one bit stays unbiased',
  notes: `The geometric proof of Lemma 3.2 (QJL is unbiased), in one picture.

Split the query along the key: q = (⟨q,k⟩/‖k‖²)·k + q⊥, where q⊥ is orthogonal to k (blue dashed = parallel part, violet = q⊥). Drag q and k to see the decomposition move.

One row of the estimator is sign(sᵀk)·sᵀq = (⟨q,k⟩/‖k‖²)·|sᵀk| + sign(sᵀk)·sᵀq⊥.
- Second term: for Gaussian s, sᵀq⊥ and sᵀk are jointly Gaussian and uncorrelated (q⊥ ⊥ k), hence independent. The sign of one carries no information about the other, so the term averages to 0. Left histogram: symmetric about 0, running mean → 0.
- First term: sᵀk ~ N(0, ‖k‖²), so |sᵀk| is half-normal with mean √(2/π)·‖k‖. Right histogram with the exact half-normal density (white curve) and its mean.
So E[sign(sᵀk)·sᵀq] = √(2/π)·⟨q,k⟩/‖k‖. Multiplying by √(π/2)·‖k‖ gives exactly ⟨q,k⟩. That is the whole reason for the √(π/2) constant and for storing ‖k‖.

The error. Each row X = √(π/2)‖k‖·sign(sᵀk)·sᵀq has E[X²] = (π/2)‖k‖²·E[(sᵀq)²] = (π/2)‖q‖²‖k‖², because sign² = 1. So Var[X] = (π/2)‖q‖²‖k‖² − ⟨q,k⟩², and averaging m rows divides by m. This variance formula is derived by us from the paper's moment computation; the paper doesn't print it in this form. It is at most (π/(2m))·‖q‖²‖k‖².

Plot (top right): log–log, variance vs m for unit q, k at the angle you set in the sandbox. Dots are a live simulation (300 trials per m); lines are the formulas. Teal = QJL with 1 bit per row; yellow = unquantized JL with a float per row, variance (1 + cos²θ)/m; dashed = the π/(2m) bound. Both have slope −1. Quantizing each row all the way down to 1 bit costs at most a constant: at θ = 90° the ratio is exactly π/2; when q is near k, QJL can even beat float JL because the stored norm ‖k‖ is exact.

Lemma 3.5 (concentration): if m ≥ (4/3)·(1+ε)/ε²·log(2/δ), then |Prod_QJL − ⟨q,k⟩| ≤ ε‖q‖‖k‖ with probability ≥ 1 − δ. (Note: the paper's printed tail bound is missing a minus sign in the exponent; it's a typo, the intended bound is exp(−·).)
Theorem 3.6: if all ‖k_i‖, ‖q‖ ≤ r and m ≥ 2r²ε⁻² log n, all softmax scores are within a (1 ± 3ε) factor w.h.p. So m ≈ ε⁻² log n bits per key, independent of the head dimension d.

The catch, and the cliffhanger: error² ∝ 1/m, and m is the number of bits. Halving the error costs 4× the bits. That's polynomial decay. Information theory allows much better: can the error fall exponentially in the number of bits (like 4^(−b))? That's what TurboQuant does next.`,
  mount(body, ctx) {
    const rng = new RNG(1111);
    const state = { q: [0.62, 0.85], k: [1.05, 0.18] };
    const smp = new DirSampler(rng, MAX_SAMPLES + 50);
    let tPerp = [];
    let tAbs = [];

    const sdot = (s, v) => s[0] * v[0] + s[1] * v[1];
    const nk = () => Math.hypot(...state.k);
    const ipTrue = () => sdot(state.q, state.k);
    const parts = () => {
      const c = ipTrue() / nk() ** 2;
      const qpar = [c * state.k[0], c * state.k[1]];
      return { c, qpar, qperp: [state.q[0] - qpar[0], state.q[1] - qpar[1]] };
    };
    const sgn = (v) => (v >= 0 ? 1 : -1);
    function recomputeAll() {
      const { qperp } = parts();
      tPerp = smp.s.map((s) => sgn(sdot(s, state.k)) * sdot(s, qperp));
      tAbs = smp.s.map((s) => Math.abs(sdot(s, state.k)));
    }

    const rPerp = readout('avg $\\sign(s^\\top k)\\,s^\\top q^\\perp$');
    const rAbs = readout('avg $|s^\\top k|$ / $\\sqrt{2/\\pi}\\,\\norm{k}$');
    const rEst = readout('QJL est. / $\\langle q,k\\rangle$');
    for (const r of [rAbs, rEst]) r.el.querySelector('.v').style.fontSize = '0.98em';

    // ---------- sandbox + two histograms ----------
    const v = viz('', 'drag $q$ and $k$');
    let view;
    const cv = createCanvas(v, (g, w, h) => {
      const planeH = Math.round(h * 0.6);
      const plane = { x: 0, y: 0, w, h: planeH };
      view = new View2D({ w, h: planeH, xmin: -1.45, xmax: 1.45, ymin: -1.25, ymax: 1.25, pad: 10 });
      const { q, k } = state;
      const { qpar, qperp } = parts();
      clipRect(g, plane, () => {
        gridLines(g, view, 0.5);
        halfPlane(g, view, k, 'accent', 0.07);
        dirLine(g, view, k, { color: 'accent', alpha: 0.3, width: 1 });
        dirLine(g, view, [-k[1], k[0]], { color: 'accent', alpha: 0.35, width: 1, dash: [6, 5] });
        // newest random direction
        const S = smp.s;
        if (S.length) {
          const s = S[S.length - 1];
          dirLine(g, view, s, { color: 'ink-3', alpha: 0.45, width: 1 });
          const sc = Math.min(0.5, 1.2 / Math.hypot(s[0], s[1]));
          arrow(g, view.sx(0), view.sy(0), view.sx(sc * s[0]), view.sy(sc * s[1]), { color: 'ink-2', width: 1.8, head: 8 });
          label(g, 's', view.sx(sc * s[0]) + 7, view.sy(sc * s[1]) - 7, { color: 'ink-2', size: 14, weight: 600 });
        }
        // decomposition of q
        arrow(g, view.sx(0), view.sy(0), view.sx(qperp[0]), view.sy(qperp[1]), { color: 'violet', width: 2, alpha: 0.35, head: 9 });
        arrow(g, view.sx(0), view.sy(0), view.sx(qpar[0]), view.sy(qpar[1]), { color: 'blue', width: 3, dash: [7, 5], head: 10 });
        arrow(g, view.sx(qpar[0]), view.sy(qpar[1]), view.sx(q[0]), view.sy(q[1]), { color: 'violet', width: 3, head: 10 });
        // right-angle mark
        const kn = nk();
        const u = [k[0] / kn, k[1] / kn];
        const pn = Math.hypot(...qperp) || 1;
        const pv = [qperp[0] / pn, qperp[1] / pn];
        const r = 0.08;
        if (pn > 0.12) {
          g.save();
          g.strokeStyle = 'rgba(195,203,219,0.6)';
          g.lineWidth = 1.2;
          g.beginPath();
          const a = [qpar[0] - r * u[0], qpar[1] - r * u[1]];
          g.moveTo(view.sx(a[0]), view.sy(a[1]));
          g.lineTo(view.sx(a[0] + r * pv[0]), view.sy(a[1] + r * pv[1]));
          g.lineTo(view.sx(qpar[0] + r * pv[0]), view.sy(qpar[1] + r * pv[1]));
          g.stroke();
          g.restore();
        }
        arrow(g, view.sx(0), view.sy(0), view.sx(k[0]), view.sy(k[1]), { color: 'accent', width: 3.5 });
        arrow(g, view.sx(0), view.sy(0), view.sx(q[0]), view.sy(q[1]), { color: 'blue', width: 3.5 });
        circle(g, view.sx(0), view.sy(0), 3, { fill: 'ink-3' });
        label(g, 'k', view.sx(k[0]) + 8, view.sy(k[1]) - 10, { color: 'accent', size: 18, weight: 600 });
        label(g, 'q', view.sx(q[0]) + 8, view.sy(q[1]) - 10, { color: 'blue', size: 18, weight: 600 });
        label(g, 'q⊥', view.sx((qpar[0] + q[0]) / 2) + 8, view.sy((qpar[1] + q[1]) / 2), { color: 'violet', size: 15, weight: 600 });
        label(g, '⟨q,k⟩/‖k‖² · k', view.sx(qpar[0] / 2) + 4, view.sy(qpar[1] / 2) + 16, { color: 'blue', size: 13 });
      });
      line(g, 0, planeH, w, planeH, { color: 'line' });

      // two histograms side by side
      const gap = 18;
      const bw = (w - 3 * gap) / 2;
      const by = planeH + 8;
      const bh = h - planeH - 12;
      const pn2 = Math.hypot(...qperp);
      const Rp = 3.2 * pn2 * nk() + 0.2;
      const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
      const falling = smp.recent.filter((rr) => rr.age < 0.45);
      const skip = new Set(falling.map((rr) => rr.i));
      const landed = (arr) => (skip.size ? arr.filter((_, i) => !skip.has(i)) : arr);
      strip(g, { x: gap, y: by, w: bw, h: bh }, {
        lo: -Rp,
        hi: Rp,
        bins: 40,
        values: landed(tPerp),
        color: 'violet',
        drops: falling.map((rr) => ({ v: tPerp[rr.i], t: rr.age / 0.45 })),
        title: 'sign(sᵀk) · sᵀq⊥',
        markers: [
          { v: 0, color: 'ink', dash: [5, 4], label: '0' },
          { v: mean(tPerp), color: 'teal', label: 'avg' },
        ],
      });
      const K = nk();
      const Ra = 3.3 * K;
      const target = Math.sqrt(2 / Math.PI) * K;
      strip(g, { x: 2 * gap + bw, y: by, w: bw, h: bh }, {
        lo: 0,
        hi: Ra,
        bins: 40,
        values: landed(tAbs),
        color: 'accent',
        pdf: (x) => 2 * normalPdf(x, K),
        drops: falling.map((rr) => ({ v: tAbs[rr.i], t: rr.age / 0.45 })),
        title: '|sᵀk|  (half-normal)',
        markers: [
          { v: target, color: 'ink', dash: [5, 4], label: '√(2/π)‖k‖' },
          { v: mean(tAbs), color: 'teal', label: 'avg' },
        ],
      });
    });

    function refresh() {
      const n = smp.s.length;
      const mp = n ? tPerp.reduce((a, b) => a + b, 0) / n : null;
      const ma = n ? tAbs.reduce((a, b) => a + b, 0) / n : null;
      rPerp.set(mp == null ? '–' : mp.toFixed(3), 'violet');
      rAbs.set(`${ma == null ? '–' : ma.toFixed(3)} / ${(Math.sqrt(2 / Math.PI) * nk()).toFixed(3)}`, 'accent');
      if (n) {
        let acc = 0;
        for (const s of smp.s) acc += sgn(sdot(s, state.k)) * sdot(s, state.q);
        rEst.set(`${((SQ * nk() * acc) / n).toFixed(3)} / ${ipTrue().toFixed(3)}`, 'teal');
      } else rEst.set('–');
      cv.request();
    }

    // ---------- variance vs m (unit vectors at the sandbox angle) ----------
    const sim = { theta: 0, stats: [], rng: new RNG(31337), cursor: 0 };
    const angle = () => {
      const a = Math.atan2(state.q[1], state.q[0]) - Math.atan2(state.k[1], state.k[0]);
      return Math.abs(Math.atan2(Math.sin(a), Math.cos(a)));
    };
    function resetSim() {
      sim.theta = angle();
      sim.stats = MS.map((m) => ({ m, n: 0, eQ: 0, eJ: 0 }));
      sim.cursor = 0;
    }
    function stepSim(budget) {
      const c = Math.cos(sim.theta);
      const sn = Math.sin(sim.theta);
      const r = sim.rng;
      while (budget > 0 && sim.cursor < MS.length) {
        const st = sim.stats[sim.cursor];
        const m = st.m;
        let aq = 0;
        let aj = 0;
        for (let i = 0; i < m; i++) {
          const s0 = r.normal();
          const s1 = r.normal();
          const pq = s0 * c + s1 * sn;
          aq += sgn(s0) * pq;
          aj += s0 * pq;
        }
        st.eQ += ((SQ * aq) / m - c) ** 2;
        st.eJ += (aj / m - c) ** 2;
        st.n++;
        budget -= m;
        if (st.n >= T_PER_M) sim.cursor++;
      }
    }
    const vP = viz('', 'variance vs $m$ · unit $q,k$ at the sandbox angle');
    Object.assign(vP.style, { height: 'auto', flex: '1', minHeight: '220px' });
    const rRatio = readout('QJL ÷ JL variance');
    const cvP = createCanvas(vP, (g, w, h) => {
      const c2 = Math.cos(sim.theta) ** 2;
      const p = new Plot(g, {
        x: 0,
        y: 26,
        w,
        h: h - 26,
        xmin: 1,
        xmax: 1024,
        ymin: 1e-4,
        ymax: 10,
        xlog: true,
        ylog: true,
        xticks: [1, 4, 16, 64, 256, 1024],
        xlabel: 'm  (bits per key, plus one float for ‖k‖)',
        ylabel: 'Var / ‖q‖²‖k‖²',
        margin: { l: 62, b: 40, t: 4, r: 12 },
      });
      p.frame();
      p.fn((m) => Math.PI / (2 * m), { color: 'ink-2', dash: [6, 5], width: 1.5 });
      p.fn((m) => (1 + c2) / m, { color: 'yellow', width: 2, alpha: 0.8 });
      p.fn((m) => (Math.PI / 2 - c2) / m, { color: 'teal', width: 2 });
      const done = sim.stats.filter((s) => s.n >= 30);
      p.points(
        done.map((s) => s.m),
        done.map((s) => s.eJ / s.n),
        { color: 'yellow', r: 4.5, alpha: 0.9 },
      );
      p.points(
        done.map((s) => s.m),
        done.map((s) => s.eQ / s.n),
        { color: 'teal', r: 4.5, alpha: 0.95 },
      );
      p.legend(
        [
          { label: 'QJL, 1 bit/row: (π/2 − cos²θ)/m', color: 'teal' },
          { label: 'JL, float/row: (1 + cos²θ)/m', color: 'yellow' },
          { label: 'bound π/(2m)', color: 'ink-2', dash: [6, 5] },
        ],
        'tr',
      );
      label(g, `θ = ${((sim.theta * 180) / Math.PI).toFixed(0)}° · dots: live simulation, ${T_PER_M} trials per m`, p.px + 8, p.py + p.ph - 12, {
        size: 12,
        color: 'ink-3',
      });
      label(g, 'slope −1', p.tx(3), p.ty(Math.PI / 2 / 3) - 16, { size: 12.5, color: 'ink-2' });
      rRatio.set(`${((Math.PI / 2 - c2) / (1 + c2)).toFixed(2)}×`, 'teal');
    });

    dragQK(cv, () => view, state, () => {
      recomputeAll();
      refresh();
      resetSim();
      cvP.request();
    }, { rmax: 1.2 });

    let started = false;
    ctx.onEnter(() => {
      if (!started) {
        started = true;
        resetSim();
      }
      refresh();
      cvP.request();
    });
    ctx.loop((dt) => {
      if (!started) return;
      if (sim.cursor < MS.length) {
        stepSim(50000);
        cvP.redraw();
      }
      if (smp.s.length < MAX_SAMPLES && smp.pending === 0) smp.add(smp.s.length < 24 ? 1 : 20);
      if (!smp.busy) return;
      const n = smp.step(dt, 3);
      if (n) {
        const { qperp } = parts();
        for (let i = smp.s.length - n; i < smp.s.length; i++) {
          const s = smp.s[i];
          tPerp.push(sgn(sdot(s, state.k)) * sdot(s, qperp));
          tAbs.push(Math.abs(sdot(s, state.k)));
        }
        refresh();
      }
      cv.redraw();
    });

    const restart = button('restart', () => {
      smp.reset();
      recomputeAll();
      refresh();
    });

    const left = el(
      'div.col',
      {},
      v,
      el(
        'div.row',
        { style: { gap: '10px', flexWrap: 'nowrap' } },
        el('div.readouts', { style: { flex: '1', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.25fr) minmax(0, 1.25fr)', gap: '8px' } }, rPerp.el, rAbs.el, rEst.el),
        restart,
      ),
    );
    Object.assign(v.style, { height: 'auto', flex: '1', minHeight: '340px' });

    const right = el(
      'div.col',
      {},
      el(
        'div.card',
        { 'data-reveal': '1', style: { gap: '2px', padding: '10px 16px' } },
        eq('q=\\tfrac{\\ip{q}{k}}{\\norm{k}^2}\\,k+q^{\\perp}'),
        eq(
          '\\E\\big[\\sign(s^\\top k)\\,s^\\top q\\big]=\\tfrac{\\ip{q}{k}}{\\norm{k}^2}\\underbrace{\\E|s^\\top k|}_{\\sqrt{2/\\pi}\\,\\norm{k}}+\\underbrace{\\E\\big[\\sign(s^\\top k)\\,s^\\top q^{\\perp}\\big]}_{0}',
        ),
        eq('\\Var=\\frac{\\frac{\\pi}{2}\\norm{q}^2\\norm{k}^2-\\ip{q}{k}^2}{m}\\ \\le\\ \\frac{\\pi}{2m}\\norm{q}^2\\norm{k}^2'),
      ),
      vP,
      el(
        'div.card',
        { 'data-reveal': '2', style: { gap: '4px', padding: '10px 16px' } },
        eq('m\\ge\\tfrac43\\,\\tfrac{1+\\varepsilon}{\\varepsilon^2}\\log\\tfrac2\\delta\\ \\Rightarrow\\ \\big|\\mathrm{Prod}_{\\rm QJL}-\\ip{q}{k}\\big|\\le\\varepsilon\\norm{q}\\norm{k}\\ \\text{w.p. }1-\\delta'),
        el(
          'div.row',
          { style: { justifyContent: 'space-between', gap: '6px 16px', fontSize: '0.92em' } },
          rich('span', 'error$^2\\propto 1/m$ = 1/bits: <b>polynomial</b>'),
          rich('span.muted', 'Thm 3.6: $m\\approx\\varepsilon^{-2}\\log n$ bits/key, no $d$'),
        ),
        rich('div.c-yellow', 'Can error fall <b>exponentially</b> in bits?', { style: { fontWeight: '600', fontSize: '1.02em' } }),
      ),
    );
    body.append(el('div.stage.even', {}, left, right));
    // the ratio readout lives next to the plot's legend
    vP.append(el('div', { style: { position: 'absolute', left: '78px', bottom: '72px', width: '150px' } }, rRatio.el));
    recomputeAll();
    refresh();
  },
};

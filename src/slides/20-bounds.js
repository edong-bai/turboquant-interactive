import { el, eq, segmented, readout, viz, legend } from '../lib/ui.js';
import { createCanvas, label } from '../lib/canvas.js';
import { Plot } from '../lib/plot.js';
import { bounds } from '../lib/math.js';
import { sim as getSim, BMAX, PAPER, D, measuredDmse, measuredQjl, ipErrors, moments } from './_shared-tq.js';
import { drawDmsePlot, band, square, disc, progressBadge } from './_shared-tq-draw.js';

// Gap between TurboQuant_mse and the lower bound 4^{-b} (paper: b = 1..4, and √3π/2 as b → ∞).
const GAP_INF = (Math.sqrt(3) * Math.PI) / 2;
const INF = 6;

export default {
  id: 'bounds',
  chapter: 'TurboQuant',
  kicker: 'How good is it?',
  title: 'Within a small constant of any quantizer',
  notes: `Theorem 3 (lower bound): for ANY randomized quantizer using b bits per coordinate there is a hard unit vector x with D_mse ≥ 4^{-b}, and a hard unit query y with D_prod ≥ (1/d)·4^{-b}. TurboQuant is within a factor √3π/2 ≈ 2.7 of that, at every b.

LEFT, why it costs 4^{-b}: intuition by counting volume. b·d bits = 2^{bd} cells. If each cell had radius ρ, they cover volume ~ 2^{bd}·ρ^d (in units of the sphere's own "volume"), which must be ≳ 1, so ρ ≳ 2^{-b}, and the typical squared error ≳ ρ² = 4^{-b}. The real proof uses the Shannon lower bound D ≥ (d/2πe)·2^{(2/d)(h(x) − B)} with the differential entropy of the uniform distribution on the sphere computed via Stirling; it is asymptotic in d. (Proof detail, not on screen: the bound must hold for randomized quantizers too. The paper uses Yao's minimax principle: the best randomized quantizer's worst-case error is at least the best deterministic quantizer's error on the hardest input distribution, which is at least its error on x uniform on the sphere. So it suffices to bound the uniform-on-sphere case.) For D_prod: the d coordinate directions e_j split the MSE (Σ_j E⟨e_j, x − x̃⟩² = D_mse), so by pigeonhole some axis carries ≥ D_mse/d ≥ (1/d)·4^{-b}.
The plot: TurboQuant_mse (yellow: the paper's Theorem 1 numbers 0.36, 0.117, 0.03, 0.009; teal: our live d = 128 simulation) inside the band between 4^{-b} and 2.72·4^{-b}. Pick b to read the gap: ratio 1.45, 1.88, 2.21, 2.43 → 2.72 (paper); our d = 128 simulation gives 1.44, 1.86, 2.18, 2.40, 2.49 (b = 5). In bits per coordinate (log₄ of the ratio): 0.27, 0.45, 0.57, 0.64 → 0.72. So TurboQuant uses at most ~0.7 extra bits per coordinate compared to an impossible ideal quantizer.

RIGHT, why TurboQuant beats QJL at the same bit budget. QJL with m = b·d sign bits has D_prod ≤ π/(2bd): the error falls only like 1/b (polynomial). TurboQuant_prod with b bits per coordinate has D_prod ≤ (√3π²/d)·4^{-b}: exponential in b. The plot shows d·D_prod. At b = 1 they are the same algorithm (TurboQuant_prod with 1 bit is QJL with m = d). Live simulation at d = 128 (dots; query ⟂ x, which is QJL's worst case): QJL measured ≈ 1.67, 0.82, 0.53, 0.39, 0.32 vs theory π/(2b) = 1.57, 0.79, 0.52, 0.39, 0.31; TurboQuant_prod measured ≈ 1.56, 0.56, 0.17, 0.052, 0.015 (≈ (π/2)·D_mse(b−1); paper Thm 2 text: 1.57, 0.56, 0.18, 0.047). Same bits, 20× less error at b = 5. The reason: QJL spends every bit on 1-bit sketches of the whole vector; TurboQuant spends them on a fine scalar grid and keeps only one sketch for the tiny residual.

Simulation noise: about ±5% on the QJL dots (640 vectors, one query each).`,
  mount(body, ctx) {
    const sim = getSim();
    const state = { gapB: 2 };

    // ---------- (b) the gap ----------
    const vg = viz('', '□ paper (Thm 1) · ● live sim $d=128$');
    const gapPlot = createCanvas(vg, (g, w, h) => drawDmsePlot(g, w, h, { sim, highlight: state.gapB <= BMAX ? state.gapB : null, paper: 'thm1', annotate: false, top: 34 }));
    const rRatio = readout('paper: ratio to $4^{-b}$');
    const rBits = readout('gap, bits / coord');
    const rLive = readout('live ratio, $d=128$');
    const updateGap = () => {
      const b = state.gapB;
      const live = b <= BMAX && sim.count > 0 ? measuredDmse(b, sim) * 4 ** b : null;
      const paper = b === INF ? GAP_INF : PAPER.gap[b] ?? null;
      const ratio = paper ?? live;
      rRatio.set(paper ? `×${paper.toFixed(2)}` : 'not stated', paper ? 'yellow' : 'ink-3');
      rBits.set(ratio ? (Math.log(ratio) / Math.log(4)).toFixed(2) : '–', 'ink');
      rLive.set(live ? `×${live.toFixed(2)}` : '–', 'teal');
    };
    const segGap = segmented({
      label: '$b$',
      options: [1, 2, 3, 4, 5, { label: '$\\infty$', value: INF }],
      value: state.gapB,
      onChange: (v) => {
        state.gapB = v;
        updateGap();
        gapPlot.request();
      },
    });

    // ---------- (c) TurboQuant vs QJL at equal bits ----------
    const vq = viz('', 'same bits · dots: live sim $d=128$ · dashed: bounds');
    const qplot = createCanvas(vq, (g, w, h) => {
      if (w < 40 || h < 40) return;
      const p = new Plot(g, {
        x: 0,
        y: 0,
        w,
        h,
        xmin: 0.6,
        xmax: BMAX + 0.4,
        ymin: 7e-4,
        ymax: 12,
        ylog: true,
        xticks: [1, 2, 3, 4, 5],
        xfmt: (v) => `b=${v}`,
        ylabel: 'd · D_prod',
        margin: { l: 58, r: 14, t: 30, b: 30 },
      });
      p.frame();
      const tq = (b) => D * bounds.prodUpper(b, D);
      const qjl = (b) => (D * bounds.qjlVar(b * D));
      const lower = (b) => D * bounds.prodLower(b, D);
      band(p, () => 7e-4, lower, { color: 'ink-3', alpha: 0.12 });
      p.fn(qjl, { color: 'red', width: 2, dash: [6, 4] });
      p.fn(tq, { color: 'teal', width: 2, dash: [6, 4] });
      p.fn(lower, { color: 'blue', width: 1.5, dash: [6, 4] });
      label(g, 'QJL, m = b·d:  π/(2b)', p.tx(4), p.ty(qjl(4)) - 18, { color: 'red', size: 12, align: 'center' });
      label(g, 'TurboQuant: √3π²·4⁻ᵇ', p.tx(2.2), p.ty(tq(2.2)) - 8, { color: 'teal', size: 12, align: 'left' });
      label(g, 'impossible: < 4⁻ᵇ', p.tx(1.0), p.ty(lower(1.0)) + 26, { color: 'ink-3', size: 12, align: 'left' });
      for (let b = 1; b < PAPER.prodThm2.length; b++) square(p, b, PAPER.prodThm2[b], { color: 'yellow', r: 5 });
      if (sim.count > 0) {
        for (let b = 1; b <= BMAX; b++) disc(p, b, D * moments(ipErrors('prod', b, 0, sim)).ms, { color: 'teal', r: 5 });
      }
      if (sim.qjlCount > 0) {
        for (let b = 1; b <= BMAX; b++) disc(p, b + 0.06, measuredQjl(b, sim), { color: 'red', r: 4 });
      }
      progressBadge(p, Math.min(sim.progress, sim.qjlCount / sim.N));
    });

    const grow = (node, min) => Object.assign(node.style, { flex: '1 1 0', height: 'auto', minHeight: min });
    grow(vg, '200px');
    grow(vq, '260px');

    const colB = el(
      'div.col',
      {},
      el(
        'div.card',
        { 'data-reveal': '1', style: { padding: '8px 14px', gap: '0', fontSize: '0.86em' } },
        eq('\\textbf{Thm 3:}\\ \\ \\Dmse\\ge 4^{-b},\\quad \\Dprod\\ge\\tfrac1d\\,4^{-b}'),
        eq('2^{bd}\\,\\text{cells}\\cdot\\rho^{d}\\gtrsim 1\\ \\Rightarrow\\ \\rho\\gtrsim 2^{-b}\\ \\Rightarrow\\ D\\gtrsim\\rho^2=4^{-b}'),
        eq('\\textstyle\\sum_j \\E\\ip{e_j}{x-\\tilde x}^2=\\Dmse\\ \\Rightarrow\\ \\exists j:\\ \\ge\\tfrac1d\\Dmse'),
      ),
      vg,
      el('div.row', {}, segGap.el),
      el('div.readouts', {}, rRatio.el, rBits.el, rLive.el),
    );
    const colC = el(
      'div.col',
      {},
      vq,
      legend([
        { label: 'TurboQuant$_{\\rm prod}$', color: 'teal' },
        { label: 'QJL, $m=bd$', color: 'red' },
        { label: 'paper, Thm 2', color: 'yellow' },
        { label: '$4^{-b}$', color: 'blue', dash: true },
      ]),
      el(
        'div.card',
        { 'data-reveal': '3', style: { padding: '8px 14px', gap: '0', fontSize: '0.86em' } },
        eq('\\text{QJL},\\ m=bd:\\ \\ \\Dprod\\le\\tfrac{\\pi}{2bd}'),
        eq('\\text{TurboQuant}:\\ \\ \\Dprod\\le\\tfrac{\\sqrt3\\,\\pi^2}{d}\\,4^{-b}'),
      ),
    );
    body.append(el('div.stage.even', {}, colB, colC));
    updateGap();

    let seen = -1;
    ctx.onEnter(() => {
      sim.qjlWanted = true;
      sim.done = false;
      gapPlot.request();
      qplot.request();
    });
    ctx.loop(() => {
      if (!sim.done) sim.step(5);
      const key = sim.count + sim.qjlCount * 10000;
      if (key !== seen) {
        seen = key;
        gapPlot.redraw();
        qplot.redraw();
        updateGap();
      }
    });
  },
};

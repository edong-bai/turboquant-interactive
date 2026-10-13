import { el, eq, viz, texify } from '../lib/ui.js';
import { createCanvas, alpha, color } from '../lib/canvas.js';
import * as M from '../lib/math.js';

// Each step of the chain: what it is, the problem it fixes (or exposes), a tiny equation, and where we saw it.
const STEPS = [
  {
    t: 'Requirements',
    s: 'online · vectorizable · zero overhead',
    kind: 'why',
    fix: 'Decoding is memory-bound: every new token re-reads the whole KV cache. We need a quantizer that works per vector, on the fly, on a GPU, and stores no extra constants.',
    eq: 'Q:\\R^d\\to\\{0,1\\}^{b\\cdot d}\\ \\text{(worst-case } x\\text{)}',
    links: [
      ['requirements', 'Requirements'],
      ['metrics', 'Two errors'],
    ],
  },
  {
    t: 'Naive rounding',
    s: 'fails on outliers',
    kind: 'bad',
    fix: 'Per-block scale and zero point cost 1–2 extra bits per number (QJL’s estimate), and one large coordinate stretches the grid for all the others. Adversarial inputs break it.',
    eq: '\\tilde x_j = s\\cdot\\mathrm{round}(x_j/s) + z',
    links: [
      ['naive', 'Naive rounding'],
      ['naive-fails', 'Where it fails'],
    ],
  },
  {
    t: 'Keys need only $\\ip{q}{k}$',
    s: 'a weaker target',
    kind: 'qjl',
    fix: 'Attention never needs $k$ back, only its inner products with future queries. So we may store any $f(k)$ that preserves inner products, which is exactly what lets us use random projections.',
    eq: '\\mathrm{softmax}\\big(\\ip{q}{k_1},\\dots,\\ip{q}{k_n}\\big)',
    links: [['jl', 'Only inner products → JL']],
  },
  {
    t: 'JL projection',
    s: 'unbiased, but 16 bits per row',
    kind: 'qjl',
    fix: 'A random Gaussian sketch, drawn before any data, preserves every inner product: unbiased, random (not systematic) error shrinking like $1/\\sqrt m$, with $m$ independent of $d$. It saves memory only by dropping rows, and each row still costs 16 bits.',
    eq: '\\E\\,\\tfrac1m\\ip{Sq}{Sk} = \\ip{q}{k}',
    links: [['jl', 'Only inner products → JL']],
  },
  {
    t: '1-bit QJL',
    s: 'unbiased · zero overhead',
    kind: 'qjl',
    fix: 'Keep only the signs of $Sk$ (plus $\\norm{k}$) and leave the query unquantized. Still unbiased, no per-block constants, and at the same bits, many sign rows beat a few float rows by ≈3–7× in error. But variance only falls like $1/m$: polynomial in the bit budget.',
    eq: '\\begin{gathered}\\tfrac{\\sqrt{\\pi/2}}{m}\\norm{k}\\ip{Sq}{\\sign(Sk)}\\\\ \\Var\\le\\tfrac{\\pi}{2m}\\norm{q}^2\\norm{k}^2\\end{gathered}',
    links: [
      ['qjl', 'QJL'],
      ['qjl-why', 'Why it is unbiased'],
    ],
  },
  {
    t: 'Random rotation',
    s: 'worst case → average case',
    kind: 'tq',
    fix: 'The lesson of QJL’s projection: randomizing first makes the coordinate distribution known, the error random instead of systematic, and the energy evenly spread. A random rotation does the same while keeping lengths: whatever $x$ is (even adversarial), $\\Pi x$ is uniform on the sphere.',
    eq: '\\Pi x\\sim\\mathrm{Unif}(\\S^{d-1})',
    links: [
      ['why-project', 'Why randomness helps'],
      ['rotation', 'Rotate first'],
    ],
  },
  {
    t: 'Known density',
    s: 'Beta $\\approx\\mathcal N(0,1/d)$',
    kind: 'tq',
    fix: 'Every rotated coordinate has the same known density, and in high dimension the coordinates are nearly independent. So one scalar quantizer per coordinate is enough.',
    eq: 'f(x)\\propto(1-x^2)^{\\frac{d-3}{2}}\\;\\to\\;\\mathcal N(0,\\tfrac1d)',
    links: [
      ['sphere-d3', 'Sphere, d = 3'],
      ['sphere-d4', 'Sphere, d = 4'],
      ['concentration', 'Concentration'],
    ],
  },
  {
    t: 'Lloyd–Max codebook',
    s: '$\\Dmse\\le2.72\\cdot4^{-b}$',
    kind: 'tq',
    fix: 'The optimal scalar codebook for that density is computed once per $b$. Error now falls exponentially in bits, like $4^{-b}$, with no data-dependent constants.',
    eq: '\\Dmse\\le\\tfrac{\\sqrt3\\pi}{2}\\,4^{-b}\\approx 2.72\\cdot4^{-b}',
    links: [
      ['lloyd-max', 'Lloyd–Max'],
      ['tq-mse', 'TurboQuant_mse'],
    ],
  },
  {
    t: 'Biased inner products',
    s: 'shrink by $1-D$',
    kind: 'bad',
    fix: 'MSE-optimal reconstructions are pulled toward 0, so inner products come out systematically too small: every attention logit shrinks, which flattens the softmax. At $b=1$ the factor is exactly $2/\\pi$; the bias shrinks as $b$ grows.',
    eq: '\\begin{gathered}\\E\\ip{x}{\\tilde x}=1-\\Dmse\\\\ b{=}1:\\ \\E\\ip{y}{\\tilde x}=\\tfrac2\\pi\\ip{y}{x}\\end{gathered}',
    links: [['mse-bias', 'The bias']],
  },
  {
    t: 'QJL on the residual',
    s: 'unbiased · $4^{-b}$',
    kind: 'tq',
    fix: 'Spend $b-1$ bits on TurboQuant_mse, then 1 bit of QJL on the residual $r = x-\\tilde x$. Unbiased again, and the error keeps the exponential rate.',
    eq: '\\Dprod\\le\\tfrac{\\sqrt3\\pi^2\\norm{y}^2}{d}\\,4^{-b}',
    links: [
      ['tq-prod', 'TurboQuant_prod'],
      ['mse-vs-prod', 'mse vs prod'],
    ],
  },
  {
    t: 'Near-optimal',
    s: 'within $2.7\\times$ of $4^{-b}$',
    kind: 'tq',
    fix: 'Shannon lower bound: no quantizer, randomized or not, beats $4^{-b}$ on its worst input. TurboQuant is within $\\approx 2.7\\times$ (1.45× at $b=1$), and its error falls like $4^{-b}$ where QJL’s falls like $1/b$.',
    eq: '4^{-b}\\le\\Dmse\\le 2.72\\cdot4^{-b}',
    links: [['bounds', 'Lower bounds']],
  },
  {
    t: 'Practice',
    s: 'outlier split · 32 @ 3 + 96 @ 2',
    kind: 'practice',
    fix: 'A few key channels hold most of the energy, so run two independent instances and give the outliers more bits (paper: 32 channels at 3 bits + 96 at 2, printed as 2.5 bits but actually 2.25). Quality-neutral KV cache at 3.5 bits, and near-zero indexing time for vector search.',
    eq: '\\bar b=\\tfrac{n_o b_{\\text{hi}}+(128-n_o)\\,b_{\\text{lo}}}{128}',
    links: [
      ['outliers', 'Outlier channels'],
      ['results-kv', 'KV cache results'],
      ['results-nn', 'Nearest neighbors'],
    ],
  },
];

const KIND = {
  why: { c: 'ink-2', name: 'setup' },
  bad: { c: 'red', name: 'breaks' },
  qjl: { c: 'blue', name: 'QJL' },
  tq: { c: 'teal', name: 'TurboQuant' },
  practice: { c: 'yellow', name: 'practice' },
};

export default {
  id: 'recap',
  chapter: 'Recap',
  kicker: 'Recap',
  title: 'Every step was forced',
  notes: `The whole talk as one chain. Each box is a decision, and each one is forced by a failure of the box before it. Hover or click any box to show the problem it fixes, its equation, and links back to the slides where we built it. The chain auto-advances until you touch it.
1. Requirements: online (no calibration pass), vectorizable (GPU-friendly), zero overhead (no per-block scale or zero point), worst-case inputs.
2. Naive rounding fails: block constants cost 1–2 extra bits per number (QJL's claim), and outliers stretch the grid.
3. For keys we only need ⟨q,k⟩, a weaker target than reconstruction, so any inner-product-preserving map will do.
4. A JL projection is such a map: unbiased, random error, m independent of d, drawn before any data. But each row costs 16 bits, and saving memory means dropping rows.
5. Keep 1 bit per row and use many more rows (QJL): still unbiased (asymmetric, query unquantized), no overhead, ≈3–7× less error than floats at the same bits. But error falls only like 1/m, polynomial in bits.
6. Why randomness helped: it made the coordinate distribution known, the error random, the energy spread. A random rotation does the same while preserving lengths: worst case becomes average case.
7. Rotated coordinates follow a known Beta-type density, approximately N(0, 1/d) in high dimension, and are nearly independent.
8. So quantize each coordinate with the Lloyd–Max codebook for that density: D_mse ≤ (√3π/2)·4^{-b} ≈ 2.72·4^{-b}. That's exponential in bits.
9. But MSE-optimal reconstructions are biased for inner products: E⟨x, x̃⟩ = 1 − D_mse (our derivation), exactly 2/π at b = 1 (paper). Shrunk logits flatten the softmax.
10. Fix: (b−1)-bit TurboQuant_mse plus 1-bit QJL on the residual. Unbiased, D_prod ≤ (√3π²/d)·4^{-b}.
11. Lower bound (Shannon, extended to randomized quantizers via Yao's principle): no quantizer beats 4^{-b} (MSE) or (1/d)·4^{-b} (inner product). TurboQuant is within about 2.7× (1.45× at b = 1), and decays like 4^{-b} where QJL decays like 1/b.
12. In practice: outlier channels get their own instance with more bits. Paper's split 32 @ 3 + 96 @ 2 (printed as 2.5 bits, actually 2.25); quality-neutral LongBench at 3.5; essentially zero indexing time for nearest-neighbor search.
Color code: grey = setup, red = where an approach breaks, blue = QJL (2024), teal = TurboQuant (2025), yellow = practice.
References at the bottom link to both arXiv papers.`,
  mount(body, ctx) {
    const state = { active: 0, pinned: false, hoverIdx: null, t: 0, lastStep: 0, flow: 0 };

    const v = viz('', '');
    v.style.flex = '1 1 0';
    v.style.height = 'auto';
    v.style.minHeight = '420px';

    // Node buttons positioned over the canvas.
    const nodes = STEPS.map((st, i) => {
      const k = KIND[st.kind];
      const b = el(
        'button',
        {
          type: 'button',
          style: {
            position: 'absolute',
            zIndex: '1',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'flex-start',
            justifyContent: 'center',
            gap: '3px',
            padding: '8px 12px',
            borderRadius: '12px',
            background: 'var(--panel)',
            border: '1px solid var(--line-2)',
            borderLeft: `4px solid var(--${k.c})`,
            color: 'var(--ink)',
            textAlign: 'left',
            cursor: 'pointer',
            transition: 'transform 0.2s, box-shadow 0.2s, border-color 0.2s, opacity 0.3s',
            overflow: 'hidden',
          },
          onmouseenter: () => {
            state.hoverIdx = i;
            show(i);
          },
          onmouseleave: () => {
            state.hoverIdx = null;
            show(state.active);
          },
          onfocus: () => show(i),
          onclick: () => {
            state.pinned = true;
            state.active = i;
            show(i);
          },
        },
        el(
          'div',
          { style: { display: 'flex', gap: '8px', alignItems: 'baseline', width: '100%' } },
          el('span', { style: { fontFamily: 'var(--mono)', fontSize: '0.72em', color: `var(--${k.c})` } }, String(i + 1).padStart(2, '0')),
          el('span', { style: { fontWeight: '600', fontSize: '0.92em', lineHeight: '1.2' }, html: texify(st.t) }),
        ),
        el('span', { style: { fontSize: '0.76em', color: 'var(--ink-3)', lineHeight: '1.25' }, html: texify(st.s) }),
      );
      v.append(b);
      return b;
    });

    // Detail card.
    const dNum = el('div.kicker');
    const dTitle = el('h3', { style: { fontSize: '1.15em', textTransform: 'none', letterSpacing: '0', color: 'var(--ink)' } });
    const dFix = el('p.note', { style: { lineHeight: '1.5' } });
    const dEq = el('div.eq', { style: { minHeight: '3.2em', fontSize: '1em' } });
    const dLinks = el('div', { style: { display: 'flex', flexDirection: 'column', gap: '6px' } });
    const detail = el(
      'div.panel',
      { style: { gap: '12px' } },
      dNum,
      dTitle,
      dFix,
      dEq,
      el('div.muted', { style: { fontSize: '0.75em', letterSpacing: '0.14em', textTransform: 'uppercase', marginTop: '4px' } }, 'Go to'),
      dLinks,
      el('div.muted', { style: { marginTop: 'auto', fontSize: '0.78em' } }, 'hover or click a step'),
    );

    let shown = -1;
    function show(i) {
      if (i === shown) return;
      shown = i;
      const st = STEPS[i];
      const k = KIND[st.kind];
      dNum.textContent = `Step ${i + 1} of ${STEPS.length} · ${k.name}`;
      dNum.style.color = `var(--${k.c})`;
      dTitle.innerHTML = texify(st.t);
      dFix.innerHTML = texify(st.fix);
      dEq.innerHTML = eq(st.eq).innerHTML;
      dLinks.innerHTML = '';
      for (const [id, name] of st.links) dLinks.append(el('a', { href: `#${id}`, style: { textDecoration: 'none' } }, `→ ${name}`));
      nodes.forEach((n, j) => {
        n.style.boxShadow = j === i ? `0 0 0 2px var(--${k.c}), 0 8px 24px rgba(0,0,0,0.35)` : 'none';
        n.style.transform = j === i ? 'translateY(-2px)' : 'none';
      });
    }

    // Layout: serpentine grid; returns node centers for the path.
    let centers = [];
    let geom = null;
    function layout(w, h) {
      const cols = w < 640 ? 2 : w < 980 ? 3 : 4;
      const rows = Math.ceil(STEPS.length / cols);
      const padX = 22;
      const padY = 22;
      const gapX = Math.max(28, w * 0.04);
      const nw = (w - 2 * padX - gapX * (cols - 1)) / cols;
      const cellH = (h - 2 * padY) / rows;
      const nh = Math.min(84, cellH * 0.62);
      centers = STEPS.map((_, i) => {
        const r = Math.floor(i / cols);
        let c = i % cols;
        if (r % 2 === 1) c = cols - 1 - c;
        const x = padX + c * (nw + gapX) + nw / 2;
        const y = padY + r * cellH + cellH / 2;
        return [x, y];
      });
      nodes.forEach((n, i) => {
        n.style.left = `${centers[i][0] - nw / 2}px`;
        n.style.top = `${centers[i][1] - nh / 2}px`;
        n.style.width = `${nw}px`;
        n.style.height = `${nh}px`;
      });
      // Path: straight within a row; at the row ends, a rounded U-turn outside the nodes.
      const pts = [];
      for (let i = 0; i < centers.length; i++) {
        pts.push(centers[i]);
        if (i + 1 < centers.length) {
          const r = Math.floor(i / cols);
          if (Math.floor((i + 1) / cols) !== r) {
            const [x0, y0] = centers[i];
            const [, y1] = centers[i + 1];
            const right = r % 2 === 0;
            const ex = right ? Math.min(w - 8, x0 + nw / 2 + gapX * 0.45) : Math.max(8, x0 - nw / 2 - gapX * 0.45);
            pts.push([ex, y0], [ex, y1]);
          }
        }
      }
      // Arc-length parameterization; remember where each node sits along the path.
      const seg = [0];
      for (let i = 1; i < pts.length; i++) seg.push(seg[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
      const nodeS = centers.map((c) => seg[pts.findIndex((p) => p === c)]);
      geom = { pts, seg, total: seg[seg.length - 1], nodeS, nw, nh };
    }

    function at(s) {
      const { pts, seg } = geom;
      let k = 1;
      while (k < seg.length - 1 && seg[k] < s) k++;
      const t = (s - seg[k - 1]) / Math.max(1e-6, seg[k] - seg[k - 1]);
      return [M.lerp(pts[k - 1][0], pts[k][0], t), M.lerp(pts[k - 1][1], pts[k][1], t)];
    }

    const cv = createCanvas(v, (g, w, h) => {
      layout(w, h);
      const { pts, total, nodeS } = geom;
      const sActive = nodeS[state.active];
      // base path
      g.save();
      g.lineJoin = 'round';
      g.lineCap = 'round';
      g.strokeStyle = color('line-2');
      g.lineWidth = 3;
      g.beginPath();
      pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
      g.stroke();
      // traveled part (up to the active node)
      g.strokeStyle = alpha('teal', 0.55);
      g.lineWidth = 4;
      g.beginPath();
      g.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length && geom.seg[i] <= sActive; i++) g.lineTo(pts[i][0], pts[i][1]);
      const [ax, ay] = at(sActive);
      g.lineTo(ax, ay);
      g.stroke();
      g.restore();
      // flowing particles along the full chain
      const spacing = 46;
      for (let s = state.flow % spacing; s < total; s += spacing) {
        const [x, y] = at(s);
        const lit = s <= sActive;
        g.fillStyle = lit ? alpha('teal', 0.95) : alpha('ink-3', 0.55);
        g.beginPath();
        g.arc(x, y, lit ? 3.2 : 2.4, 0, Math.PI * 2);
        g.fill();
      }
    });

    // References footer.
    const refs = el(
      'div',
      { style: { display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '0.82em', color: 'var(--ink-3)' } },
      el(
        'div',
        {},
        el('b', { style: { color: 'var(--ink-2)' } }, 'QJL'),
        ' · Zandieh, Daliri, Han · “QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead” · ',
        el('a', { href: 'https://arxiv.org/abs/2406.03482', target: '_blank', rel: 'noopener' }, 'arXiv:2406.03482'),
      ),
      el(
        'div',
        {},
        el('b', { style: { color: 'var(--ink-2)' } }, 'TurboQuant'),
        ' · Zandieh, Daliri, Hadian, Mirrokni · “TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate” · ',
        el('a', { href: 'https://arxiv.org/abs/2504.19874', target: '_blank', rel: 'noopener' }, 'arXiv:2504.19874'),
      ),
    );

    body.append(el('div.stage', {}, el('div.col', {}, v), detail));
    body.append(refs);
    show(0);

    ctx.onEnter(() => {
      if (!state.pinned) {
        state.active = 0;
        state.lastStep = state.t;
        show(0);
      }
    });
    ctx.loop((dt, t) => {
      state.t = t;
      state.flow += dt * 38;
      if (!state.pinned && state.hoverIdx === null && t - state.lastStep > 3.2) {
        state.lastStep = t;
        state.active = (state.active + 1) % STEPS.length;
        show(state.active);
      }
      cv.redraw();
    });
  },
};

import { el, rich, eq, slider, segmented, button, readout, viz, legend } from '../lib/ui.js';
import { createCanvas, View2D, arrow, line, label, text, alpha } from '../lib/canvas.js';
import { Plot } from '../lib/plot.js';
import * as M from '../lib/math.js';

const N_PTS = 40;
const M_MAX = 256;
const C_EPS = 2; // illustrative constant in ε = C·√(log n / m)

/** n points in R^d with clusters of very different spreads, so pairwise distances span ~3 decades. */
function makePoints(d, rng) {
  const K = 5;
  const sig = [0.06, 0.15, 0.3, 0.5, 0.8];
  const centers = Array.from({ length: K }, () => M.scale(M.randn(d, rng), 1 / Math.sqrt(d)));
  const pts = [];
  for (let i = 0; i < N_PTS; i++) {
    const g = i % K;
    pts.push(M.add(centers[g], M.scale(M.randn(d, rng), sig[g] / Math.sqrt(d))));
  }
  return pts;
}

export default {
  id: 'jl',
  chapter: 'QJL',
  kicker: 'Look again at attention',
  title: 'Keys only appear inside $\\langle q,k\\rangle$, so JL is enough',
  notes: `Re-read the attention formula. A key k_i is never used as a vector: it only enters through the single number ⟨q, k_i⟩ (softmax only ever sees those scores). Values are different: the output is literally a weighted sum of value vectors, so values must be reconstructed (MSE). Keys need only inner products.

That weaker requirement is exactly what lets us use JL. We may replace k by ANY sketch f(k) as long as ⟨f(q), f(k)⟩ ≈ ⟨q, k⟩. A random Gaussian projection is such a map, and it comes with useful random properties:
- it preserves inner products (and distances), not coordinates: Sk looks nothing like k, and that's fine, because coordinates are exactly what we said we don't need. (If we had to reconstruct k, as for values, JL with m < d would be useless.)
- S is drawn before any key or query exists (data-oblivious), and the guarantee holds for queries we haven't seen yet: online, as the requirements slide demanded;
- the error is random, not systematic, and the target dimension m depends on how many vectors we handle (n) and the accuracy (ε), not on d.

JL lemma (standard form; the QJL paper cites Johnson–Lindenstrauss and Dasgupta–Gupta but doesn't restate it): for n points, ε ∈ (0,1) and m = O(ε⁻² log n), f(x) = Sx/√m keeps every pairwise squared distance within 1 ± ε w.h.p. Polarization, ⟨x,y⟩ = (‖x‖² + ‖y‖² − ‖x−y‖²)/2, turns that into |⟨f(x),f(y)⟩ − ⟨x,y⟩| ≤ ε‖x‖‖y‖.

LEFT, what attention needs: one query and ten keys in d dimensions (synthetic). Orange outlines are the true scores ⟨q, k_i⟩; teal bars are the scores computed entirely in the projected space, ⟨Sq, Sk_i⟩/m. Drag m: at m = 2 the scores are garbage; by m ≈ 64–128 the teal bars sit on the outlines and the top key is preserved, even at d = 1024. Readouts: worst error in units of ‖q‖‖k‖, whether the top-scoring key is the same, and floats stored per key (m instead of d).

RIGHT, the guarantee for every pair at once: 40 synthetic points (five clusters), all 780 pairs. x = true cosine, y = projected inner product / (‖x‖‖y‖); the teal band is ±ε with ε = 2·√(log n/m) (the constant 2 is ours and illustrative; JL only says O(·)). Red pairs fall outside. The toggle shows the classical squared-distance form on log–log axes.
- Switch d between 64, 256, 1024: the spread at a given m doesn't change. m depends on n and ε, not d.
- "re-draw S": different points move, same overall tightness. The guarantee is over the randomness of S for a fixed (worst-case) point set.

Why it works, one pair at a time (header equation): each row s of S gives one sample (sᵀq)(sᵀk), and E[(sᵀq)(sᵀk)] = qᵀE[ssᵀ]k = ⟨q,k⟩ because E[ssᵀ] = I. So ⟨Sq,Sk⟩/m is an average of m independent unbiased samples: unbiased, with variance (⟨q,k⟩² + ‖q‖²‖k‖²)/m. The errors are random, not systematic, so they average out like 1/√m; a union bound over the n² pairs costs the log n in m. (Equivalently, ‖Sx‖²/m ~ ‖x‖²·χ²_m/m concentrates like √(2/m).)

The catch (bottom line on the left): float JL does save memory when m < d (the "floats per key" readout), but only by dropping rows, and the error sd is ~‖q‖‖k‖/√m. With head dim d = 128 there's little room: m = 32 is a 4× saving with an error of ~0.18–0.25‖q‖‖k‖, large for attention logits. And each row still costs 16 bits.
Better: spend the bits on more rows, not more precision per row. At the same 16m bits per key (our arithmetic from the two variance formulas):
• m floats (JL): Var = (⟨q,k⟩² + ‖q‖²‖k‖²)/m, up to 2‖q‖²‖k‖²/m;
• 16m sign bits (QJL, next slide): Var = ((π/2)‖q‖²‖k‖² − ⟨q,k⟩²)/(16m) ≤ π/(32m)·‖q‖²‖k‖².
Error sd ratio ≈ 3.2× (q ⟂ k) to 7.5× (q ∥ k) in favor of signs. Example at 256 bits per key: 16 floats → sd up to ≈ 0.35‖q‖‖k‖; 256 sign bits → ≈ 0.08‖q‖‖k‖.

Transition: so quantize the projection as hard as possible: keep only the sign of each coordinate of Sk, and use many more rows. That's QJL.`,
  mount(body, ctx) {
    const rng = new M.RNG(808);
    const cache = new Map(); // d → {pts, orig, P}
    const state = { d: 256, logm: 4, m: 16, mode: 'ip' };

    function project(entry) {
      const S = M.gaussianMatrix(M_MAX, entry.pts[0].length, rng);
      entry.P = S.map((row) => Float64Array.from(entry.pts, (x) => M.dot(row, x)));
    }
    function data(d) {
      if (!cache.has(d)) {
        const pts = makePoints(d, new M.RNG(1000 + d));
        const orig = [];
        const origIP = [];
        const norms = pts.map((x) => M.norm(x));
        for (let i = 0; i < N_PTS; i++)
          for (let j = i + 1; j < N_PTS; j++) {
            orig.push(M.norm(M.sub(pts[i], pts[j])) ** 2);
            origIP.push(M.dot(pts[i], pts[j]) / (norms[i] * norms[j]));
          }
        const entry = { pts, orig, origIP, norms };
        project(entry);
        cache.set(d, entry);
      }
      return cache.get(d);
    }
    function projected(entry, m) {
      const out = new Float64Array(entry.orig.length);
      let p = 0;
      for (let i = 0; i < N_PTS; i++)
        for (let j = i + 1; j < N_PTS; j++) {
          let s = 0;
          for (let r = 0; r < m; r++) {
            const t = entry.P[r][i] - entry.P[r][j];
            s += t * t;
          }
          out[p++] = s / m;
        }
      return out;
    }
    /** Projected inner products, in units of ‖x‖‖y‖. */
    function projectedIP(entry, m) {
      const out = new Float64Array(entry.origIP.length);
      let p = 0;
      for (let i = 0; i < N_PTS; i++)
        for (let j = i + 1; j < N_PTS; j++) {
          let s = 0;
          for (let r = 0; r < m; r++) s += entry.P[r][i] * entry.P[r][j];
          out[p++] = s / m / (entry.norms[i] * entry.norms[j]);
        }
      return out;
    }
    const epsOf = (m) => C_EPS * Math.sqrt(Math.log(N_PTS) / m);

    const rEps = readout('$\\varepsilon\\ \\propto\\ \\sqrt{\\log n/m}$');
    const rMax = readout('max distortion');
    const rOut = readout('pairs outside');

    // ---------- (a) scatter of projected vs original squared distances ----------
    const vA = viz('', 'every pair at once · 40 points · 780 pairs · synthetic');
    let ready = false;
    let proj = null;
    let projIP = null;
    function drawIP(g, w, h) {
      const entry = data(state.d);
      const { origIP } = entry;
      const eps = epsOf(state.m);
      let lo = Infinity;
      let hi = -Infinity;
      for (const v of origIP) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
      lo = Math.max(-1.2, lo - 0.35);
      hi = Math.min(1.2, hi + 0.25);
      const p = new Plot(g, {
        x: 8,
        y: 30,
        w: w - 16,
        h: h - 36,
        xmin: lo,
        xmax: hi,
        ymin: lo - 0.3,
        ymax: hi + 0.3,
        xlabel: `⟨x, y⟩ / ‖x‖‖y‖   (original, d = ${state.d})`,
        ylabel: `⟨f(x), f(y)⟩ / ‖x‖‖y‖   (m = ${state.m})`,
        margin: { l: 66, b: 46, t: 6, r: 12 },
      });
      p.frame();
      p.clip(() => {
        g.save();
        g.fillStyle = alpha('teal', 0.14);
        g.beginPath();
        g.moveTo(p.tx(lo), p.ty(lo + eps));
        g.lineTo(p.tx(hi), p.ty(hi + eps));
        g.lineTo(p.tx(hi), p.ty(hi - eps));
        g.lineTo(p.tx(lo), p.ty(lo - eps));
        g.closePath();
        g.fill();
        g.restore();
      });
      p.fn((x) => x + eps, { color: 'teal', width: 1.2, alpha: 0.6 });
      p.fn((x) => x - eps, { color: 'teal', width: 1.2, alpha: 0.6 });
      p.fn((x) => x, { color: 'ink', width: 1.2, dash: [6, 5], alpha: 0.6 });
      let maxErr = 0;
      let outside = 0;
      const inX = [];
      const inY = [];
      const outX = [];
      const outY = [];
      for (let i = 0; i < origIP.length; i++) {
        const err = Math.abs(projIP[i] - origIP[i]);
        maxErr = Math.max(maxErr, err);
        const y = M.clamp(projIP[i], lo - 0.28, hi + 0.28);
        if (err > eps) {
          outside++;
          outX.push(origIP[i]);
          outY.push(y);
        } else {
          inX.push(origIP[i]);
          inY.push(y);
        }
      }
      p.points(inX, inY, { color: 'accent', r: 2.6, alpha: 0.75 });
      p.points(outX, outY, { color: 'red', r: 3.2, alpha: 0.95 });
      label(g, `±ε band (ε ≈ ${eps.toFixed(2)}, illustrative constant)`, p.px + 10, p.py + 14, { size: 12.5, color: 'teal' });
      rEps.set(eps >= 1 ? '≥ 1' : eps.toFixed(2), 'teal');
      rMax.set(maxErr.toFixed(2), maxErr > eps ? 'red' : 'accent');
      rOut.set(`${outside} / ${origIP.length}`, outside ? 'red' : 'teal');
    }
    const cvA = createCanvas(vA, (g, w, h) => {
      if (!ready) return;
      if (state.mode === 'ip') {
        drawIP(g, w, h);
        return;
      }
      const entry = data(state.d);
      const { orig } = entry;
      let omin = Infinity;
      let omax = 0;
      for (const v of orig) {
        omin = Math.min(omin, v);
        omax = Math.max(omax, v);
      }
      const lo = omin / 4;
      const hi = omax * 4;
      const p = new Plot(g, {
        x: 8,
        y: 30,
        w: w - 16,
        h: h - 36,
        xmin: lo,
        xmax: hi,
        ymin: lo,
        ymax: hi,
        xlog: true,
        ylog: true,
        xlabel: `‖x − y‖²   (original, d = ${state.d})`,
        ylabel: `‖f(x) − f(y)‖²   (projected, m = ${state.m})`,
        margin: { l: 66, b: 46, t: 6, r: 12 },
      });
      p.frame();
      const eps = epsOf(state.m);
      const lowF = 1 - eps;
      // ±ε band
      p.clip(() => {
        g.save();
        g.fillStyle = alpha('teal', 0.14);
        g.beginPath();
        g.moveTo(p.tx(lo), p.ty(Math.min(hi, lo * (1 + eps))));
        g.lineTo(p.tx(hi / (1 + eps)), p.ty(hi));
        g.lineTo(p.tx(hi), p.ty(hi));
        if (lowF > 0) {
          g.lineTo(p.tx(hi), p.ty(hi * lowF));
          g.lineTo(p.tx(lo), p.ty(lo * lowF));
        } else {
          g.lineTo(p.tx(hi), p.ty(lo));
          g.lineTo(p.tx(lo), p.ty(lo));
        }
        g.closePath();
        g.fill();
        g.restore();
      });
      p.fn((x) => x * (1 + eps), { color: 'teal', width: 1.2, alpha: 0.6 });
      if (lowF > 0) p.fn((x) => x * lowF, { color: 'teal', width: 1.2, alpha: 0.6 });
      p.fn((x) => x, { color: 'ink', width: 1.2, dash: [6, 5], alpha: 0.6 });

      let maxDist = 0;
      let outside = 0;
      const inX = [];
      const inY = [];
      const outX = [];
      const outY = [];
      for (let i = 0; i < orig.length; i++) {
        const ratio = proj[i] / orig[i];
        maxDist = Math.max(maxDist, Math.abs(ratio - 1));
        const y = M.clamp(proj[i], lo * 1.05, hi / 1.05);
        if (Math.abs(ratio - 1) > eps) {
          outside++;
          outX.push(orig[i]);
          outY.push(y);
        } else {
          inX.push(orig[i]);
          inY.push(y);
        }
      }
      p.points(inX, inY, { color: 'accent', r: 2.6, alpha: 0.75 });
      p.points(outX, outY, { color: 'red', r: 3.2, alpha: 0.95 });
      label(g, 'y = x', p.tx(hi / 2.2), p.ty(hi / 2.2) - 12, { size: 12, color: 'ink-2', align: 'right' });
      label(g, `±ε band (ε ≈ ${eps.toFixed(2)}, illustrative constant)`, p.px + 10, p.py + 14, { size: 12.5, color: 'teal' });

      rEps.set(eps >= 1 ? '≥ 1' : eps.toFixed(2), 'teal');
      rMax.set(`${(100 * maxDist).toFixed(0)}%`, maxDist > eps ? 'red' : 'accent');
      rOut.set(`${outside} / ${orig.length}`, outside ? 'red' : 'teal');
    });

    function recompute() {
      if (!ready) return;
      proj = projected(data(state.d), state.m);
      projIP = projectedIP(data(state.d), state.m);
      cvA.request();
      cvB.request();
    }

    const sM = slider({
      label: '$m$',
      min: 1,
      max: 8,
      step: 0.05,
      value: state.logm,
      format: (v) => String(Math.round(2 ** v)),
      onInput: (v) => {
        state.logm = v;
        const m = Math.round(2 ** v);
        if (m !== state.m) {
          state.m = m;
          recompute();
        }
      },
    });
    sM.el.style.flex = '1';
    sM.el.style.minWidth = '200px';
    const segD = segmented({
      label: '$d$',
      options: [64, 256, 1024],
      value: state.d,
      onChange: (d) => {
        state.d = d;
        recompute();
      },
    });
    const redraw = button('re-draw $S$', () => {
      if (!ready) return;
      project(data(state.d));
      attProject(attData(state.d));
      recompute();
    });

    // ---------- (b) what attention needs: scores of one query against NK keys ----------
    const NK = 10;
    const attCache = new Map(); // d → {q, keys, Sq, Sk}
    const attRng = new M.RNG(707);
    function attData(d) {
      if (!attCache.has(d)) {
        const r = new M.RNG(5000 + d);
        const q = M.normalize(M.randn(d, r));
        const keys = [];
        for (let i = 0; i < NK; i++) {
          const c = -0.55 + (1.45 * i) / (NK - 1) + 0.08 * r.normal(); // cosines spread from −0.55 to 0.9
          const u0 = M.randn(d, r);
          const u = M.normalize(M.sub(u0, M.scale(q, M.dot(u0, q))));
          const cc = M.clamp(c, -0.95, 0.95);
          const len = 0.6 + 0.8 * r.next();
          keys.push(M.scale(M.add(M.scale(q, cc), M.scale(u, Math.sqrt(1 - cc * cc))), len));
        }
        // shuffle so the best key isn't always last
        for (let i = NK - 1; i > 0; i--) {
          const j = r.int(i + 1);
          [keys[i], keys[j]] = [keys[j], keys[i]];
        }
        const entry = { q, keys };
        attProject(entry);
        attCache.set(d, entry);
      }
      return attCache.get(d);
    }
    function attProject(entry) {
      const S = M.gaussianMatrix(M_MAX, entry.q.length, attRng);
      entry.Sq = S.map((row) => M.dot(row, entry.q));
      entry.Sk = entry.keys.map((k) => S.map((row) => M.dot(row, k)));
    }

    const rAttErr = readout('worst score error / $\\|q\\|\\|k\\|$');
    const rTop = readout('top key kept?');
    const rFloats = readout('floats per key');

    const vB = viz('', 'one query · ten keys · synthetic');
    const cvB = createCanvas(vB, (g, w, h) => {
      if (!ready) return;
      const { q, keys, Sq, Sk } = attData(state.d);
      const m = state.m;
      const truth = keys.map((k) => M.dot(q, k));
      const est = Sk.map((sk) => {
        let s = 0;
        for (let r = 0; r < m; r++) s += Sq[r] * sk[r];
        return s / m;
      });
      const lim = Math.max(1.4, ...truth.map(Math.abs)) * 1.15;
      const p = new Plot(g, {
        x: 8,
        y: 30,
        w: w - 16,
        h: h - 36,
        xmin: 0,
        xmax: NK,
        ymin: -lim,
        ymax: lim,
        xticks: [],
        ylabel: 'attention score',
        margin: { l: 56, b: 30, t: 6, r: 12 },
      });
      p.frame();
      p.hline(0, { color: 'line-2', dash: null });
      const bw = (p.tx(1) - p.tx(0)) * 0.56;
      let worst = 0;
      for (let i = 0; i < NK; i++) {
        const cx = p.tx(i + 0.5);
        const kn = M.norm(keys[i]);
        worst = Math.max(worst, Math.abs(est[i] - truth[i]) / kn);
        // projected score (teal, filled)
        const e = M.clamp(est[i], -lim, lim);
        g.fillStyle = alpha('teal', 0.75);
        g.fillRect(cx - bw / 2, Math.min(p.ty(0), p.ty(e)), bw, Math.abs(p.ty(e) - p.ty(0)));
        // true score (orange outline)
        g.strokeStyle = alpha('accent', 1);
        g.lineWidth = 2;
        g.strokeRect(cx - bw / 2 - 3, Math.min(p.ty(0), p.ty(truth[i])), bw + 6, Math.abs(p.ty(truth[i]) - p.ty(0)));
        text(g, `k${'₀₁₂₃₄₅₆₇₈₉'[i] || i}`, cx, p.py + p.ph + 14, { size: 13, color: 'accent', align: 'center' });
      }
      const top = truth.indexOf(Math.max(...truth));
      const topEst = est.indexOf(Math.max(...est));
      label(g, '▲ true top key', p.tx(top + 0.5), p.ty(-lim) - 10, { size: 12, color: 'accent', align: 'center' });
      label(g, `m = ${m}`, p.px + p.pw - 6, p.py + 12, { size: 12.5, color: 'ink-2', align: 'right' });
      rAttErr.set(worst.toFixed(2), worst > epsOf(m) ? 'red' : 'teal');
      rTop.set(top === topEst ? 'yes' : 'no', top === topEst ? 'teal' : 'red');
      rFloats.set(`${m} vs ${state.d}`, m < state.d ? 'teal' : 'red');
    });

    ctx.onEnter(() => {
      if (!ready) {
        ready = true;
        recompute();
      }
    });

    const chip = (html, cls, r) => rich(`div.tag.${cls}`, html, { 'data-reveal': r, style: { fontSize: '0.92em', padding: '5px 14px' } });
    const arrowSep = () => el('span', { style: { color: 'var(--ink-3)', fontSize: '1.2em' } }, '→');
    const head = el(
      'div.col',
      { style: { gap: '4px' } },
      el(
        'div.row',
        { style: { gap: '4px 28px' } },
        el('div', { 'data-reveal': '1', style: { minWidth: '0' } }, eq('o=\\sum_i \\operatorname{softmax}_i\\!\\Big(\\tfrac{\\textcolor{#f0a06a}{\\langle q,k_i\\rangle}}{\\sqrt d}\\Big)\\,v_i', 'big')),
        rich('div', 'We never need $k$. Only $\\langle q,k\\rangle$.', {
          'data-reveal': '2',
          style: { fontFamily: 'var(--serif)', fontWeight: '600', fontSize: 'clamp(20px, 1.9vw, 34px)', color: 'var(--yellow)' },
        }),
      ),
      el(
        'div.row',
        { style: { gap: '8px 12px' } },
        chip('store any $f(k)$ with $\\langle f(q),f(k)\\rangle\\approx\\langle q,k\\rangle$', 'c-accent', '3'),
        arrowSep(),
        chip('JL: random $S$, drawn before any $q,k$ exist', 'c-teal', '3'),
        el(
          'div',
          { 'data-reveal': '4', style: { minWidth: '0' } },
          eq('\\E\\big[\\tfrac1m\\langle Sq,Sk\\rangle\\big]=\\langle q,k\\rangle,\\quad \\Var=\\tfrac{\\langle q,k\\rangle^2+\\norm{q}^2\\norm{k}^2}{m}\\ \\Rightarrow\\ \\text{all pairs within }\\pm\\varepsilon\\norm{q}\\norm{k}\\text{ for }m=O\\!\\left(\\tfrac{\\log n}{\\varepsilon^{2}}\\right)'),
        ),
      ),
    );

    const fill = { height: 'auto', flex: '1', minHeight: '220px' };
    Object.assign(vA.style, fill);
    Object.assign(vB.style, fill);

    const segMode = segmented({
      options: [
        { label: 'inner products', value: 'ip' },
        { label: 'distances', value: 'dist' },
      ],
      value: state.mode,
      onChange: (v) => {
        state.mode = v;
        cvA.request();
      },
    });
    const punch = rich('p.note', 'Savings need fewer rows ($m<d$), and each row still costs 16 bits. Same bits, more rows: $16m$ sign bits beat $m$ floats by ≈3–7× in error. <b class="c-yellow">Keep only the sign →</b>', { style: { margin: '0', fontSize: '0.9em' } });
    const left = el('div.col', {}, vB, el('div.readouts', {}, rAttErr.el, rTop.el, rFloats.el), punch);
    const right = el('div.col', {}, vA, el('div.readouts', {}, rEps.el, rMax.el, rOut.el));
    const controls = el('div.row', {}, sM.el, segD.el, segMode.el, redraw);
    body.append(head, el('div.stage.even', {}, left, right), controls);
  },
};

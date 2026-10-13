// The naive uniform rounder used on the "Naive" slides (05, 06).
//
// Step Δ = 2m / (2^b − 1) on [−m, m]:
//   b = 1  → sign × m            (levels ±m = ±Δ/2; one bit cannot hold a symmetric zero level)
//   b ≥ 2  → round to the nearest multiple of Δ, clamped to |k| ≤ 2^(b−1) − 1
//            (2^b − 1 levels including 0; one code is unused, as in symmetric INT-b)
// Either way every coordinate with |v| ≤ m is off by at most Δ/2, so ‖x − x̃‖² ≤ d·Δ²/4.

export const naiveStep = (b, m = 1) => (2 * m) / (2 ** b - 1);

export function naiveLevels(b, m = 1) {
  if (b === 1) return [-m, m];
  const D = naiveStep(b, m);
  const K = 2 ** (b - 1) - 1;
  const out = [];
  for (let k = -K; k <= K; k++) out.push(k * D);
  return out;
}

export function naiveRound(v, b, m = 1) {
  if (!(m > 0)) return 0;
  if (b === 1) return v >= 0 ? m : -m;
  const D = naiveStep(b, m);
  const K = 2 ** (b - 1) - 1;
  const k = Math.max(-K, Math.min(K, Math.round(v / D)));
  return k * D + 0; // + 0 turns −0 into 0
}

/**
 * Quantize x with one of the two naive schemes.
 * mode 'fixed'  : store ‖x‖, round x/‖x‖ on a fixed grid over [−1, 1], rescale.
 * mode 'minmax' : store m = max|x_i|, round x on a grid over [−m, m] (KIVI-style scaling).
 * Returns { xq, scale, grid } where grid = the per-coordinate values the rounder saw (x/scale)
 * and gridQ = their rounded values, both in [−1, 1] units.
 */
export function naiveQuantize(x, b, mode = 'fixed') {
  const d = x.length;
  let scale = 0;
  if (mode === 'fixed') {
    for (let i = 0; i < d; i++) scale += x[i] * x[i];
    scale = Math.sqrt(scale);
  } else {
    for (let i = 0; i < d; i++) scale = Math.max(scale, Math.abs(x[i]));
  }
  const grid = new Float64Array(d);
  const gridQ = new Float64Array(d);
  const xq = new Float64Array(d);
  if (scale > 0) {
    for (let i = 0; i < d; i++) {
      grid[i] = x[i] / scale;
      gridQ[i] = naiveRound(grid[i], b, 1);
      xq[i] = gridQ[i] * scale;
    }
  }
  return { xq, scale, grid, gridQ };
}

export function relErr(x, xq) {
  let e = 0;
  let n = 0;
  for (let i = 0; i < x.length; i++) {
    e += (x[i] - xq[i]) ** 2;
    n += x[i] * x[i];
  }
  return n > 0 ? e / n : 0;
}

/** Percent formatting that stays readable from 0.01% to 100 000%. */
export function pct(v) {
  const p = 100 * v;
  if (p === 0) return '0%';
  if (p < 0.1) return `${p.toFixed(3)}%`;
  if (p < 10) return `${p.toFixed(2)}%`;
  if (p < 1000) return `${p.toFixed(1)}%`;
  return `${Math.round(p).toLocaleString('en-US')}%`;
}

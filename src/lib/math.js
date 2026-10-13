// Numerical helpers shared by every slide: seeded randomness, vectors,
// random rotations, the sphere-coordinate density, Lloyd–Max, QJL and TurboQuant.

// ---------- Randomness ----------

/** Small, fast, seedable PRNG (mulberry32). */
export class RNG {
  constructor(seed = 1) {
    this.s = seed >>> 0 || 1;
    this._spare = null;
  }
  next() {
    let t = (this.s = (this.s + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  /** Standard normal via Box–Muller (caches the spare). */
  normal() {
    if (this._spare !== null) {
      const v = this._spare;
      this._spare = null;
      return v;
    }
    let u = 0;
    while (u === 0) u = this.next();
    const v = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    this._spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }
  uniform(a = 0, b = 1) {
    return a + (b - a) * this.next();
  }
  int(n) {
    return Math.floor(this.next() * n);
  }
}

export const defaultRng = new RNG(20250425);

// ---------- Vectors ----------

export const zeros = (d) => new Float64Array(d);

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export const norm = (a) => Math.sqrt(dot(a, a));

export function scale(a, c) {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * c;
  return out;
}

export function add(a, b) {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

export function sub(a, b) {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

export function normalize(a) {
  const n = norm(a);
  return n > 0 ? scale(a, 1 / n) : Float64Array.from(a);
}

export function randn(d, rng = defaultRng) {
  const v = new Float64Array(d);
  for (let i = 0; i < d; i++) v[i] = rng.normal();
  return v;
}

/** Uniformly random point on the unit sphere S^{d-1}. */
export const randomUnit = (d, rng = defaultRng) => normalize(randn(d, rng));

// ---------- Matrices (array of Float64Array rows) ----------

export function gaussianMatrix(m, d, rng = defaultRng) {
  const M = new Array(m);
  for (let i = 0; i < m; i++) M[i] = randn(d, rng);
  return M;
}

/** Haar-random orthogonal d×d matrix: Gram–Schmidt on Gaussian rows (= QR with positive diag R). */
export function randomRotation(d, rng = defaultRng) {
  const Q = [];
  for (let i = 0; i < d; i++) {
    let v = randn(d, rng);
    // Two passes of modified Gram–Schmidt for numerical stability.
    for (let pass = 0; pass < 2; pass++) {
      for (let j = 0; j < i; j++) {
        const c = dot(v, Q[j]);
        for (let t = 0; t < d; t++) v[t] -= c * Q[j][t];
      }
    }
    Q.push(normalize(v));
  }
  return Q;
}

export function matVec(M, x) {
  const out = new Float64Array(M.length);
  for (let i = 0; i < M.length; i++) out[i] = dot(M[i], x);
  return out;
}

/** Mᵀx for M stored as rows. */
export function matTVec(M, x) {
  const d = M[0].length;
  const out = new Float64Array(d);
  for (let i = 0; i < M.length; i++) {
    const xi = x[i];
    const row = M[i];
    for (let j = 0; j < d; j++) out[j] += row[j] * xi;
  }
  return out;
}

/** 2×2 rotation matrix rows. */
export const rot2 = (theta) => [
  Float64Array.of(Math.cos(theta), -Math.sin(theta)),
  Float64Array.of(Math.sin(theta), Math.cos(theta)),
];

// ---------- Special functions & densities ----------

/** log Γ(x) for x > 0 (Lanczos, g = 7). */
export function lgamma(x) {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/**
 * Density of one coordinate of a uniform point on S^{d-1} (TurboQuant Lemma 1):
 * f(x) = Γ(d/2) / (√π Γ((d−1)/2)) · (1 − x²)^{(d−3)/2},  x ∈ [−1, 1].
 */
export function sphereCoordPdf(x, d) {
  if (x <= -1 || x >= 1) return d === 3 ? (Math.abs(x) === 1 ? 0.5 : 0) : 0;
  const logC = lgamma(d / 2) - 0.5 * Math.log(Math.PI) - lgamma((d - 1) / 2);
  return Math.exp(logC + ((d - 3) / 2) * Math.log(1 - x * x));
}

export const normalPdf = (x, sigma = 1, mu = 0) =>
  Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));

// ---------- Lloyd–Max (1-D continuous k-means) ----------

/**
 * Discretize a density on [lo, hi] into n cells. Besides the cell midpoints/weights {xs, ws}
 * (Σws = 1) it keeps cumulative moments W, M1, M2 at the cell edges, so bucket integrals can be
 * evaluated at arbitrary (non-grid) boundaries. That keeps Lloyd iterations moving even when an
 * update is smaller than one cell.
 */
export function makeGrid(pdf, lo, hi, n = 4000) {
  const xs = new Float64Array(n);
  const ws = new Float64Array(n);
  const dx = (hi - lo) / n;
  let total = 0;
  for (let i = 0; i < n; i++) {
    const x = lo + (i + 0.5) * dx;
    xs[i] = x;
    ws[i] = pdf(x) * dx;
    total += ws[i];
  }
  const W = new Float64Array(n + 1);
  const M1 = new Float64Array(n + 1);
  const M2 = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    ws[i] /= total;
    const x = xs[i];
    W[i + 1] = W[i] + ws[i];
    M1[i + 1] = M1[i] + ws[i] * x;
    // exact second moment of a uniform cell: x² + dx²/12
    M2[i + 1] = M2[i] + ws[i] * (x * x + (dx * dx) / 12);
  }
  return { xs, ws, lo, hi, n, dx, W, M1, M2 };
}

/** Cumulative moment `arr` evaluated at x (linear within a cell). */
function cumAt(grid, arr, x) {
  const p = Math.min(grid.n, Math.max(0, (x - grid.lo) / grid.dx));
  const i = Math.min(grid.n - 1, Math.floor(p));
  return arr[i] + (p - i) * (arr[i + 1] - arr[i]);
}

/** Bucket integrals (mass, first, second moment) between a and b. */
function bucket(grid, a, b) {
  return [
    cumAt(grid, grid.W, b) - cumAt(grid, grid.W, a),
    cumAt(grid, grid.M1, b) - cumAt(grid, grid.M1, a),
    cumAt(grid, grid.M2, b) - cumAt(grid, grid.M2, a),
  ];
}

export const midpoints = (c) => {
  const b = new Float64Array(c.length - 1);
  for (let i = 0; i < b.length; i++) b[i] = (c[i] + c[i + 1]) / 2;
  return b;
};

const edges = (grid, c) => {
  const m = midpoints(c);
  return [grid.lo, ...m, grid.hi];
};

/** Expected squared error of codebook c (sorted) under the gridded density. */
export function quantCost(grid, c) {
  const e = edges(grid, c);
  let cost = 0;
  for (let k = 0; k < c.length; k++) {
    const [w, m1, m2] = bucket(grid, e[k], e[k + 1]);
    cost += m2 - 2 * c[k] * m1 + c[k] * c[k] * w;
  }
  return Math.max(0, cost);
}

/** One Lloyd iteration: boundaries at midpoints, centroids at conditional means. */
export function lloydStep(grid, c) {
  const e = edges(grid, c);
  const next = new Float64Array(c.length);
  for (let k = 0; k < c.length; k++) {
    const [w, m1] = bucket(grid, e[k], e[k + 1]);
    next[k] = w > 1e-300 ? m1 / w : c[k];
  }
  return next;
}

/** Evenly spaced initial codebook covering the central mass. */
export function initCodebook(levels, lo, hi) {
  const c = new Float64Array(levels);
  for (let i = 0; i < levels; i++) c[i] = lo + ((i + 0.5) / levels) * (hi - lo);
  return c;
}

export function lloydMax(grid, levels, { iters = 1000, init } = {}) {
  let c = init ? Float64Array.from(init) : initCodebook(levels, grid.lo, grid.hi);
  for (let it = 0; it < iters; it++) c = lloydStep(grid, c);
  return { centroids: c, boundaries: midpoints(c), cost: quantCost(grid, c) };
}

const codebookCache = new Map();

/**
 * Optimal b-bit scalar codebook for one coordinate of a uniform point on S^{d-1}.
 * Returns {centroids, boundaries, cost} where d·cost = TurboQuant_mse distortion.
 */
export function sphereCodebook(d, b) {
  const key = `${d}:${b}`;
  if (codebookCache.has(key)) return codebookCache.get(key);
  const levels = 2 ** b;
  const s = 1 / Math.sqrt(d);
  const lo = d <= 4 ? -1 : Math.max(-1, -9 * s);
  const hi = -lo;
  const grid = makeGrid((x) => sphereCoordPdf(x, d), lo, hi, 6000);
  const init = initCodebook(levels, Math.max(lo, -3 * s), Math.min(hi, 3 * s));
  const res = lloydMax(grid, levels, { iters: 3000, init });
  codebookCache.set(key, res);
  return res;
}

/** Index of nearest centroid (boundaries = midpoints, sorted). */
export function nearestIndex(v, boundaries) {
  let lo = 0;
  let hi = boundaries.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (v > boundaries[mid]) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// ---------- Quantizers ----------

/** Uniform b-bit rounding grid on [lo, hi] (2^b levels including both ends). */
export function uniformLevels(b, lo = -1, hi = 1) {
  const n = 2 ** b;
  const c = new Float64Array(n);
  for (let i = 0; i < n; i++) c[i] = n === 1 ? (lo + hi) / 2 : lo + (i * (hi - lo)) / (n - 1);
  return c;
}

export function quantizeWith(v, centroids) {
  const bounds = midpoints(centroids);
  const out = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = centroids[nearestIndex(v[i], bounds)];
  return out;
}

/** QJL: sign(Sk) plus ‖k‖. */
export function qjlEncode(S, k) {
  const p = matVec(S, k);
  const bits = new Int8Array(p.length);
  for (let i = 0; i < p.length; i++) bits[i] = p[i] >= 0 ? 1 : -1;
  return { bits, norm: norm(k) };
}

/** Asymmetric QJL estimate of ⟨q, k⟩: √(π/2)/m · ‖k‖ · ⟨Sq, sign(Sk)⟩. */
export function qjlInner(S, q, code) {
  const m = S.length;
  let s = 0;
  for (let i = 0; i < m; i++) s += dot(S[i], q) * code.bits[i];
  return (Math.sqrt(Math.PI / 2) / m) * code.norm * s;
}

/** QJL dequantization as a vector: √(π/2)/m · ‖k‖ · Sᵀ sign(Sk). */
export function qjlDecode(S, code) {
  const m = S.length;
  const v = matTVec(S, Float64Array.from(code.bits));
  return scale(v, (Math.sqrt(Math.PI / 2) / m) * code.norm);
}

/** TurboQuant_mse on a unit vector x: rotate, per-coordinate Lloyd–Max, rotate back. */
export function turboMse(x, Pi, b) {
  const d = x.length;
  const { centroids } = sphereCodebook(d, b);
  const y = matVec(Pi, x);
  const yq = quantizeWith(y, centroids);
  return matTVec(Pi, yq);
}

/** TurboQuant_prod: (b−1)-bit TurboQuant_mse, then 1-bit QJL (S is d×d) on the residual. */
export function turboProd(x, Pi, S, b) {
  const xm = b > 1 ? turboMse(x, Pi, b - 1) : zeros(x.length);
  const r = sub(x, xm);
  const code = qjlEncode(S, r);
  return add(xm, qjlDecode(S, code));
}

// ---------- Stats ----------

export function histogram(values, lo, hi, bins) {
  const h = new Float64Array(bins);
  const w = (hi - lo) / bins;
  let n = 0;
  for (const v of values) {
    const k = Math.floor((v - lo) / w);
    if (k >= 0 && k < bins) h[k]++;
    n++;
  }
  // Normalize to a density.
  for (let i = 0; i < bins; i++) h[i] = n ? h[i] / (n * w) : 0;
  return h;
}

export function mean(a) {
  let s = 0;
  for (const v of a) s += v;
  return a.length ? s / a.length : 0;
}

export function variance(a) {
  const m = mean(a);
  let s = 0;
  for (const v of a) s += (v - m) ** 2;
  return a.length ? s / a.length : 0;
}

export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smoothstep = (t) => t * t * (3 - 2 * t);

/** Theoretical bounds from TurboQuant (unit-norm x, b bits per coordinate). */
export const bounds = {
  mseUpper: (b) => ((Math.sqrt(3) * Math.PI) / 2) * 4 ** -b,
  mseLower: (b) => 4 ** -b,
  prodUpper: (b, d, y2 = 1) => ((Math.sqrt(3) * Math.PI ** 2 * y2) / d) * 4 ** -b,
  prodLower: (b, d, y2 = 1) => (y2 / d) * 4 ** -b,
  /** QJL variance with m projections, unit vectors: ≤ π/(2m). */
  qjlVar: (m, q2 = 1, k2 = 1) => (Math.PI / (2 * m)) * q2 * k2,
};

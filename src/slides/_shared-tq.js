// Shared, lazily computed TurboQuant simulation at d = 128 for slides 17–21.
// No DOM here (importable from node for checks). Work is done incrementally:
// call `sim.step(ms)` from a slide's ctx.loop until `sim.done`.
import * as M from '../lib/math.js';

export const D = 128;
export const BMAX = 5;
const N = 640; // random unit vectors x
const K = 6; // query directions u ⟂ x per vector

/** Paper numbers (TurboQuant arXiv 2504.19874), indexed by b. */
export const PAPER = {
  /** Thm 1 text: D_mse ≈ 0.36, 0.117, 0.03, 0.009 (b = 1..4). */
  mseThm1: [null, 0.36, 0.117, 0.03, 0.009],
  /** Fig. 3b (measured, d = 1536): 0.36, 0.117, 0.035, 0.0095, 0.0027 (b = 1..5). */
  mseFig3b: [null, 0.36, 0.117, 0.035, 0.0095, 0.0027],
  /** Thm 2 text: d·D_prod ≈ 1.57, 0.56, 0.18, 0.047 (b = 1..4). */
  prodThm2: [null, 1.57, 0.56, 0.18, 0.047],
  /** Gap upper/lower: 1.45, 1.88, 2.21, 2.43 → 2.72. */
  gap: [null, 1.45, 1.88, 2.21, 2.43],
};

let Pi = null;
let S = null;
let G = null;

/** One Haar rotation Π (d×d), generated once. */
export function rotation() {
  if (!Pi) Pi = M.randomRotation(D, new M.RNG(17001));
  return Pi;
}

/** One Gaussian S (d×d), generated once. */
export function gaussianS() {
  if (!S) S = M.gaussianMatrix(D, D, new M.RNG(17002));
  return S;
}

/** (BMAX·d)×d Gaussian whose first d rows are S — QJL with m = b·d sign bits. */
function bigG() {
  if (!G) {
    const extra = M.gaussianMatrix((BMAX - 1) * D, D, new M.RNG(17003));
    G = [...gaussianS(), ...extra];
  }
  return G;
}

/** Lloyd–Max codebook for one coordinate at d = 128 (cached inside math.js). */
export const codebook = (b) => M.sphereCodebook(D, b);

/** d·C(f_X, b): the exact TurboQuant_mse distortion predicted by the codebook. */
export const codebookDmse = (b) => D * codebook(b).cost;

function perB() {
  const out = [null];
  for (let b = 1; b <= BMAX; b++) out.push({ a: new Float64Array(N), n2: new Float64Array(N), w: new Float64Array(N * K) });
  return out;
}

function makeSim() {
  const rng = new M.RNG(99);
  const s = {
    N,
    K,
    count: 0, // vectors processed so far
    done: false,
    phase: 0, // 0: rotation+S, 1..BMAX: codebooks, BMAX+1: vectors
    xs: [],
    us: [], // u_{i,0}
    theta: new Float64Array(N * K), // pair angle: y = cosθ·x + sinθ·u
    mse: perB(), // a = ⟨x, x̃⟩, n2 = ‖x̃‖², w = ⟨u, x̃⟩
    prod: perB(),
    // QJL with m = b·d (slide 20), computed only when requested.
    qjlWanted: false,
    qjlCount: 0,
    qjlErr: [null, ...Array.from({ length: BMAX }, () => new Float64Array(N))],
    get progress() {
      return this.count / N;
    },
    /** Run for at most `ms` milliseconds. Returns true if anything changed. */
    step(ms = 6) {
      const t0 = performance.now();
      let changed = false;
      while (performance.now() - t0 < ms) {
        if (this.phase === 0) {
          rotation();
          gaussianS();
          this.phase = 1;
        } else if (this.phase <= BMAX) {
          codebook(this.phase);
          this.phase++;
        } else if (this.count < N) {
          processOne(this, rng);
          changed = true;
        } else if (this.qjlWanted && this.qjlCount < N) {
          processQjl(this);
          changed = true;
        } else break;
      }
      this.done = this.count >= N && (!this.qjlWanted || this.qjlCount >= N);
      return changed;
    },
  };
  return s;
}

function processOne(s, rng) {
  const i = s.count;
  const P = rotation();
  const Sm = gaussianS();
  const x = M.randomUnit(D, rng);
  const us = [];
  for (let k = 0; k < K; k++) {
    const g = M.randn(D, rng);
    const c = M.dot(g, x);
    for (let t = 0; t < D; t++) g[t] -= c * x[t];
    us.push(M.normalize(g));
    s.theta[i * K + k] = rng.uniform(0, Math.PI);
  }
  s.xs.push(x);
  s.us.push(us[0]);

  const rec = (slot, xt) => {
    slot.a[i] = M.dot(x, xt);
    slot.n2[i] = M.dot(xt, xt);
    for (let k = 0; k < K; k++) slot.w[i * K + k] = M.dot(us[k], xt);
  };

  const yv = M.matVec(P, x);
  const xm = [M.zeros(D)];
  for (let b = 1; b <= BMAX; b++) {
    const yq = M.quantizeWith(yv, codebook(b).centroids);
    xm.push(M.matTVec(P, yq));
    rec(s.mse[b], xm[b]);
  }
  for (let b = 1; b <= BMAX; b++) {
    // TurboQuant_prod with total b bits: (b−1)-bit mse, then QJL(S) of the residual.
    const r = M.sub(x, xm[b - 1]);
    const xq = M.qjlDecode(Sm, M.qjlEncode(Sm, r));
    rec(s.prod[b], M.add(xm[b - 1], xq));
  }
  s.count++;
}

function processQjl(s) {
  const i = s.qjlCount;
  if (i >= s.count) return;
  const Gm = bigG();
  const x = s.xs[i];
  const u = s.us[i];
  let acc = 0;
  let r = 0;
  for (let b = 1; b <= BMAX; b++) {
    const m = b * D;
    for (; r < m; r++) acc += M.dot(Gm[r], u) * (M.dot(Gm[r], x) >= 0 ? 1 : -1);
    // y = u ⟂ x, so ⟨y,x⟩ = 0 and the error is minus the estimate (‖x‖ = 1).
    s.qjlErr[b][i] = -(Math.sqrt(Math.PI / 2) / m) * acc;
  }
  s.qjlCount++;
}

let _sim = null;
/** The shared simulation (created on first use; does no work until step()). */
export function sim() {
  if (!_sim) _sim = makeSim();
  return _sim;
}

// ---------- Derived statistics (over the vectors processed so far) ----------

/** Measured D_mse = E‖x − x̃‖² for TurboQuant_mse at b bits. */
export function measuredDmse(b, s = sim()) {
  const { a, n2 } = s.mse[b];
  let t = 0;
  for (let i = 0; i < s.count; i++) t += 1 - 2 * a[i] + n2[i];
  return s.count ? t / s.count : NaN;
}

/**
 * Inner-product errors e = ⟨y,x⟩ − ⟨y,x̃⟩ for unit queries y = c·x + √(1−c²)·u (u ⟂ x).
 * method: 'mse' | 'prod'. Returns one value per (x, u) pair.
 */
export function ipErrors(method, b, c, s = sim()) {
  const slot = s[method][b];
  const sn = Math.sqrt(Math.max(0, 1 - c * c));
  const out = new Float64Array(s.count * K);
  for (let i = 0; i < s.count; i++) {
    const base = c * (1 - slot.a[i]);
    for (let k = 0; k < K; k++) out[i * K + k] = base - sn * slot.w[i * K + k];
  }
  return out;
}

/** Pairs spanning all inner products: y = cosθ·x + sinθ·u. Returns {ip, est}. */
export function ipPairs(method, b, s = sim()) {
  const slot = s[method][b];
  const n = s.count * K;
  const ip = new Float64Array(n);
  const est = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const th = s.theta[j];
    const c = Math.cos(th);
    ip[j] = c;
    est[j] = c * slot.a[(j / K) | 0] + Math.sin(th) * slot.w[j];
  }
  return { ip, est };
}

/** Least-squares slope through the origin of est vs ip. */
export function slopeThroughOrigin(ip, est) {
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < ip.length; i++) {
    sxy += ip[i] * est[i];
    sxx += ip[i] * ip[i];
  }
  return sxx > 0 ? sxy / sxx : NaN;
}

/** {mean, var, ms} of an array (ms = mean square = mean² + var). */
export function moments(arr) {
  let s = 0;
  let s2 = 0;
  for (const v of arr) {
    s += v;
    s2 += v * v;
  }
  const n = arr.length || 1;
  const mean = s / n;
  const ms = s2 / n;
  return { mean, var: Math.max(0, ms - mean * mean), ms };
}

/** Measured d·D_prod of QJL with m = b·d sign bits, query ⟂ x. */
export function measuredQjl(b, s = sim()) {
  let t = 0;
  for (let i = 0; i < s.qjlCount; i++) t += s.qjlErr[b][i] ** 2;
  return s.qjlCount ? (D * t) / s.qjlCount : NaN;
}

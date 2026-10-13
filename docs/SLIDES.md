# Writing a slide

The deck is a vertical scroll of full-viewport sections. Every slide is one ES module in
`src/slides/NN-name.js`, registered (in order) in `src/slides/index.js`.

```js
import { el, rich, eq, slider, segmented, toggle, button, readout, legend, viz, texify } from '../lib/ui.js';
import { createCanvas, View2D, arrow, circle, dot, line, text, label, gridLines, draggable, alpha, color } from '../lib/canvas.js';
import { Plot } from '../lib/plot.js';
import { Orbit3D } from '../lib/orbit3d.js';
import * as M from '../lib/math.js';

export default {
  id: 'kebab-id',            // also the URL hash (#kebab-id)
  chapter: 'QJL',            // shown bottom-left; groups the progress rail
  kicker: 'Small label',     // optional, orange uppercase; $…$ allowed
  title: 'Short title',      // serif headline; $…$ allowed
  sub: 'optional one-liner', // use sparingly
  className: 'statement',    // optional extra classes on <section>
  bare: false,               // true = no header (you lay out everything)
  notes: `Speaker notes. Plain text; $…$ is rendered. This is where the words go.`,
  mount(body, ctx) {
    // body = .slide-body (flex column, fills the rest of the viewport)
    // ctx.loop((dt, t) => …)  runs every frame ONLY while the slide is visible
    // ctx.onEnter(fn) / ctx.onLeave(fn)
    // ctx.section, ctx.visible
  },
};
```

Exemplars to copy from: `src/slides/01-title.js` (bare, 3-D animation) and
`src/slides/04-metrics.js` (canvas sandbox + side panel, draggable vectors, toggles, readouts).

## Design rules

- **Visual first, minimal words.** On-screen text is labels, equations and at most one short line.
  All explanation goes in `notes` (press **N** in the deck). A slide should make sense to the
  presenter talking over it.
- **Every slide has something to touch**: a drag handle, a slider, a toggle, a re-roll button,
  or at least an animation that plays while visible.
- **Color semantics are global** (CSS vars, use names like `'accent'` in canvas helpers):
  `accent` (orange) = data vector $x$ / key $k$; `blue` = query $y$ / $q$;
  `teal` = reconstruction $\tilde x$ / good / TurboQuant; `violet` = residual / error;
  `red` = bad / bias / naive; `yellow` = highlight; `ink`, `ink-2`, `ink-3` = text levels.
- Layout helpers (in `styles.css`): `.stage` (2-col: big viz + side panel), `.stage.even`,
  `.stage.wide`, `.stage.three`, `.col`, `.row`, `.card` (+`.hot`), `.panel`, `.viz`
  (`.short`, `.tall`), `.readouts`, `.eq`, `.eq.big`, `.note`, `.muted`, `.big-number`,
  `pre.algo` (with `<span class="cm">` comments, `<span class="hl">` highlights), `table.data`,
  `data-reveal="1..5"` on elements for a staggered fade-in when the slide first appears.
- Use `viz(cls, label)` for a canvas container, then `createCanvas(container, draw)`. Call
  `cv.request()` after input changes, `cv.redraw()` inside `ctx.loop`.
- The deck must fit a 1440×900 or 1920×1080 viewport without scrolling inside a slide and
  remain usable down to ~800px wide (the stage collapses to one column below 900px).
- Equations: KaTeX via `eq(latex)` or `$…$` in any string passed through `rich`, `texify`,
  slider labels, readout labels, legend labels. Macros: `\R \E \S \ip{a}{b} \norm{x} \sign \Dmse \Dprod \xt \Var`.
- American English everywhere (color, behavior, normalize, …).

## Performance rules

- All 25 slides mount at page load. **Do not run heavy simulation in `mount`.** Compute lazily
  on first `ctx.onEnter`, or incrementally in `ctx.loop` (a few ms per frame max), and cache.
- Animation only through `ctx.loop` (it pauses offscreen). No `setInterval`.
- Simulations in high dimension: `d = 128` with a few hundred vectors is fine. Generate a random
  rotation once (`randomRotation(128)` ≈ 2M flops) and reuse it.
- Use seeded `new M.RNG(seed)` so the talk is reproducible run to run.

## Accuracy rules

- Facts, constants and experiment numbers come from `docs/paper-notes-qjl.md` and
  `docs/paper-notes-turboquant.md`. Never invent a number. If data is synthetic/illustrative,
  label it on the slide ("illustrative", "live simulation", "read off Fig. 5").
- Where the paper is ambiguous (e.g. the 3.5-bit split), say so in notes and label examples as ours.

## Don't touch shared files

`src/lib/*`, `src/styles.css`, `src/main.js`, `src/slides/index.js` are shared. If you need a
helper, put it in your slide file (or `src/slides/_shared-<topic>.js` if two of your slides need it).
Small one-off styling: inline `style` objects.

## Checking your work

- `npx vite build` must succeed.
- The dev server runs at http://localhost:5173 (hot reload). Open `http://localhost:5173/#<id>`
  in **your own new browser tab** to check visually; close your tab when done.

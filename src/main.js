import '@fontsource/ibm-plex-sans/400.css';
import '@fontsource/ibm-plex-sans/500.css';
import '@fontsource/ibm-plex-sans/600.css';
import '@fontsource/jetbrains-mono/400.css';
import '@fontsource/jetbrains-mono/600.css';
import '@fontsource-variable/source-serif-4';
import 'katex/dist/katex.min.css';
import './styles.css';

import slides from './slides/index.js';
import { el, texify } from './lib/ui.js';

const deck = document.getElementById('deck');
const rail = document.getElementById('rail');
const notesBox = document.getElementById('notes');
const hudChapter = document.getElementById('hud-chapter');
const hudCount = document.getElementById('hud-count');

/** Per-slide runtime: animation callbacks run only while the slide is on screen. */
const runtimes = [];
let current = 0;

slides.forEach((slide, i) => {
  const section = el(`section.slide${slide.className ? '.' + slide.className.split(' ').join('.') : ''}`, {
    id: slide.id,
    'data-index': i,
    'aria-label': slide.title || slide.id,
  });
  if (!slide.bare) {
    const head = el('header.slide-head');
    if (slide.kicker) head.append(el('div.kicker', { html: texify(slide.kicker) }));
    if (slide.title) head.append(el('h2.slide-title', { html: texify(slide.title) }));
    if (slide.sub) head.append(el('p.slide-sub', { html: texify(slide.sub) }));
    section.append(head);
  }
  const body = el('div.slide-body');
  section.append(body);
  deck.append(section);

  const rt = {
    slide,
    section,
    visible: false,
    loops: [],
    enter: [],
    leave: [],
    last: 0,
    t: 0,
  };
  runtimes.push(rt);

  const ctx = {
    section,
    /** fn(dt, t) runs every animation frame while the slide is visible. */
    loop: (fn) => rt.loops.push(fn),
    onEnter: (fn) => rt.enter.push(fn),
    onLeave: (fn) => rt.leave.push(fn),
    get visible() {
      return rt.visible;
    },
  };

  try {
    slide.mount?.(body, ctx);
  } catch (err) {
    console.error(`Slide "${slide.id}" failed to mount`, err);
    body.append(el('pre.algo', {}, `Slide failed to mount: ${err.message}`));
  }

  const prevChapter = i > 0 ? slides[i - 1].chapter : null;
  const dot = el('a', {
    href: `#${slide.id}`,
    title: `${i + 1}. ${(slide.title || slide.id).replace(/\$[^$]*\$/g, '…')}`,
    class: i > 0 && slide.chapter !== prevChapter ? 'chapter-start' : '',
  });
  dot.addEventListener('click', (e) => {
    e.preventDefault();
    go(i);
  });
  rail.append(dot);
  rt.dot = dot;
});

// ---------- Visibility & animation ----------

const io = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      const rt = runtimes[Number(e.target.dataset.index)];
      const vis = e.isIntersecting;
      if (vis && !rt.visible) {
        rt.visible = true;
        rt.last = performance.now();
        rt.section.classList.add('seen');
        rt.enter.forEach((f) => f());
      } else if (!vis && rt.visible) {
        rt.visible = false;
        rt.leave.forEach((f) => f());
      }
    }
  },
  { threshold: 0.15 },
);
runtimes.forEach((rt) => io.observe(rt.section));

// The "current" slide is whichever covers the viewport's middle line.
const centerIO = new IntersectionObserver(
  (entries) => {
    for (const e of entries) if (e.isIntersecting) setCurrent(Number(e.target.dataset.index));
  },
  { rootMargin: '-50% 0px -50% 0px', threshold: 0 },
);
runtimes.forEach((rt) => centerIO.observe(rt.section));

function frame(now) {
  for (const rt of runtimes) {
    if (!rt.visible || rt.loops.length === 0) continue;
    const dt = Math.min(0.05, (now - rt.last) / 1000);
    rt.last = now;
    rt.t += dt;
    for (const f of rt.loops) {
      try {
        f(dt, rt.t);
      } catch (err) {
        console.error(`Slide "${rt.slide.id}" loop error`, err);
        rt.loops = [];
      }
    }
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ---------- Navigation ----------

function setCurrent(i) {
  if (i === current && runtimes[i].dot.classList.contains('active')) return;
  runtimes[current]?.dot.classList.remove('active');
  current = i;
  const rt = runtimes[i];
  rt.dot.classList.add('active');
  hudChapter.textContent = rt.slide.chapter || '';
  hudCount.textContent = `${i + 1} / ${runtimes.length}`;
  history.replaceState(null, '', `#${rt.slide.id}`);
  if (!notesBox.hidden) showNotes();
}

function go(i) {
  const k = Math.max(0, Math.min(runtimes.length - 1, i));
  runtimes[k].section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

const escapeHtml = (t) => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** Notes are plain text: escape everything outside $…$ math, then render the math. */
const plainWithMath = (t) =>
  t
    .split(/(\$\$[\s\S]+?\$\$|\$[^$]+?\$)/g)
    .map((part, i) => (i % 2 ? texify(part) : escapeHtml(part)))
    .join('');

function showNotes() {
  const s = runtimes[current].slide;
  notesBox.innerHTML = `<b>${current + 1}. ${texify(s.title || s.id)}</b>\n${plainWithMath(s.notes || '(no notes)')}`;
}

window.addEventListener('keydown', (e) => {
  const tag = e.target?.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.metaKey || e.ctrlKey || e.altKey) return;
  switch (e.key) {
    case 'ArrowDown':
    case 'PageDown':
    case 'ArrowRight':
    case ' ':
      e.preventDefault();
      go(current + (e.shiftKey && e.key === ' ' ? -1 : 1));
      break;
    case 'ArrowUp':
    case 'PageUp':
    case 'ArrowLeft':
      e.preventDefault();
      go(current - 1);
      break;
    case 'Home':
      e.preventDefault();
      go(0);
      break;
    case 'End':
      e.preventDefault();
      go(runtimes.length - 1);
      break;
    case 'n':
    case 'N':
      notesBox.hidden = !notesBox.hidden;
      if (!notesBox.hidden) showNotes();
      break;
    case 'f':
    case 'F':
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen?.();
      break;
    case 'p':
    case 'P':
      document.body.classList.toggle('presenting');
      break;
    default:
  }
});

// Deep links: on load and when the hash is changed by hand.
history.scrollRestoration = 'manual';
function jumpToHash() {
  const i = runtimes.findIndex((rt) => `#${rt.slide.id}` === location.hash);
  if (i >= 0) {
    // Wait for fonts/layout so section offsets are final.
    requestAnimationFrame(() => window.scrollTo({ top: runtimes[i].section.offsetTop, behavior: 'instant' }));
  }
}
window.addEventListener('hashchange', jumpToHash);
if (location.hash) document.fonts.ready.then(jumpToHash);

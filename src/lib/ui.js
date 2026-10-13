// Tiny DOM helpers for slide layouts and controls.
import { tex } from './tex.js';

/**
 * el('div.card.hot', {attrs}, ...children) — tag with optional .classes and #id.
 * Strings are text; use html`...` or {html: '...'} for markup.
 */
export function el(spec, attrs = {}, ...children) {
  if (typeof attrs === 'string' || attrs instanceof Node || Array.isArray(attrs)) {
    children.unshift(attrs);
    attrs = {};
  }
  const [tagAndId, ...classes] = spec.split('.');
  const [tag, id] = tagAndId.split('#');
  const node = document.createElement(tag || 'div');
  if (id) node.id = id;
  if (classes.length) node.className = classes.join(' ');
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'html') node.innerHTML = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

/** Element whose innerHTML is `str` with $…$ / $$…$$ rendered by KaTeX. */
export function rich(spec, str, attrs = {}) {
  return el(spec, { ...attrs, html: texify(str) });
}

/** Replace $$…$$ (display) and $…$ (inline) in an HTML string with KaTeX markup. */
export function texify(str) {
  return str
    .replace(/\$\$([\s\S]+?)\$\$/g, (_, m) => tex(m, true))
    .replace(/\$([^$]+?)\$/g, (_, m) => tex(m, false));
}

/** Display equation block. */
export const eq = (latex, cls = '') => el(`div.eq${cls ? '.' + cls : ''}`, { html: tex(latex, true) });

/**
 * Range slider with label and live readout.
 * slider({label, min, max, step, value, format, onInput}) → {el, input, value, set(v)}
 */
export function slider({ label, min = 0, max = 1, step = 0.01, value = 0, format = (v) => v, onInput }) {
  const input = el('input', { type: 'range', min, max, step, value });
  const out = el('output');
  const lab = el('label', { html: texify(label) });
  const node = el('div.ctrl', {}, lab, input, out);
  const api = {
    el: node,
    input,
    get value() {
      return Number(input.value);
    },
    set(v, fire = true) {
      input.value = v;
      out.innerHTML = texify(String(format(Number(input.value))));
      if (fire && onInput) onInput(Number(input.value));
    },
  };
  input.addEventListener('input', () => api.set(input.value));
  out.innerHTML = texify(String(format(Number(value))));
  return api;
}

/** Segmented control: options [{label, value}] or plain values. */
export function segmented({ options, value, onChange, label }) {
  const opts = options.map((o) => (typeof o === 'object' ? o : { label: String(o), value: o }));
  let current = value;
  const buttons = opts.map((o) =>
    el('button', {
      type: 'button',
      'aria-pressed': String(o.value === value),
      html: texify(o.label),
      onclick: () => api.set(o.value),
    }),
  );
  const group = el('div.seg', { role: 'group' }, buttons);
  const node = label ? el('div.row', {}, el('span.muted', { html: texify(label) }), group) : group;
  const api = {
    el: node,
    get value() {
      return current;
    },
    set(v, fire = true) {
      current = v;
      buttons.forEach((b, i) => b.setAttribute('aria-pressed', String(opts[i].value === v)));
      if (fire && onChange) onChange(v);
    },
  };
  return api;
}

export function toggle({ label, value = false, onChange }) {
  const input = el('input', { type: 'checkbox' });
  input.checked = value;
  const node = el('label.toggle', {}, input, el('span', { html: texify(label) }));
  input.addEventListener('change', () => onChange && onChange(input.checked));
  return {
    el: node,
    get value() {
      return input.checked;
    },
    set(v, fire = true) {
      input.checked = v;
      if (fire && onChange) onChange(v);
    },
  };
}

export const button = (label, onClick, kind = '') =>
  el(`button.btn${kind ? '.' + kind : ''}`, { type: 'button', onclick: onClick, html: texify(label) });

/** Labeled numeric readout: readout('MSE') → {el, set(text, color?)} */
export function readout(label, value = '–') {
  const v = el('span.v', { html: texify(String(value)) });
  const node = el('div.readout', {}, el('span.k', { html: texify(label) }), v);
  return {
    el: node,
    set(text, colorVar) {
      v.innerHTML = texify(String(text));
      v.style.color = colorVar ? `var(--${colorVar})` : '';
    },
  };
}

/** Legend row: [{label, color, dash}] */
export function legend(items) {
  return el(
    'div.legend',
    {},
    items.map((i) =>
      el(
        'span',
        {},
        el('span.sw', {
          style: {
            background: i.dash ? 'transparent' : `var(--${i.color})`,
            borderTop: i.dash ? `2px dashed var(--${i.color})` : '',
            height: i.dash ? '0' : '4px',
          },
        }),
        el('span', { html: texify(i.label) }),
      ),
    ),
  );
}

/** Full-size .viz container with an optional corner label. */
export function viz(cls = '', labelText = '') {
  const node = el(`div.viz${cls ? '.' + cls : ''}`);
  if (labelText) node.append(el('div.viz-label', { html: texify(labelText) }));
  return node;
}

export const fmt = {
  f: (d) => (v) => Number(v).toFixed(d),
  pct: (v) => `${(100 * v).toFixed(1)}%`,
  sci: (v) => (Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : v.toFixed(4)),
};

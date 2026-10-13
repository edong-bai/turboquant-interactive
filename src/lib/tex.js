import katex from 'katex';

// Shared macros so every slide writes the same notation.
const macros = {
  '\\R': '\\mathbb{R}',
  '\\E': '\\mathbb{E}',
  '\\S': '\\mathbb{S}',
  '\\ip': '\\langle #1, #2 \\rangle',
  '\\norm': '\\lVert #1 \\rVert',
  '\\sign': '\\operatorname{sign}',
  '\\Dmse': 'D_{\\mathrm{mse}}',
  '\\Dprod': 'D_{\\mathrm{prod}}',
  '\\xt': '\\tilde{x}',
  '\\Var': '\\operatorname{Var}',
};

const cache = new Map();

/** Render LaTeX to an HTML string. */
export function tex(src, display = false) {
  const key = (display ? 'D:' : 'I:') + src;
  if (cache.has(key)) return cache.get(key);
  const html = katex.renderToString(src, { displayMode: display, throwOnError: false, macros, strict: false });
  cache.set(key, html);
  return html;
}

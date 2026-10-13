// Writes docs/SPEAKER_NOTES.md from the slide modules (same order as the deck).
// Run: npm run notes
import { writeFileSync } from 'node:fs';
import slides from '../src/slides/index.js';

const plain = (s = '') => s.replace(/<[^>]+>/g, '');
const out = ['# Speaker notes', '', '_Generated from `src/slides/*.js` by `npm run notes`. Press **N** in the deck to see these live._', ''];
slides.forEach((s, i) => {
  out.push(`## ${i + 1}. ${plain(s.title || s.id)}`);
  const meta = [`\`#${s.id}\``, s.chapter, s.kicker && plain(s.kicker)].filter(Boolean).join(' · ');
  out.push('', `*${meta}*`, '', (s.notes || '_(no notes)_').trim(), '');
});
writeFileSync(new URL('../docs/SPEAKER_NOTES.md', import.meta.url), out.join('\n'));
console.log(`wrote docs/SPEAKER_NOTES.md (${slides.length} slides)`);
slides.forEach((s, i) => console.log(`${i + 1}\t${s.id}\t${plain(s.title || '')}`));

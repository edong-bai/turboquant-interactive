import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset paths so dist/ can be hosted from any sub-path (or opened via a static server).
  base: './',
  build: {
    // One bundle on purpose: the whole talk should load once and run offline.
    chunkSizeWarningLimit: 1500,
  },
});

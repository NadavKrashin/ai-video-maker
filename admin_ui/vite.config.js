import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The panel is served by the pipeline itself (`pipeline.py serve` mounts
// admin_ui/dist at /), so production requests are same-origin. In dev, vite
// proxies /api to a locally running server instead.
export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8300'
    }
  }
});

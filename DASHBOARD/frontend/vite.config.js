import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  // Built HTML/JS/CSS is served by FastAPI via the StaticFiles mount at /static/
  // and lives in /static/dist/. The base path makes Vite emit URLs that match.
  base: '/static/dist/',
  build: {
    outDir: '../app/static/dist',
    emptyOutDir: true,
    rollupOptions: {
      input: './index.html',
    },
  },
});

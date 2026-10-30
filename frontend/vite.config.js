import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: './',
  // plotly.js/lib modules (used for the trimmed bundle in src/utils/Plot.js)
  // reference Node's `global`; map it to the browser's globalThis.
  define: { global: 'globalThis' },
  optimizeDeps: { esbuildOptions: { define: { global: 'globalThis' } } },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  },
  build: {
    outDir: 'dist',
    // No production source maps: they were 20 MB and shipped to every deploy
    sourcemap: false
  }
})
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
      // DATECT_API_PROXY points the dev server at another API, e.g. the deployed one:
      // DATECT_API_PROXY=https://datect.onrender.com npm run dev
      '/api': {
        target: process.env.DATECT_API_PROXY || 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  },
  test: { environment: 'node', include: ['src/**/*.test.js'] },
  build: {
    outDir: 'dist',
    // No production source maps: they were 20 MB and shipped to every deploy
    sourcemap: false
  }
})
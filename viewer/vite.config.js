import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Don't pre-bundle the MuJoCo WASM module — it self-locates its .wasm and the
  // optimizer mangles that. We serve the .wasm from /public via locateFile.
  optimizeDeps: { exclude: ['@mujoco/mujoco'] },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:3001', changeOrigin: true },
    },
  },
})

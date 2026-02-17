import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  // Tauri expects a fixed port
  server: {
    port: 1420,
    strictPort: true,
  },
  // Tauri uses a custom protocol
  envPrefix: ['VITE_', 'TAURI_'],
  build: {
    // Tauri uses Chromium on Windows/Linux and WebKit on macOS
    target: process.env.TAURI_PLATFORM === 'windows'
      ? 'chrome105'
      : process.env.TAURI_PLATFORM === 'macos'
        ? 'safari13'
        : 'chrome105',
    minify: !process.env.TAURI_DEBUG ? 'esbuild' : false,
    sourcemap: !!process.env.TAURI_DEBUG,
  },
})

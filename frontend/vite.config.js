import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: '/',
  plugins: [react()],
  server: {
    host: true, // Exposes Vite dev server to local network (LAN)
    port: 5173,
    allowedHosts: true, // Allows ngrok, cloudflare tunnels, and remote hosts
    proxy: {
      '/profiles': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/chat': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        bypass: (req) => {
          // If browser is loading the web page (HTML navigation), let Vite SPA serve index.html
          if (req.headers.accept && req.headers.accept.includes('text/html')) {
            return '/index.html';
          }
        },
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/memories': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/llm-status': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})

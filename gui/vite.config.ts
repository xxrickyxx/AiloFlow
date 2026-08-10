import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// The GUI never talks to a remote host: it proxies /v1 to the local AILOFlow
// runtime, whose port comes from AILOFLOW_API_PORT (default 11500 — 11434 is
// Ollama's and must stay free).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const apiPort = env.AILOFLOW_API_PORT || '11500';

  return {
    plugins: [react()],
    server: {
      port: 3000,
      host: true,
      proxy: {
        '/v1': {
          target: `http://127.0.0.1:${apiPort}`,
          changeOrigin: false,
          // Metrics and generation are server-sent event streams.
          ws: false,
          configure: (proxy) => {
            proxy.on('proxyRes', (proxyRes) => {
              if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
                proxyRes.headers['x-accel-buffering'] = 'no';
              }
            });
          },
        },
      },
    },
  };
});

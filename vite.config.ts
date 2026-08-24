import { defineConfig } from 'vite';
import typegpu from 'unplugin-typegpu/vite';

export default defineConfig({
  base: '/gpu-depth-lighting/',
  plugins: [typegpu()],
  server: { host: '127.0.0.1' },
});

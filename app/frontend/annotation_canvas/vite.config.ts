import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ command }) => {
  // Development mode: serve as regular app
  if (command === 'serve') {
    return {
      plugins: [react()],
      server: {
        port: 5174,
        cors: true,
      },
    }
  }

  // Production mode: build as library
  return {
    plugins: [react()],
    build: {
      outDir: 'build',
      lib: {
        entry: 'src/index.tsx',
        formats: ['es'],
        fileName: 'index'
      },
      rollupOptions: {
        external: ['react', 'react-dom'],
        output: {
          globals: {
            react: 'React',
            'react-dom': 'ReactDOM'
          }
        }
      }
    }
  }
})

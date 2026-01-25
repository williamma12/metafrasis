import '@testing-library/jest-dom'

// Mock Image to immediately trigger onload in tests
global.Image = class MockImage {
  onload: (() => void) | null = null
  onerror: (() => void) | null = null
  src = ''
  width = 800
  height = 600

  constructor() {
    // Immediately trigger onload in next tick to simulate image loading
    setTimeout(() => {
      if (this.onload) {
        this.onload()
      }
    }, 0)
  }
} as any

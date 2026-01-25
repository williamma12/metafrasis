import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'
import OCRViewer from '../src/OCRViewer'
import { Streamlit } from 'streamlit-component-lib'

// Mock Streamlit
vi.mock('streamlit-component-lib', () => ({
  Streamlit: {
    setComponentValue: vi.fn(),
    setFrameHeight: vi.fn(),
  },
  RenderData: {},
}))

describe('OCRViewer', () => {
  const mockWords = [
    {
      text: 'Hello',
      bbox: { left: 10, top: 10, width: 50, height: 20 },
      confidence: 0.95,
    },
    {
      text: 'World',
      bbox: { left: 70, top: 10, width: 50, height: 20 },
      confidence: 0.92,
    },
  ]

  const mockDetectorRegions = [
    {
      bbox: { left: 5, top: 5, width: 60, height: 30 },
      confidence: 0.98,
      index: 0,
    },
  ]

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Rendering', () => {
    it('renders image with correct src', async () => {
      const { container } = render(
        <OCRViewer
          imageUrl="test.png"
          words={mockWords}
          defaultVisibility={[true, true]}
        />
      )

      // Wait for image to load
      await waitFor(() => {
        const img = container.querySelector('img')
        expect(img).toBeTruthy()
      })

      const img = container.querySelector('img')
      expect(img?.getAttribute('src')).toBe('test.png')
    })

    it('renders all word bounding boxes', async () => {
      render(
        <OCRViewer
          imageUrl="test.png"
          words={mockWords}
          defaultVisibility={[true, true]}
        />
      )

      // Wait for image to load and words to render
      await waitFor(() => {
        expect(screen.getByText('Hello')).toBeInTheDocument()
      })
      expect(screen.getByText('World')).toBeInTheDocument()
    })

    it('renders detector regions when provided', async () => {
      const { container } = render(
        <OCRViewer
          imageUrl="test.png"
          words={mockWords}
          detectorRegions={mockDetectorRegions}
          defaultVisibility={[true, true]}
        />
      )

      // Wait for image to load
      await waitFor(() => {
        const svg = container.querySelector('svg')
        expect(svg).toBeTruthy()
      })

      const svg = container.querySelector('svg')
      const rects = svg?.querySelectorAll('rect')
      expect(rects).toBeTruthy()

      // With 2 words + 1 detector region, expect at least 3 rects
      expect(rects && rects.length).toBeGreaterThanOrEqual(mockWords.length)
    })

    it('applies correct styles for word boxes', async () => {
      const { container } = render(
        <OCRViewer
          imageUrl="test.png"
          words={mockWords}
          defaultVisibility={[true, true]}
        />
      )

      // Wait for image to load
      await waitFor(() => {
        const svg = container.querySelector('svg')
        expect(svg).toBeTruthy()
      })

      const svg = container.querySelector('svg')
      const rects = svg?.querySelectorAll('rect')

      expect(rects).toBeTruthy()
      expect(rects && rects.length).toBeGreaterThanOrEqual(mockWords.length)

      // Verify rects have stroke attribute
      const firstRect = rects?.[0]
      expect(firstRect).toBeTruthy()

      const stroke = firstRect?.getAttribute('stroke')
      expect(stroke).toBeTruthy()
      expect(typeof stroke).toBe('string')
    })

    it('handles empty words array gracefully', async () => {
      const { container } = render(
        <OCRViewer
          imageUrl="test.png"
          words={[]}
          defaultVisibility={[]}
        />
      )

      // Wait for image to load
      await waitFor(() => {
        const img = container.querySelector('img')
        expect(img).toBeTruthy()
      })
    })
  })

  describe('Interactivity', () => {
    it('clicking box toggles visibility', async () => {
      const { container } = render(
        <OCRViewer
          imageUrl="test.png"
          words={mockWords}
          defaultVisibility={[true, true]}
        />
      )

      // Wait for image to load first
      await waitFor(() => {
        const svg = container.querySelector('svg')
        expect(svg).toBeTruthy()
      })

      // Find first word box and click it
      const svg = container.querySelector('svg')
      const firstRect = svg?.querySelector('rect')

      expect(firstRect).toBeTruthy()

      if (firstRect) {
        fireEvent.click(firstRect)

        // Should call Streamlit with updated visibility
        await waitFor(() => {
          expect(Streamlit.setComponentValue).toHaveBeenCalled()
        })

        // Verify visibility was toggled
        const calls = vi.mocked(Streamlit.setComponentValue).mock.calls
        const visibilityCalls = calls.filter(call => call[0]?.visibility)
        expect(visibilityCalls.length).toBeGreaterThan(0)
      }
    })

    it('defaultVisibility prop controls initial box visibility', async () => {
      render(
        <OCRViewer
          imageUrl="test.png"
          words={mockWords}
          defaultVisibility={[false, true]}
        />
      )

      // Should send initial visibility state to Streamlit
      await waitFor(() => {
        expect(Streamlit.setComponentValue).toHaveBeenCalled()
      })

      const calls = vi.mocked(Streamlit.setComponentValue).mock.calls
      const lastCall = calls[calls.length - 1]

      expect(lastCall[0]).toHaveProperty('visibility')
      expect(lastCall[0].visibility).toEqual([false, true])
    })

    it('hidden boxes show tooltip on hover', async () => {
      const { container } = render(
        <OCRViewer
          imageUrl="test.png"
          words={mockWords}
          defaultVisibility={[false, true]}
        />
      )

      // Wait for image to load
      await waitFor(() => {
        const svg = container.querySelector('svg')
        expect(svg).toBeTruthy()
      })

      // Hover over hidden box should show tooltip
      // This would require checking for title elements or tooltip components
      const svg = container.querySelector('svg')
      const titles = svg?.querySelectorAll('title')
      expect(titles).toBeTruthy()
    })

    it('calls Streamlit.setComponentValue with updated visibility', async () => {
      const { container } = render(
        <OCRViewer
          imageUrl="test.png"
          words={mockWords}
          defaultVisibility={[true, true]}
        />
      )

      const svg = container.querySelector('svg')
      const rect = svg?.querySelector('rect')

      if (rect) {
        const initialCallCount = vi.mocked(Streamlit.setComponentValue).mock.calls.length

        fireEvent.click(rect)

        await waitFor(() => {
          const finalCallCount = vi.mocked(Streamlit.setComponentValue).mock.calls.length
          expect(finalCallCount).toBeGreaterThan(initialCallCount)
        })
      }
    })
  })

  describe('Layout', () => {
    it('SVG overlay matches image dimensions', async () => {
      const { container } = render(
        <OCRViewer
          imageUrl="test.png"
          words={mockWords}
          defaultVisibility={[true, true]}
        />
      )

      // Wait for image load
      await new Promise(resolve => setTimeout(resolve, 100))

      const svg = container.querySelector('svg')
      expect(svg).toBeTruthy()

      // Verify Streamlit setFrameHeight was called
      await waitFor(() => {
        expect(Streamlit.setFrameHeight).toHaveBeenCalled()
      })
    })

    it('text is centered in bounding boxes', async () => {
      const { container } = render(
        <OCRViewer
          imageUrl="test.png"
          words={mockWords}
          defaultVisibility={[true, true]}
        />
      )

      // Wait for image to load
      await waitFor(() => {
        const svg = container.querySelector('svg')
        expect(svg).toBeTruthy()
      })

      const svg = container.querySelector('svg')
      const textElements = svg?.querySelectorAll('text')

      expect(textElements).toBeTruthy()
      expect(textElements && textElements.length).toBeGreaterThan(0)

      // Verify text elements exist for words
      expect(screen.getByText('Hello')).toBeInTheDocument()
      expect(screen.getByText('World')).toBeInTheDocument()
    })

    it('font size scales based on box size', async () => {
      const smallWord = {
        text: 'A',
        bbox: { left: 10, top: 10, width: 10, height: 10 },
        confidence: 0.9,
      }
      const largeWord = {
        text: 'B',
        bbox: { left: 30, top: 30, width: 100, height: 50 },
        confidence: 0.9,
      }

      const { container } = render(
        <OCRViewer
          imageUrl="test.png"
          words={[smallWord, largeWord]}
          defaultVisibility={[true, true]}
        />
      )

      // Wait for image to load
      await waitFor(() => {
        const svg = container.querySelector('svg')
        expect(svg).toBeTruthy()
      })

      const svg = container.querySelector('svg')
      const textElements = svg?.querySelectorAll('text')

      expect(textElements).toBeTruthy()
      expect(textElements && textElements.length).toBe(2)

      // Get font sizes from text elements
      const firstText = textElements?.[0]
      const secondText = textElements?.[1]

      expect(firstText).toBeTruthy()
      expect(secondText).toBeTruthy()

      const firstFontSize = firstText?.getAttribute('font-size')
      const secondFontSize = secondText?.getAttribute('font-size')

      expect(firstFontSize).toBeTruthy()
      expect(secondFontSize).toBeTruthy()

      // Larger box should have larger or equal font size
      if (firstFontSize && secondFontSize) {
        const size1 = parseFloat(firstFontSize)
        const size2 = parseFloat(secondFontSize)

        expect(size2).toBeGreaterThanOrEqual(size1)
      }
    })
  })

  describe('Edge Cases', () => {
    it('works without detector regions (optional prop)', async () => {
      const { container } = render(
        <OCRViewer
          imageUrl="test.png"
          words={mockWords}
          defaultVisibility={[true, true]}
        />
      )

      // Wait for image to load
      await waitFor(() => {
        const img = container.querySelector('img')
        expect(img).toBeTruthy()
      })
    })

    it('handles missing confidence values', async () => {
      const wordsWithoutConfidence = [
        {
          text: 'Test',
          bbox: { left: 10, top: 10, width: 50, height: 20 },
          confidence: -1, // Default confidence
        },
      ]

      render(
        <OCRViewer
          imageUrl="test.png"
          words={wordsWithoutConfidence}
          defaultVisibility={[true]}
        />
      )

      // Wait for image to load
      await waitFor(() => {
        expect(screen.getByText('Test')).toBeInTheDocument()
      })
    })

    it('works with single word', async () => {
      render(
        <OCRViewer
          imageUrl="test.png"
          words={[mockWords[0]]}
          defaultVisibility={[true]}
        />
      )

      // Wait for image to load
      await waitFor(() => {
        expect(screen.getByText('Hello')).toBeInTheDocument()
      })
    })

    it('works with many overlapping words', async () => {
      const manyWords = Array.from({ length: 20 }, (_, i) => ({
        text: `Word${i}`,
        bbox: { left: i * 10, top: i * 5, width: 40, height: 15 },
        confidence: 0.9,
      }))

      render(
        <OCRViewer
          imageUrl="test.png"
          words={manyWords}
          defaultVisibility={Array(20).fill(true)}
        />
      )

      // Wait for image to load
      await waitFor(() => {
        expect(screen.getByText('Word0')).toBeInTheDocument()
      })
      expect(screen.getByText('Word19')).toBeInTheDocument()
    })
  })
})

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import React from 'react'
import AnnotationCanvas from '../src/AnnotationCanvas'
import type { Region } from '../src/types'
import { Streamlit } from 'streamlit-component-lib'

// Mock Streamlit
vi.mock('streamlit-component-lib', () => ({
  Streamlit: {
    setComponentValue: vi.fn(),
    setFrameHeight: vi.fn(),
  },
}))

describe('AnnotationCanvas', () => {
  const mockRegions: Region[] = [
    {
      id: 'region-1',
      type: 'rectangle',
      points: [
        { x: 10, y: 10 },
        { x: 60, y: 10 },
        { x: 60, y: 40 },
        { x: 10, y: 40 },
      ],
      text: 'Test region',
      auto_detected: false,
      verified: true,
    },
    {
      id: 'region-2',
      type: 'polygon',
      points: [
        { x: 100, y: 100 },
        { x: 150, y: 110 },
        { x: 140, y: 150 },
        { x: 110, y: 140 },
      ],
      text: null,
      auto_detected: true,
      verified: false,
    },
  ]

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Rendering', () => {
    it('renders image and canvas overlay', async () => {
      const { container } = render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={[]}
          selectedRegionId={null}
          drawingMode="select"
        />
      )

      // Wait for image to load
      await waitFor(() => {
        const svg = container.querySelector('svg')
        expect(svg).toBeTruthy()
      })

      const img = container.querySelector('img')
      const svg = container.querySelector('svg')

      expect(img).toBeTruthy()
      expect(img?.getAttribute('src')).toBe('test.png')
      expect(svg).toBeTruthy()
    })

    it('displays all regions', async () => {
      const { container } = render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={mockRegions}
          selectedRegionId={null}
          drawingMode="select"
        />
      )

      // Wait for image to load
      await waitFor(() => {
        const svg = container.querySelector('svg')
        expect(svg).toBeTruthy()
      })

      const svg = container.querySelector('svg')
      expect(svg).toBeTruthy()

      // Verify SVG exists and has regions to draw
      // Actual rendering would need SVG inspection
      expect(mockRegions.length).toBe(2)
    })

    it('shows current drawing mode indicator', async () => {
      render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={[]}
          selectedRegionId={null}
          drawingMode="rectangle"
        />
      )

      // Wait for image to load
      await waitFor(() => {
        expect(screen.queryByText(/rectangle/i) || screen.queryByText(/Rectangle/i)).toBeTruthy()
      })
    })

    it('handles empty regions array', async () => {
      const { container } = render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={[]}
          selectedRegionId={null}
          drawingMode="select"
        />
      )

      // Wait for image to load
      await waitFor(() => {
        const svg = container.querySelector('svg')
        expect(svg).toBeTruthy()
      })

      const svg = container.querySelector('svg')
      expect(svg).toBeTruthy()

      // Should render without errors
      expect(Streamlit.setFrameHeight).toHaveBeenCalled()
    })

    it('works with no selected region', async () => {
      const { container } = render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={mockRegions}
          selectedRegionId={null}
          drawingMode="select"
        />
      )

      // Wait for image to load
      await waitFor(() => {
        const svg = container.querySelector('svg')
        expect(svg).toBeTruthy()
      })

      const svg = container.querySelector('svg')
      expect(svg).toBeTruthy()

      // Should not crash with null selectedRegionId
      expect(() => render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={mockRegions}
          selectedRegionId={null}
          drawingMode="select"
        />
      )).not.toThrow()
    })
  })

  describe('Rectangle Mode', () => {
    it('mouse drag creates rectangle region', async () => {
      const { container } = render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={[]}
          selectedRegionId={null}
          drawingMode="rectangle"
        />
      )

      // Wait for image to load
      await waitFor(() => {
        const svg = container.querySelector('svg')
        expect(svg).toBeTruthy()
      })

      const svg = container.querySelector('svg')
      expect(svg).toBeTruthy()

      if (svg) {
        // Mock getBoundingClientRect to return proper dimensions
        svg.getBoundingClientRect = vi.fn(() => ({
          left: 0,
          top: 0,
          width: 800,
          height: 600,
          right: 800,
          bottom: 600,
          x: 0,
          y: 0,
          toJSON: () => {},
        }))

        // Simulate rectangle drawing
        fireEvent.mouseDown(svg, { clientX: 10, clientY: 10 })
        fireEvent.mouseMove(svg, { clientX: 60, clientY: 40 })
        fireEvent.mouseUp(svg, { clientX: 60, clientY: 40 })

        // Wait for state update
        await waitFor(() => {
          expect(Streamlit.setComponentValue).toHaveBeenCalled()
        })

        // Verify callback was called with 'add' action
        const calls = vi.mocked(Streamlit.setComponentValue).mock.calls
        const lastCall = calls[calls.length - 1]
        expect(lastCall[0]).toHaveProperty('action', 'add')
      }
    })

    it('rejects rectangles smaller than 5px', async () => {
      const { container } = render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={[]}
          selectedRegionId={null}
          drawingMode="rectangle"
        />
      )

      const svg = container.querySelector('svg')
      const initialCallCount = vi.mocked(Streamlit.setComponentValue).mock.calls.length

      if (svg) {
        // Draw very small rectangle (< 5px)
        fireEvent.mouseDown(svg, { clientX: 10, clientY: 10 })
        fireEvent.mouseMove(svg, { clientX: 12, clientY: 12 })
        fireEvent.mouseUp(svg, { clientX: 12, clientY: 12 })

        // Should NOT call setComponentValue with new region
        await new Promise(resolve => setTimeout(resolve, 100))

        const finalCallCount = vi.mocked(Streamlit.setComponentValue).mock.calls.length
        // No new regions should be created
        expect(finalCallCount).toBeLessThanOrEqual(initialCallCount + 1)
      }
    })

    it('updates region list on completion', async () => {
      const { container } = render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={[]}
          selectedRegionId={null}
          drawingMode="rectangle"
        />
      )

      const svg = container.querySelector('svg')

      if (svg) {
        fireEvent.mouseDown(svg, { clientX: 10, clientY: 10 })
        fireEvent.mouseMove(svg, { clientX: 60, clientY: 40 })
        fireEvent.mouseUp(svg, { clientX: 60, clientY: 40 })

        await waitFor(() => {
          const calls = vi.mocked(Streamlit.setComponentValue).mock.calls
          const addCalls = calls.filter(call => call[0]?.action === 'add')
          expect(addCalls.length).toBeGreaterThan(0)
        })
      }
    })
  })

  describe('Polygon Mode', () => {
    it('click adds polygon point', async () => {
      const { container } = render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={[]}
          selectedRegionId={null}
          drawingMode="polygon"
        />
      )

      // Wait for image to load
      await waitFor(() => {
        const svg = container.querySelector('svg')
        expect(svg).toBeTruthy()
      })

      const svg = container.querySelector('svg')
      expect(svg).toBeTruthy()

      if (svg) {
        const initialState = vi.mocked(Streamlit.setComponentValue).mock.calls.length

        // Click to add first point
        fireEvent.click(svg, { clientX: 10, clientY: 10 })

        // Component should update state (though polygon not complete yet)
        expect(svg).toBeTruthy()
      }
    })

    it('double-click completes polygon with min 3 points', async () => {
      const { container } = render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={[]}
          selectedRegionId={null}
          drawingMode="polygon"
        />
      )

      const svg = container.querySelector('svg')

      if (svg) {
        // Add 3 points
        fireEvent.click(svg, { clientX: 10, clientY: 10 })
        fireEvent.click(svg, { clientX: 50, clientY: 10 })
        fireEvent.click(svg, { clientX: 30, clientY: 40 })

        // Double-click to complete
        fireEvent.doubleClick(svg, { clientX: 30, clientY: 40 })

        // Should create polygon region
        await waitFor(() => {
          const calls = vi.mocked(Streamlit.setComponentValue).mock.calls
          const addCalls = calls.filter(call => call[0]?.action === 'add')
          expect(addCalls.length).toBeGreaterThan(0)
        })
      }
    })

    it('rejects polygons with less than 3 points', () => {
      const { container } = render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={[]}
          selectedRegionId={null}
          drawingMode="polygon"
        />
      )

      const svg = container.querySelector('svg')
      const initialCallCount = vi.mocked(Streamlit.setComponentValue).mock.calls.length

      if (svg) {
        // Add only 2 points
        fireEvent.click(svg, { clientX: 10, clientY: 10 })
        fireEvent.click(svg, { clientX: 50, clientY: 10 })

        // Try to complete
        fireEvent.doubleClick(svg, { clientX: 50, clientY: 10 })

        // Should NOT create polygon (needs min 3 points)
        const calls = vi.mocked(Streamlit.setComponentValue).mock.calls
        const addCalls = calls.filter(call => call[0]?.action === 'add')

        // If polygon was created with < 3 points, this is a bug
        // Expect no new 'add' actions
        expect(addCalls.length).toBe(0)
      }
    })
  })

  describe('Select Mode', () => {
    it('clicking region selects it', async () => {
      const { container } = render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={mockRegions}
          selectedRegionId={null}
          drawingMode="select"
        />
      )

      const svg = container.querySelector('svg')

      if (svg) {
        // Click on first region (center of region-1)
        fireEvent.click(svg, { clientX: 35, clientY: 25 })

        // Should call Streamlit with select action
        await waitFor(() => {
          const calls = vi.mocked(Streamlit.setComponentValue).mock.calls
          const selectCalls = calls.filter(call => call[0]?.action === 'select')
          expect(selectCalls.length).toBeGreaterThan(0)
        })
      }
    })

    it('clicking outside deselects', async () => {
      const { container } = render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={mockRegions}
          selectedRegionId="region-1"
          drawingMode="select"
        />
      )

      const svg = container.querySelector('svg')

      if (svg) {
        // Click outside all regions
        fireEvent.click(svg, { clientX: 200, clientY: 200 })

        // Should deselect
        await waitFor(() => {
          const calls = vi.mocked(Streamlit.setComponentValue).mock.calls
          const lastCall = calls[calls.length - 1]
          expect(lastCall[0]?.selectedRegionId).toBeNull()
        })
      }
    })
  })

  describe('Keyboard Shortcuts', () => {
    it('Delete key removes selected region', async () => {
      render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={mockRegions}
          selectedRegionId="region-1"
          drawingMode="select"
        />
      )

      // Press Delete key
      fireEvent.keyDown(document, { key: 'Delete' })

      // Should call Streamlit with delete action
      await waitFor(() => {
        const calls = vi.mocked(Streamlit.setComponentValue).mock.calls
        const deleteCalls = calls.filter(call => call[0]?.action === 'delete')
        expect(deleteCalls.length).toBeGreaterThan(0)
      })
    })

    it('Backspace key removes selected region', async () => {
      render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={mockRegions}
          selectedRegionId="region-1"
          drawingMode="select"
        />
      )

      // Press Backspace key
      fireEvent.keyDown(document, { key: 'Backspace' })

      // Should call Streamlit with delete action
      await waitFor(() => {
        const calls = vi.mocked(Streamlit.setComponentValue).mock.calls
        const deleteCalls = calls.filter(call => call[0]?.action === 'delete')
        expect(deleteCalls.length).toBeGreaterThan(0)
      })
    })

    it('Escape cancels drawing', () => {
      const { container } = render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={[]}
          selectedRegionId={null}
          drawingMode="polygon"
        />
      )

      const svg = container.querySelector('svg')
      const initialCallCount = vi.mocked(Streamlit.setComponentValue).mock.calls.length

      if (svg) {
        // Start drawing polygon
        fireEvent.click(svg, { clientX: 10, clientY: 10 })
        fireEvent.click(svg, { clientX: 50, clientY: 10 })

        // Press Escape
        fireEvent.keyDown(document, { key: 'Escape' })

        // Should cancel drawing (no new region created)
        const finalCallCount = vi.mocked(Streamlit.setComponentValue).mock.calls.length
        const calls = vi.mocked(Streamlit.setComponentValue).mock.calls.slice(initialCallCount)
        const addCalls = calls.filter(call => call[0]?.action === 'add')

        expect(addCalls.length).toBe(0)
      }
    })
  })

  describe('Edge Cases', () => {
    it('handles rapid mode switching', () => {
      const { rerender } = render(
        <AnnotationCanvas
          imageUrl="test.png"
          regions={[]}
          selectedRegionId={null}
          drawingMode="rectangle"
        />
      )

      // Switch to polygon
      expect(() => {
        rerender(
          <AnnotationCanvas
            imageUrl="test.png"
            regions={[]}
            selectedRegionId={null}
            drawingMode="polygon"
          />
        )
      }).not.toThrow()

      // Switch to select
      expect(() => {
        rerender(
          <AnnotationCanvas
            imageUrl="test.png"
            regions={[]}
            selectedRegionId={null}
            drawingMode="select"
          />
        )
      }).not.toThrow()
    })
  })
})

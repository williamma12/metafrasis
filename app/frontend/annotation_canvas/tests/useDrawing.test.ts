import { describe, it, expect, vi } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useDrawing } from '../src/hooks/useDrawing'
import type { Region } from '../src/types'

describe('useDrawing', () => {
  describe('State Management', () => {
    it('initializes with correct default state', () => {
      const mockCallback = vi.fn()
      const { result } = renderHook(() => useDrawing('rectangle', mockCallback))

      expect(result.current.drawingState.isDrawing).toBe(false)
      expect(result.current.drawingState.startPoint).toBeNull()
      expect(result.current.drawingState.currentPoint).toBeNull()
      expect(result.current.drawingState.polygonPoints).toEqual([])
    })

    it('sets isDrawing=true on mouse down', () => {
      const mockCallback = vi.fn()
      const { result } = renderHook(() => useDrawing('rectangle', mockCallback))

      act(() => {
        result.current.handleMouseDown({ x: 10, y: 10 })
      })

      expect(result.current.drawingState.isDrawing).toBe(true)
      expect(result.current.drawingState.startPoint).toEqual({ x: 10, y: 10 })
    })

    it('updates currentPoint on mouse move while drawing', () => {
      const mockCallback = vi.fn()
      const { result } = renderHook(() => useDrawing('rectangle', mockCallback))

      act(() => {
        result.current.handleMouseDown({ x: 10, y: 10 })
      })

      act(() => {
        result.current.handleMouseMove({ x: 50, y: 30 })
      })

      expect(result.current.drawingState.currentPoint).toEqual({ x: 50, y: 30 })
      expect(result.current.drawingState.isDrawing).toBe(true)
    })

    it('completes drawing on mouse up', () => {
      const mockCallback = vi.fn()
      const { result } = renderHook(() => useDrawing('rectangle', mockCallback))

      act(() => {
        result.current.handleMouseDown({ x: 10, y: 10 })
      })

      act(() => {
        result.current.handleMouseMove({ x: 60, y: 40 })
      })

      act(() => {
        result.current.handleMouseUp({ x: 60, y: 40 })
      })

      expect(result.current.drawingState.isDrawing).toBe(false)
    })
  })

  describe('Rectangle Creation', () => {
    it('generates region with correct bbox from start/end points', () => {
      const mockCallback = vi.fn()
      const { result } = renderHook(() => useDrawing('rectangle', mockCallback))

      act(() => {
        result.current.handleMouseDown({ x: 10, y: 10 })
      })

      act(() => {
        result.current.handleMouseMove({ x: 60, y: 40 })
      })

      act(() => {
        result.current.handleMouseUp({ x: 60, y: 40 })
      })

      expect(mockCallback).toHaveBeenCalledTimes(1)
      const region: Region = mockCallback.mock.calls[0][0]

      expect(region).toBeDefined()
      expect(region.type).toBe('rectangle')
      expect(region.points.length).toBe(4)

      // Verify bbox coordinates
      expect(region.points[0]).toEqual({ x: 10, y: 10 }) // Top-left
      expect(region.points[1]).toEqual({ x: 60, y: 10 }) // Top-right
      expect(region.points[2]).toEqual({ x: 60, y: 40 }) // Bottom-right
      expect(region.points[3]).toEqual({ x: 10, y: 40 }) // Bottom-left
    })

    it('rejects rectangles smaller than 5px', () => {
      const mockCallback = vi.fn()
      const { result } = renderHook(() => useDrawing('rectangle', mockCallback))

      act(() => {
        result.current.handleMouseDown({ x: 10, y: 10 })
        result.current.handleMouseMove({ x: 12, y: 12 }) // Only 2px x 2px
        result.current.handleMouseUp({ x: 12, y: 12 })
      })

      expect(mockCallback).not.toHaveBeenCalled() // Should reject small rectangles
    })

    it('assigns unique IDs', () => {
      const mockCallback = vi.fn()
      const { result } = renderHook(() => useDrawing('rectangle', mockCallback))

      act(() => result.current.handleMouseDown({ x: 10, y: 10 }))
      act(() => result.current.handleMouseMove({ x: 60, y: 40 }))
      act(() => result.current.handleMouseUp({ x: 60, y: 40 }))

      const firstId = mockCallback.mock.calls[0][0].id

      act(() => result.current.handleMouseDown({ x: 100, y: 100 }))
      act(() => result.current.handleMouseMove({ x: 150, y: 140 }))
      act(() => result.current.handleMouseUp({ x: 150, y: 140 }))

      const secondId = mockCallback.mock.calls[1][0].id

      expect(firstId).toBeDefined()
      expect(secondId).toBeDefined()
      expect(firstId).not.toBe(secondId) // IDs should be unique
    })
  })

  describe('Polygon Creation', () => {
    it('adds points to polygonPoints on click', () => {
      const mockCallback = vi.fn()
      const { result } = renderHook(() => useDrawing('polygon', mockCallback))

      act(() => {
        result.current.handleMouseDown({ x: 10, y: 10 })
      })

      expect(result.current.drawingState.polygonPoints).toHaveLength(1)
      expect(result.current.drawingState.polygonPoints[0]).toEqual({ x: 10, y: 10 })

      act(() => {
        result.current.handleMouseDown({ x: 50, y: 10 })
      })

      expect(result.current.drawingState.polygonPoints).toHaveLength(2)
      expect(result.current.drawingState.polygonPoints[1]).toEqual({ x: 50, y: 10 })
    })

    it('completes on double-click with min 3 points', () => {
      const mockCallback = vi.fn()
      const { result } = renderHook(() => useDrawing('polygon', mockCallback))

      act(() => result.current.handleMouseDown({ x: 10, y: 10 }))
      act(() => result.current.handleMouseDown({ x: 50, y: 10 }))
      act(() => result.current.handleMouseDown({ x: 30, y: 40 }))
      act(() => result.current.handleDoubleClick())

      expect(mockCallback).toHaveBeenCalledTimes(1)
      const region: Region = mockCallback.mock.calls[0][0]

      expect(region).toBeDefined()
      expect(region.type).toBe('polygon')
      expect(region.points.length).toBe(3)
      expect(region.points[0]).toEqual({ x: 10, y: 10 })
      expect(region.points[1]).toEqual({ x: 50, y: 10 })
      expect(region.points[2]).toEqual({ x: 30, y: 40 })
    })

    it('requires minimum 3 points', () => {
      const mockCallback = vi.fn()
      const { result } = renderHook(() => useDrawing('polygon', mockCallback))

      act(() => {
        result.current.handleMouseDown({ x: 10, y: 10 })
        result.current.handleMouseDown({ x: 50, y: 10 })
        // Only 2 points, try to complete
        result.current.handleDoubleClick()
      })

      expect(mockCallback).not.toHaveBeenCalled() // Should reject polygon with < 3 points
    })

    it('generates polygon region with all points', () => {
      const mockCallback = vi.fn()
      const { result } = renderHook(() => useDrawing('polygon', mockCallback))

      const points = [
        { x: 10, y: 10 },
        { x: 50, y: 10 },
        { x: 60, y: 40 },
        { x: 20, y: 50 },
      ]

      points.forEach(point => {
        act(() => result.current.handleMouseDown(point))
      })
      act(() => result.current.handleDoubleClick())

      expect(mockCallback).toHaveBeenCalledTimes(1)
      const region: Region = mockCallback.mock.calls[0][0]

      expect(region).toBeDefined()
      expect(region.points).toEqual(points)
    })
  })

  describe('Utilities', () => {
    it('generateId creates unique strings', () => {
      const mockCallback = vi.fn()
      const { result } = renderHook(() => useDrawing('rectangle', mockCallback))

      act(() => result.current.handleMouseDown({ x: 10, y: 10 }))
      act(() => result.current.handleMouseMove({ x: 60, y: 40 }))
      act(() => result.current.handleMouseUp({ x: 60, y: 40 }))
      const id1 = mockCallback.mock.calls[0][0].id

      act(() => result.current.handleMouseDown({ x: 100, y: 100 }))
      act(() => result.current.handleMouseMove({ x: 150, y: 140 }))
      act(() => result.current.handleMouseUp({ x: 150, y: 140 }))
      const id2 = mockCallback.mock.calls[1][0].id

      expect(id1).toBeTruthy()
      expect(id2).toBeTruthy()
      expect(id1).not.toBe(id2)
      expect(typeof id1).toBe('string')
      expect(typeof id2).toBe('string')
    })

    it('point-to-region conversion works correctly', () => {
      const mockCallback = vi.fn()
      const { result } = renderHook(() => useDrawing('rectangle', mockCallback))

      act(() => result.current.handleMouseDown({ x: 10, y: 20 }))
      act(() => result.current.handleMouseMove({ x: 100, y: 80 }))
      act(() => result.current.handleMouseUp({ x: 100, y: 80 }))

      expect(mockCallback).toHaveBeenCalledTimes(1)
      const region: Region = mockCallback.mock.calls[0][0]

      expect(region).toBeDefined()
      expect(region.points).toHaveLength(4)

      // Verify region properties
      expect(region.type).toBe('rectangle')
      expect(region.auto_detected).toBe(false)
      expect(region.verified).toBe(false)
      expect(region.text).toBeNull()
    })
  })
})

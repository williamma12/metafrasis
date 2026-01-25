import { useState, useCallback } from 'react'
import { Point, DrawingState, DrawingMode, Region } from '../types'

/**
 * Generate a unique ID for regions
 */
const generateId = (): string => {
  return Math.random().toString(36).substring(2, 10)
}

/**
 * Custom hook for managing drawing state
 */
export function useDrawing(
  mode: DrawingMode,
  onRegionCreate: (region: Region) => void
) {
  const [drawingState, setDrawingState] = useState<DrawingState>({
    isDrawing: false,
    startPoint: null,
    currentPoint: null,
    polygonPoints: [],
  })

  const handleMouseDown = useCallback(
    (point: Point) => {
      if (mode === 'select') return

      if (mode === 'rectangle') {
        setDrawingState({
          isDrawing: true,
          startPoint: point,
          currentPoint: point,
          polygonPoints: [],
        })
      } else if (mode === 'polygon') {
        // Add point to polygon
        setDrawingState((prev) => ({
          ...prev,
          isDrawing: true,
          polygonPoints: [...prev.polygonPoints, point],
          currentPoint: point,
        }))
      }
    },
    [mode]
  )

  const handleMouseMove = useCallback(
    (point: Point) => {
      if (!drawingState.isDrawing) return

      setDrawingState((prev) => ({
        ...prev,
        currentPoint: point,
      }))
    },
    [drawingState.isDrawing]
  )

  const handleMouseUp = useCallback(
    (point: Point) => {
      if (mode === 'rectangle' && drawingState.isDrawing && drawingState.startPoint) {
        // Create rectangle region
        const start = drawingState.startPoint
        const minX = Math.min(start.x, point.x)
        const maxX = Math.max(start.x, point.x)
        const minY = Math.min(start.y, point.y)
        const maxY = Math.max(start.y, point.y)

        // Only create if it has some size
        if (maxX - minX > 5 && maxY - minY > 5) {
          const region: Region = {
            id: generateId(),
            type: 'rectangle',
            points: [
              { x: minX, y: minY },
              { x: maxX, y: minY },
              { x: maxX, y: maxY },
              { x: minX, y: maxY },
            ],
            text: null,
            auto_detected: false,
            verified: false,
          }
          onRegionCreate(region)
        }

        setDrawingState({
          isDrawing: false,
          startPoint: null,
          currentPoint: null,
          polygonPoints: [],
        })
      }
      // Polygon mode continues until double-click
    },
    [mode, drawingState, onRegionCreate]
  )

  const handleDoubleClick = useCallback(() => {
    if (mode === 'polygon' && drawingState.polygonPoints.length >= 3) {
      // Create polygon region
      const region: Region = {
        id: generateId(),
        type: 'polygon',
        points: [...drawingState.polygonPoints],
        text: null,
        auto_detected: false,
        verified: false,
      }
      onRegionCreate(region)

      setDrawingState({
        isDrawing: false,
        startPoint: null,
        currentPoint: null,
        polygonPoints: [],
      })
    }
  }, [mode, drawingState.polygonPoints, onRegionCreate])

  const cancelDrawing = useCallback(() => {
    setDrawingState({
      isDrawing: false,
      startPoint: null,
      currentPoint: null,
      polygonPoints: [],
    })
  }, [])

  return {
    drawingState,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    handleDoubleClick,
    cancelDrawing,
  }
}

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Streamlit } from 'streamlit-component-lib'
import { Point, Region, DrawingMode, AnnotationCanvasState } from './types'
import { useDrawing } from './hooks/useDrawing'

interface AnnotationCanvasProps {
  imageUrl: string
  regions: Region[]
  selectedRegionId: string | null
  drawingMode: DrawingMode
}

const AnnotationCanvas: React.FC<AnnotationCanvasProps> = ({
  imageUrl,
  regions: initialRegions,
  selectedRegionId: initialSelectedId,
  drawingMode,
}) => {
  const [regions, setRegions] = useState<Region[]>(initialRegions)
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(initialSelectedId)
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null)
  const [hoveredRegionId, setHoveredRegionId] = useState<string | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  // Sync with props
  useEffect(() => {
    setRegions(initialRegions)
  }, [initialRegions])

  useEffect(() => {
    setSelectedRegionId(initialSelectedId)
  }, [initialSelectedId])

  // Load image dimensions
  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      setImageSize({ width: img.width, height: img.height })
      Streamlit.setFrameHeight(img.height + 100)
    }
    img.src = imageUrl
  }, [imageUrl])

  // Send state back to Streamlit
  const sendState = useCallback((state: AnnotationCanvasState) => {
    Streamlit.setComponentValue(state)
  }, [])

  // Handle region creation
  const handleRegionCreate = useCallback((region: Region) => {
    const newRegions = [...regions, region]
    setRegions(newRegions)
    setSelectedRegionId(region.id)
    sendState({
      regions: newRegions,
      selectedRegionId: region.id,
      action: 'add',
      actionTimestamp: Date.now(),
    })
  }, [regions, sendState])

  // Handle region selection
  const handleRegionSelect = useCallback((regionId: string | null) => {
    if (drawingMode !== 'select') return
    setSelectedRegionId(regionId)
    sendState({
      regions,
      selectedRegionId: regionId,
      action: 'select',
      actionTimestamp: Date.now(),
    })
  }, [drawingMode, regions, sendState])

  // Handle region deletion
  const handleRegionDelete = useCallback((regionId: string) => {
    const newRegions = regions.filter((r) => r.id !== regionId)
    setRegions(newRegions)
    setSelectedRegionId(null)
    sendState({
      regions: newRegions,
      selectedRegionId: null,
      action: 'delete',
      actionTimestamp: Date.now(),
    })
  }, [regions, sendState])

  // Drawing hook
  const {
    drawingState,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    handleDoubleClick,
    cancelDrawing,
  } = useDrawing(drawingMode, handleRegionCreate)

  // Convert screen coordinates to SVG coordinates
  const getPointFromEvent = useCallback(
    (e: React.MouseEvent<SVGSVGElement>): Point | null => {
      if (!svgRef.current || !imageSize) return null

      const svg = svgRef.current
      const rect = svg.getBoundingClientRect()
      const scaleX = imageSize.width / rect.width
      const scaleY = imageSize.height / rect.height

      return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
      }
    },
    [imageSize]
  )

  // Mouse event handlers
  const onMouseDown = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const point = getPointFromEvent(e)
      if (point) {
        handleMouseDown(point)
      }
    },
    [getPointFromEvent, handleMouseDown]
  )

  const onMouseMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const point = getPointFromEvent(e)
      if (point) {
        handleMouseMove(point)
      }
    },
    [getPointFromEvent, handleMouseMove]
  )

  const onMouseUp = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const point = getPointFromEvent(e)
      if (point) {
        handleMouseUp(point)
      }
    },
    [getPointFromEvent, handleMouseUp]
  )

  const onDoubleClick = useCallback(() => {
    handleDoubleClick()
  }, [handleDoubleClick])

  // Keyboard event handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        cancelDrawing()
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && selectedRegionId) {
        handleRegionDelete(selectedRegionId)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [cancelDrawing, handleRegionDelete, selectedRegionId])

  // Get cursor based on mode
  const getCursor = (): string => {
    switch (drawingMode) {
      case 'rectangle':
        return 'crosshair'
      case 'polygon':
        return 'crosshair'
      case 'select':
        return 'default'
      default:
        return 'default'
    }
  }

  // Render a region
  const renderRegion = (region: Region) => {
    const isSelected = region.id === selectedRegionId
    const isHovered = region.id === hoveredRegionId

    const strokeColor = isSelected
      ? '#2563eb' // blue-600
      : region.auto_detected
      ? '#dc2626' // red-600
      : '#16a34a' // green-600

    const fillOpacity = isSelected ? 0.3 : isHovered ? 0.2 : 0.1

    if (region.type === 'rectangle' && region.points.length === 4) {
      const minX = Math.min(...region.points.map((p) => p.x))
      const minY = Math.min(...region.points.map((p) => p.y))
      const maxX = Math.max(...region.points.map((p) => p.x))
      const maxY = Math.max(...region.points.map((p) => p.y))

      return (
        <g key={region.id}>
          <rect
            x={minX}
            y={minY}
            width={maxX - minX}
            height={maxY - minY}
            fill={strokeColor}
            fillOpacity={fillOpacity}
            stroke={strokeColor}
            strokeWidth={isSelected ? 3 : 2}
            strokeDasharray={region.auto_detected && !region.verified ? '5,5' : 'none'}
            style={{ cursor: drawingMode === 'select' ? 'pointer' : 'crosshair' }}
            onClick={(e) => {
              e.stopPropagation()
              handleRegionSelect(region.id)
            }}
            onMouseEnter={() => setHoveredRegionId(region.id)}
            onMouseLeave={() => setHoveredRegionId(null)}
          />
          {region.text && (
            <text
              x={minX + 5}
              y={minY + 16}
              fill={strokeColor}
              fontSize="14"
              fontWeight="bold"
              style={{ pointerEvents: 'none' }}
            >
              {region.text.substring(0, 20)}
              {region.text.length > 20 ? '...' : ''}
            </text>
          )}
        </g>
      )
    } else if (region.type === 'polygon' && region.points.length >= 3) {
      const pointsStr = region.points.map((p) => `${p.x},${p.y}`).join(' ')

      return (
        <g key={region.id}>
          <polygon
            points={pointsStr}
            fill={strokeColor}
            fillOpacity={fillOpacity}
            stroke={strokeColor}
            strokeWidth={isSelected ? 3 : 2}
            strokeDasharray={region.auto_detected && !region.verified ? '5,5' : 'none'}
            style={{ cursor: drawingMode === 'select' ? 'pointer' : 'crosshair' }}
            onClick={(e) => {
              e.stopPropagation()
              handleRegionSelect(region.id)
            }}
            onMouseEnter={() => setHoveredRegionId(region.id)}
            onMouseLeave={() => setHoveredRegionId(null)}
          />
          {/* Vertex handles for selected polygon */}
          {isSelected &&
            region.points.map((point, idx) => (
              <circle
                key={idx}
                cx={point.x}
                cy={point.y}
                r={6}
                fill="white"
                stroke={strokeColor}
                strokeWidth={2}
                style={{ cursor: 'move' }}
              />
            ))}
        </g>
      )
    }
    return null
  }

  // Render drawing preview
  const renderDrawingPreview = () => {
    if (!drawingState.isDrawing) return null

    if (drawingMode === 'rectangle' && drawingState.startPoint && drawingState.currentPoint) {
      const start = drawingState.startPoint
      const current = drawingState.currentPoint
      const minX = Math.min(start.x, current.x)
      const minY = Math.min(start.y, current.y)
      const maxX = Math.max(start.x, current.x)
      const maxY = Math.max(start.y, current.y)

      return (
        <rect
          x={minX}
          y={minY}
          width={maxX - minX}
          height={maxY - minY}
          fill="rgba(37, 99, 235, 0.2)"
          stroke="#2563eb"
          strokeWidth={2}
          strokeDasharray="5,5"
          style={{ pointerEvents: 'none' }}
        />
      )
    }

    if (drawingMode === 'polygon' && drawingState.polygonPoints.length > 0) {
      const points = [...drawingState.polygonPoints]
      if (drawingState.currentPoint) {
        points.push(drawingState.currentPoint)
      }
      const pointsStr = points.map((p) => `${p.x},${p.y}`).join(' ')

      return (
        <g style={{ pointerEvents: 'none' }}>
          <polyline
            points={pointsStr}
            fill="none"
            stroke="#2563eb"
            strokeWidth={2}
            strokeDasharray="5,5"
          />
          {drawingState.polygonPoints.map((point, idx) => (
            <circle key={idx} cx={point.x} cy={point.y} r={5} fill="#2563eb" />
          ))}
        </g>
      )
    }

    return null
  }

  if (!imageSize) {
    return <div style={{ padding: '20px' }}>Loading image...</div>
  }

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <img
        src={imageUrl}
        alt="Annotation"
        style={{ display: 'block', maxWidth: '100%', height: 'auto' }}
      />

      <svg
        ref={svgRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          cursor: getCursor(),
        }}
        viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
        preserveAspectRatio="xMinYMin meet"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onDoubleClick={onDoubleClick}
      >
        {/* Render existing regions */}
        {regions.map(renderRegion)}

        {/* Render drawing preview */}
        {renderDrawingPreview()}
      </svg>

      {/* Mode indicator */}
      <div
        style={{
          position: 'absolute',
          top: 10,
          right: 10,
          background: 'rgba(0, 0, 0, 0.7)',
          color: 'white',
          padding: '6px 12px',
          borderRadius: '4px',
          fontSize: '12px',
          textTransform: 'uppercase',
        }}
      >
        {drawingMode === 'polygon' && drawingState.polygonPoints.length > 0
          ? `Polygon: ${drawingState.polygonPoints.length} points (double-click to finish)`
          : drawingMode}
      </div>

      {/* Keyboard shortcut hints */}
      <div
        style={{
          position: 'absolute',
          bottom: 10,
          left: 10,
          background: 'rgba(0, 0, 0, 0.7)',
          color: 'white',
          padding: '6px 12px',
          borderRadius: '4px',
          fontSize: '11px',
        }}
      >
        <span style={{ marginRight: '10px' }}>ESC: Cancel</span>
        <span>Delete: Remove selected</span>
      </div>
    </div>
  )
}

export default AnnotationCanvas

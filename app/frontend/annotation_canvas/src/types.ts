/**
 * Type definitions for the annotation canvas component
 */

export interface Point {
  x: number
  y: number
}

export interface Region {
  id: string
  type: 'rectangle' | 'polygon'
  points: Point[]
  text: string | null
  auto_detected: boolean
  verified: boolean
}

export type DrawingMode = 'rectangle' | 'polygon' | 'select'

export interface AnnotationCanvasProps {
  imageUrl: string
  regions: Region[]
  selectedRegionId: string | null
  drawingMode: DrawingMode
}

export interface AnnotationCanvasState {
  regions: Region[]
  selectedRegionId: string | null
  action: 'add' | 'delete' | 'update' | 'select' | null
  actionTimestamp?: number
}

export interface DrawingState {
  isDrawing: boolean
  startPoint: Point | null
  currentPoint: Point | null
  polygonPoints: Point[]
}

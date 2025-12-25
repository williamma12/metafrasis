import React, { useState, useEffect } from 'react'
import { Streamlit, RenderData } from 'streamlit-component-lib'

interface BBox {
  left: number
  top: number
  width: number
  height: number
}

interface Word {
  text: string
  bbox: BBox
  confidence: number
}

interface OCRViewerProps {
  imageUrl: string
  words: Word[]
  defaultVisibility: boolean[]
}

const OCRViewer: React.FC<OCRViewerProps> = ({ imageUrl, words, defaultVisibility }) => {
  const [visibility, setVisibility] = useState<boolean[]>(defaultVisibility)
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null)

  // Load image to get dimensions
  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      setImageSize({ width: img.width, height: img.height })
      // Notify Streamlit about height
      Streamlit.setFrameHeight(img.height + 50)
    }
    img.src = imageUrl
  }, [imageUrl])

  // Send visibility state back to Streamlit
  useEffect(() => {
    Streamlit.setComponentValue({ visibility })
  }, [visibility])

  const toggleBox = (index: number) => {
    const newVisibility = [...visibility]
    newVisibility[index] = !newVisibility[index]
    setVisibility(newVisibility)
  }

  // Calculate font size to fit text in box
  const getFontSize = (text: string, bbox: BBox): number => {
    const charWidth = bbox.width / (text.length * 0.6) // Approximate character width
    const charHeight = bbox.height * 0.7 // Leave some padding
    return Math.min(charWidth, charHeight, 16) // Cap at 16px
  }

  if (!imageSize) {
    return <div>Loading...</div>
  }

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <img
        src={imageUrl}
        alt="OCR"
        style={{ display: 'block', maxWidth: '100%', height: 'auto' }}
      />

      <svg
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none'
        }}
        viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
        preserveAspectRatio="xMinYMin meet"
      >
        {words.map((word, idx) => {
          const fontSize = getFontSize(word.text, word.bbox)

          return (
            <g key={idx}>
              {/* Visible bounding box */}
              {visibility[idx] && (
                <>
                  <rect
                    x={word.bbox.left}
                    y={word.bbox.top}
                    width={word.bbox.width}
                    height={word.bbox.height}
                    fill="white"
                    stroke="black"
                    strokeWidth={2}
                    style={{ pointerEvents: 'auto', cursor: 'pointer' }}
                    onClick={() => toggleBox(idx)}
                  />
                  <text
                    x={word.bbox.left + word.bbox.width / 2}
                    y={word.bbox.top + word.bbox.height / 2}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontSize={fontSize}
                    fontFamily="Arial, sans-serif"
                    fill="black"
                    style={{ pointerEvents: 'auto', cursor: 'pointer' }}
                    onClick={() => toggleBox(idx)}
                  >
                    {word.text}
                  </text>
                </>
              )}

              {/* Invisible clickable area for hidden boxes */}
              {!visibility[idx] && (
                <rect
                  x={word.bbox.left}
                  y={word.bbox.top}
                  width={word.bbox.width}
                  height={word.bbox.height}
                  fill="transparent"
                  stroke="transparent"
                  style={{ pointerEvents: 'auto', cursor: 'pointer' }}
                  onMouseEnter={() => setHoveredIndex(idx)}
                  onMouseLeave={() => setHoveredIndex(null)}
                  onClick={() => toggleBox(idx)}
                />
              )}
            </g>
          )
        })}
      </svg>

      {/* Tooltip for hidden boxes */}
      {hoveredIndex !== null && !visibility[hoveredIndex] && (
        <div
          style={{
            position: 'absolute',
            left: `${(words[hoveredIndex].bbox.left / imageSize.width) * 100}%`,
            top: `${(words[hoveredIndex].bbox.top / imageSize.height) * 100}%`,
            transform: 'translate(-50%, -100%)',
            marginTop: '-10px',
            background: 'rgba(0, 0, 0, 0.9)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '6px',
            fontSize: '14px',
            whiteSpace: 'nowrap',
            pointerEvents: 'none',
            zIndex: 1000,
            boxShadow: '0 2px 8px rgba(0,0,0,0.3)'
          }}
        >
          <div><strong>{words[hoveredIndex].text}</strong></div>
          {words[hoveredIndex].confidence >= 0 && (
            <div style={{ fontSize: '12px', marginTop: '4px' }}>
              Confidence: {(words[hoveredIndex].confidence * 100).toFixed(1)}%
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default OCRViewer

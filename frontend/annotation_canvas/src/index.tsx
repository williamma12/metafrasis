import React from 'react'
import ReactDOM from 'react-dom/client'
import { Streamlit, StreamlitComponentBase, withStreamlitConnection } from 'streamlit-component-lib'
import AnnotationCanvas from './AnnotationCanvas'
import { Region, DrawingMode } from './types'

interface State {}

interface Args {
  imageUrl: string
  regions: Region[]
  selectedRegionId: string | null
  drawingMode: DrawingMode
}

class StreamlitAnnotationCanvas extends StreamlitComponentBase<State> {
  public render(): React.ReactNode {
    const args = this.props.args as Args
    const { imageUrl, regions, selectedRegionId, drawingMode } = args

    return (
      <AnnotationCanvas
        imageUrl={imageUrl}
        regions={regions || []}
        selectedRegionId={selectedRegionId}
        drawingMode={drawingMode || 'select'}
      />
    )
  }
}

const ConnectedComponent = withStreamlitConnection(StreamlitAnnotationCanvas)

const root = ReactDOM.createRoot(document.getElementById('root')!)
root.render(
  <React.StrictMode>
    <ConnectedComponent />
  </React.StrictMode>
)

// Notify Streamlit that we're ready
Streamlit.setComponentReady()

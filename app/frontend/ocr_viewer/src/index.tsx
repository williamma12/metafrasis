import React from 'react'
import ReactDOM from 'react-dom/client'
import { Streamlit, StreamlitComponentBase, withStreamlitConnection } from 'streamlit-component-lib'
import OCRViewer from './OCRViewer'

interface State {}

class StreamlitOCRViewer extends StreamlitComponentBase<State> {
  public render(): React.ReactNode {
    const { imageUrl, words, defaultVisibility } = this.props.args

    return (
      <OCRViewer
        imageUrl={imageUrl}
        words={words}
        defaultVisibility={defaultVisibility}
      />
    )
  }
}

const ConnectedComponent = withStreamlitConnection(StreamlitOCRViewer)

const root = ReactDOM.createRoot(document.getElementById('root')!)
root.render(
  <React.StrictMode>
    <ConnectedComponent />
  </React.StrictMode>
)

// Notify Streamlit that we're ready
Streamlit.setComponentReady()

# Annotation Tool

## Overview
Interactive annotation tool for creating OCR training datasets. Supports manual region selection (rectangles and polygons) and optional auto-detection using existing OCR detectors. Exports data for both detector and recognizer training.

## Features

- **Region Selection**: Draw rectangles or polygons around text regions
- **Text Labeling**: Optional text transcription for each region
- **Auto-Detection**: Use CRAFT/DBNet to pre-populate regions
- **Dataset Management**: Create, load, and save annotation datasets
- **Export**: Generate training data for detectors (bounding boxes) and recognizers (cropped images + labels)

## Quick Start

1. Navigate to the "Annotate" tab in the Streamlit app
2. Create a new dataset or load an existing one
3. Upload images to annotate
4. Use the toolbar to select drawing mode:
   - **Rectangle**: Click and drag to draw axis-aligned boxes
   - **Polygon**: Click to place vertices, double-click to close
   - **Select**: Click regions to edit or delete
5. (Optional) Enable auto-detect to pre-populate regions
6. Click on a region to select it, then enter text in the sidebar
7. Save the dataset and export when done

## Directory Structure

```
services/annotation/
├── __init__.py
├── models.py           # Data models (Region, AnnotatedImage, AnnotationDataset)
├── storage.py          # JSON save/load operations
├── canvas.py           # Streamlit component wrapper
└── exporter.py         # Export for training

frontend/annotation_canvas/
├── src/
│   ├── AnnotationCanvas.tsx    # Main React component
│   ├── types.ts                # TypeScript interfaces
│   └── hooks/
│       └── useDrawing.ts       # Drawing state management
├── package.json
├── vite.config.ts
└── index.html

data/
├── annotations/        # Saved datasets (JSON)
│   └── images/        # Copied source images
└── exports/           # Exported training data
    ├── detector/      # Bounding box format (COCO JSON)
    └── recognizer/    # Image-text pairs
```

## Data Format

### Annotation Dataset (JSON)

```json
{
  "name": "ancient_greek_v1",
  "version": "1.0",
  "created_at": "2025-01-15T10:30:00Z",
  "updated_at": "2025-01-15T14:22:00Z",
  "images": [
    {
      "id": "img_abc123",
      "image_path": "images/ancient_greek_v1/page_001.jpg",
      "width": 2048,
      "height": 3072,
      "regions": [
        {
          "id": "r1a2b3c4",
          "type": "rectangle",
          "points": [
            {"x": 100, "y": 200},
            {"x": 400, "y": 200},
            {"x": 400, "y": 250},
            {"x": 100, "y": 250}
          ],
          "text": "τοῦ δὲ γενομένου",
          "auto_detected": false,
          "verified": true,
          "created_at": "2025-01-15T10:35:00Z"
        }
      ]
    }
  ]
}
```

### Region Types

| Type | Points | Use Case |
|------|--------|----------|
| Rectangle | 4 corners (clockwise from top-left) | Axis-aligned text lines |
| Polygon | N vertices | Rotated text, curved baselines |

### Export Formats

**Detector Training (COCO format):**
```
exports/detector/{dataset_name}/
├── annotations.json     # COCO-style annotations
└── export_summary.json  # Export metadata
```

**Recognizer Training:**
```
exports/recognizer/{dataset_name}/
├── images/
│   ├── {region_id}.png  # Cropped region images
│   └── {region_id}.txt  # Text labels
├── labels.csv           # Master list (region_id, text, etc.)
└── export_summary.json  # Export metadata
```

## Auto-Detection

Enable the "Auto-detect" toggle to use existing OCR detectors:

| Detector | Description |
|----------|-------------|
| CRAFT | Character-level detection, good for dense text |
| DBNet | Word/line-level detection, fast |

Auto-detected regions are marked with `auto_detected: true` and can be:
- Adjusted (select and modify)
- Deleted (select and press Delete)
- Verified (add text transcription)

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Delete` / `Backspace` | Delete selected region |
| `Escape` | Cancel current drawing |

## API Usage

```python
from services.annotation import (
    AnnotationStorage,
    AnnotationDataset,
    AnnotatedImage,
    Region,
    Point,
    AnnotationExporter,
)

# Create and save a dataset
storage = AnnotationStorage()
dataset = AnnotationDataset(name="ancient_greek_v1")

image = AnnotatedImage(
    image_path="images/page1.jpg",
    width=1024,
    height=768
)
image.add_region(
    Region.from_bbox(100, 200, 300, 50, text="τοῦ δὲ γενομένου")
)
dataset.add_image(image)

storage.save(dataset)

# Load an existing dataset
loaded = storage.load("ancient_greek_v1")
print(f"Images: {len(loaded.images)}")
print(f"Total regions: {loaded.total_regions}")
print(f"Labeled regions: {loaded.labeled_regions}")

# Export for training
exporter = AnnotationExporter()

# Export for detector training (COCO format)
detector_path = exporter.export_for_detector(loaded, storage)
print(f"Detector export: {detector_path}")

# Export for recognizer training (cropped images + labels)
recognizer_path = exporter.export_for_recognizer(loaded, storage)
print(f"Recognizer export: {recognizer_path}")

# Export both
paths = exporter.export_both(loaded, storage)
```

## Development Setup

The annotation canvas is a React component that communicates with Streamlit. To develop:

1. Start the React dev server:
```bash
cd frontend/annotation_canvas
npm install
npm run dev  # Runs on http://localhost:5174
```

2. Run Streamlit (defaults to development mode):
```bash
uv run streamlit run app.py
```

3. For production, build the component:
```bash
cd frontend/annotation_canvas
npm run build
ANNOTATION_CANVAS_RELEASE=true uv run streamlit run app.py
```

## Integration with Training

After annotation and export:

1. **Detector Training**: Use the COCO JSON export with CRAFT or DBNet training scripts
2. **Recognizer Training**: Use the cropped image-text pairs with CRNN, TrOCR, or PP-OCR training

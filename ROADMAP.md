# Metafrasis Roadmap

This document outlines planned features and future development directions for Metafrasis.

## Current Status

**Version**: Pre-1.0 (Active Development)

**Completed** (as of January 2025):
- ✅ OCR service with multiple engines (Tesseract, Kraken, PP-OCR, CRAFT+recognizers)
- ✅ Interactive annotation tool with auto-detection
- ✅ ML model implementations (CRAFT, DBNet, CRNN, PPOCRModel)
- ✅ Training infrastructure (fine-tuning scripts for all models)
- ✅ Comprehensive test suite (246 tests, 100% passing)
- ✅ Model registry and external hosting system
- ✅ Frontend components (OCR Viewer, Annotation Canvas)

---

## Planned Features

### High Priority

#### 1. OCR Engine Improvements

**EasyOCR Integration** 🎯
- **Goal**: Add EasyOCR as an additional OCR engine option
- **Status**: Mentioned in docs, not implemented
- **Motivation**: EasyOCR has strong multilingual support and different accuracy characteristics
- **Implementation approach**:
  - Create EasyOCR detector and recognizer wrappers
  - Integrate with existing OCREngineFactory
  - Support Ancient Greek if models available
  - Add to engine selection in UI
- **Estimated effort**: 1-2 weeks

**Improved Ancient Greek Models** 🎯
- **Goal**: Improve OCR accuracy for Ancient Greek texts
- **Status**: Current models work but accuracy can be improved
- **Motivation**: Better recognition of polytonic characters and manuscript variations
- **Implementation approach**:
  - Fine-tune Kraken models on more diverse Ancient Greek datasets
  - Train PP-OCR models specifically for Ancient Greek
  - Ensemble multiple models for higher confidence
  - Benchmark against standard test sets
- **Estimated effort**: 4-6 weeks

**OCR Confidence Filtering** 🎯
- **Goal**: Filter and highlight low-confidence OCR results
- **Status**: Confidence scores available, no filtering UI
- **Motivation**: Help users identify uncertain regions for manual review
- **Implementation approach**:
  - Add confidence threshold slider to OCR page
  - Highlight low-confidence words in viewer
  - Export confidence scores with results
  - Statistics on confidence distribution
- **Estimated effort**: 1 week

#### 2. Training UI

**Streamlit Training Interface** 🎯
- **Goal**: Expose training infrastructure through web UI
- **Status**: Training infrastructure exists (~3500 lines), no UI
- **Motivation**: Make fine-tuning accessible to non-technical users
- **Features**:
  - Dataset upload and validation
  - Training configuration (epochs, batch size, learning rate)
  - Live training metrics and progress
  - Model evaluation and comparison
  - Export trained models to HuggingFace Hub
- **Implementation approach**:
  - New Streamlit page: "Train"
  - Background training with progress updates
  - Integration with existing `ml/training/` infrastructure
- **Estimated effort**: 4-5 weeks

### Medium Priority

#### 3. Vision Model Annotation

**GPT-4V / Claude Vision Integration** 📋
- **Status**: Mentioned for dataset annotation, not implemented
- **Motivation**: Bootstrap annotation datasets without manual labeling
- **Implementation**:
  - Vision API integration (OpenAI GPT-4V, Claude 3)
  - Prompt engineering for Ancient Greek text extraction
  - Automatic region detection + transcription
  - Human review workflow
- **Estimated effort**: 2-3 weeks

#### 5. Advanced Features

**Region Merging/Splitting** 📋
- Merge adjacent text regions (lines → paragraphs)
- Split incorrectly merged regions
- Estimated effort: 2 weeks

**Multi-Language Switching** 📋
- Switch between Ancient Greek, Latin, and other languages
- Language-specific model selection
- Estimated effort: 1 week

**Batch Annotation Export** 📋
- Export annotations in multiple formats (COCO, Pascal VOC, YOLO)
- Automated dataset splitting (train/val/test)
- Estimated effort: 1-2 weeks

### Low Priority / Research

#### 6. Training Enhancements

**Complete Data Utilities** 🔬
- **Status**: 8 functions in `ml/training/data/base.py` have `NotImplementedError`
- **Functions needed**:
  - `region_to_polygon()` - Convert rectangular regions to polygon format
  - `compute_polygon_centroid()` - Calculate polygon center point
  - `generate_gaussian_heatmap()` - Create heatmaps for CRAFT training
  - `crop_region()` - Extract region from image
  - `resize_keeping_aspect_ratio()` - Smart image resizing
  - `get_region_mask()` - Binary mask from polygon
  - `compute_shrunk_polygon()` - Shrink polygon for DBNet
  - `compute_distance_map()` - Distance transform for text detection
- **Note**: Detector trainers (CRAFT, DB) have internal implementations, so this is not blocking
- **Estimated effort**: 1-2 weeks

**Distributed Training** 🔬
- Multi-GPU support with PyTorch DDP
- Gradient accumulation improvements
- Estimated effort: 2-3 weeks

**Mixed Precision Training** 🔬
- FP16/BF16 training for faster iteration
- Automatic mixed precision (AMP) integration
- Estimated effort: 1 week

**Data Augmentation** 🔬
- Image augmentation for training (rotation, skew, noise)
- Text-aware augmentation (preserve regions)
- Estimated effort: 2 weeks

#### 7. Developer Experience

**CLI Tool** 🔬
- Command-line interface for batch OCR processing
- Non-interactive mode for scripting
- Estimated effort: 1-2 weeks

**Docker Support** 🔬
- Containerized deployment
- Pre-built images with all dependencies
- Estimated effort: 1 week

**API Server** 🔬
- REST API for OCR processing
- Asynchronous job queue
- Estimated effort: 3-4 weeks

---

## Implementation Priorities

### Phase 1: OCR Improvements (Q1 2025)
1. EasyOCR integration
2. OCR confidence filtering UI
3. Improved Ancient Greek models (fine-tuning)
4. Model benchmarking and accuracy metrics

**Goal**: Improve core OCR accuracy and usability before adding downstream features

### Phase 2: Post-OCR Processing (Q2 2025)
1. Transliteration service (Beta Code, ALA-LC, ISO 843)
2. Translation service (API integration)
3. Multi-language switching

**Goal**: Complete the OCR → Transliteration → Translation pipeline

### Phase 3: Training & Annotation (Q3 2025)
1. Streamlit training UI
2. GPT-4V/Claude annotation integration
3. Batch annotation export formats
4. Region merging/splitting tools

**Goal**: Enable non-technical users to create custom models and improve datasets

### Phase 4: Advanced Features (Q4 2025)
1. Complete data utilities (8 placeholder functions)
2. Distributed training support
3. Mixed precision training
4. CLI tool for batch processing
5. Docker support

**Goal**: Performance optimization and developer experience

---

## Contributing

**High-impact starter tasks**:
- **EasyOCR integration** (clear API surface, immediate value)
- **OCR confidence filtering** (straightforward UI enhancement)
- **Transliteration service** (well-defined scope, independent module)
- **Data utility functions** (unit testing friendly, good first contribution)

---

## Non-Goals

Features we've decided NOT to pursue:

- **Web-based model training**: Too resource-intensive for browser environments
- **Real-time video OCR**: Out of scope for document processing focus
- **Handwriting recognition training**: Use existing pre-trained models (trOCR)
- **Mobile app**: Focus on desktop/server deployment

---

## Feedback

Have ideas for the roadmap? Open an issue with the `enhancement` label or start a discussion.

Last updated: January 2025

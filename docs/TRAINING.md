# Training Infrastructure

## Overview

Complete infrastructure for fine-tuning OCR models on Ancient Greek text, covering both **text detectors** (CRAFT, DB) and **text recognizers** (CRNN, PP-OCR, trOCR).

## Key Principles

1. **All models externally hosted** - Nothing in git, download on-demand
2. **Annotation-driven** - Use the annotation service to create training datasets
3. **Modular architecture** - Abstract base classes for extensibility
4. **Comprehensive evaluation** - Detector and recognizer metrics

## Directory Structure

```
training/
├── __init__.py
│
├── configs/                      # Training configurations
│   ├── craft.yaml               # CRAFT detector config
│   ├── db.yaml                  # DBNet detector config
│   ├── crnn.yaml                # CRNN recognizer config
│   ├── ppocr.yaml               # PP-OCR recognizer config
│   └── trocr_lora.yaml          # trOCR + LoRA config
│
├── data/                         # Dataset conversion
│   ├── __init__.py
│   ├── base.py                  # Shared utilities
│   ├── detectors/               # Detector dataset classes
│   │   ├── __init__.py
│   │   ├── base.py              # DetectorDataset ABC
│   │   ├── craft.py             # CRAFTDataset - region + affinity maps
│   │   └── db.py                # DBDataset - probability + threshold maps
│   └── recognizer_dataset.py    # RecognizerDataset - crop + labels
│
├── finetune/                     # Fine-tuning with base classes
│   ├── __init__.py
│   ├── base.py                  # BaseTrainer ABC
│   ├── utils.py                 # Training utilities
│   ├── detectors/               # Detector trainers
│   │   ├── __init__.py
│   │   ├── base.py              # DetectorTrainer(BaseTrainer)
│   │   ├── craft.py             # CRAFTTrainer - MSE + OHEM loss
│   │   └── db.py                # DBTrainer - BCE + L1 + Dice loss
│   └── recognizers/             # Recognizer trainers
│       ├── __init__.py
│       ├── base.py              # RecognizerTrainer, CTCRecognizerTrainer, TransformerRecognizerTrainer
│       ├── crnn.py              # CRNNTrainer(CTCRecognizerTrainer)
│       ├── ppocr.py             # PPOCRTrainer(CTCRecognizerTrainer)
│       └── trocr.py             # TrOCRTrainer(TransformerRecognizerTrainer)
│
├── evaluate/                     # Evaluation with base classes
│   ├── __init__.py
│   ├── base.py                  # Metric ABC
│   ├── detectors/               # Detector metrics
│   │   ├── __init__.py
│   │   ├── base.py              # DetectorMetric(Metric)
│   │   ├── iou.py               # IoUMetric
│   │   ├── precision_recall.py  # PrecisionRecallF1Metric
│   │   └── map.py               # MeanAPMetric
│   └── recognizers/             # Recognizer metrics
│       ├── __init__.py
│       ├── base.py              # RecognizerMetric(Metric)
│       ├── cer.py               # CERMetric
│       ├── wer.py               # WERMetric
│       └── accuracy.py          # AccuracyMetric
│
└── export/                       # Model export
    ├── __init__.py
    ├── to_onnx.py               # Export to ONNX
    └── to_huggingface.py        # Upload to HuggingFace Hub
```

## Class Hierarchy

### Finetune Module

```
BaseTrainer (training/finetune/base.py)
├── DetectorTrainer (training/finetune/detectors/base.py)
│     ├── CRAFTTrainer (craft.py) - MSE loss with OHEM
│     └── DBTrainer (db.py) - BCE + L1 + Dice loss
└── RecognizerTrainer (training/finetune/recognizers/base.py)
      ├── CTCRecognizerTrainer (base.py) - CTC loss
      │     ├── CRNNTrainer (crnn.py)
      │     └── PPOCRTrainer (ppocr.py) - with gradient accumulation
      └── TransformerRecognizerTrainer (base.py) - CE loss
            └── TrOCRTrainer (trocr.py) - with LoRA adapters
```

### Evaluate Module

```
Metric (training/evaluate/base.py)
├── DetectorMetric (training/evaluate/detectors/base.py)
│     ├── IoUMetric (iou.py)
│     ├── PrecisionRecallF1Metric (precision_recall.py)
│     └── MeanAPMetric (map.py)
└── RecognizerMetric (training/evaluate/recognizers/base.py)
      ├── CERMetric (cer.py)
      ├── WERMetric (wer.py)
      └── AccuracyMetric (accuracy.py)
```

## Model Types

### Detectors

| Model | Architecture | Training Maps |
|-------|-------------|---------------|
| CRAFT | VGG16-BN + U-Net | Region score + Affinity score |
| DB | ResNet18 + FPN | Probability + Threshold + Binary |

### Recognizers

| Model | Architecture | Loss |
|-------|-------------|------|
| CRNN | VGG + BiLSTM | CTC |
| PP-OCR | MobileNetV3 + BiLSTM | CTC |
| trOCR | ViT + GPT-2 | Cross-entropy (LoRA) |

## Usage

### 1. Create Annotation Dataset

Use the Streamlit annotation tool to create labeled datasets:

```bash
uv run streamlit run app/main.py
```

Navigate to the "Annotate" tab and:
1. Create a new dataset
2. Upload images
3. Draw regions and enter text labels
4. Verify annotations
5. Download dataset

### 2. Export Training Data

**For Detectors:**

```python
from services.annotation import AnnotationStorage
from training.data import CRAFTDataset, DBDataset

storage = AnnotationStorage()
dataset = storage.load("my_dataset")

# Export for CRAFT
craft_dataset = CRAFTDataset(dataset, storage)
craft_dataset.export(Path("data/exports/detector/craft"))

# Export for DB
db_dataset = DBDataset(dataset, storage)
db_dataset.export(Path("data/exports/detector/db"))
```

**For Recognizers:**

```python
from training.data import RecognizerDataset

recognizer_dataset = RecognizerDataset(
    dataset,
    storage,
    target_height=32,
    include_unverified=False  # Only use verified annotations
)
recognizer_dataset.export(Path("data/exports/recognizer"))
```

### 3. Train Models

**Using the Trainer API:**

```python
from training.finetune import CRNNTrainer, CRAFTTrainer, TrOCRTrainer

# CRNN Recognizer
config = {
    "data_dir": "data/exports/recognizer",
    "output_dir": "training/outputs/crnn",
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
}
trainer = CRNNTrainer(config)
results = trainer.train()

# CRAFT Detector
config = {
    "data_dir": "data/exports/detector/craft",
    "output_dir": "training/outputs/craft",
    "epochs": 100,
    "batch_size": 8,
    "learning_rate": 0.0001,
}
trainer = CRAFTTrainer(config)
results = trainer.train()

# trOCR with LoRA
config = {
    "data_dir": "data/exports/recognizer",
    "output_dir": "training/outputs/trocr",
    "model_name": "microsoft/trocr-base-handwritten",
    "lora_r": 16,
    "lora_alpha": 32,
    "epochs": 20,
    "batch_size": 8,
}
trainer = TrOCRTrainer(config)
results = trainer.train()
```

**Using YAML configs:**

```bash
# Load config and train
python -c "
from training.finetune import CRNNTrainer, load_config
config = load_config('training/configs/crnn.yaml')
trainer = CRNNTrainer(config)
trainer.train()
"
```

### 4. Evaluate Models

```python
from training.evaluate import (
    CERMetric, WERMetric, AccuracyMetric,
    IoUMetric, PrecisionRecallF1Metric, MeanAPMetric,
)

# Recognizer evaluation
cer = CERMetric()
wer = WERMetric()
accuracy = AccuracyMetric()

predictions = ["hello world", "ancient greek"]
targets = ["hello world", "ancient greeks"]

cer_result = cer.compute(predictions, targets)
wer_result = wer.compute(predictions, targets)
acc_result = accuracy.compute(predictions, targets)

print(f"CER: {cer_result.value:.4f}")
print(f"WER: {wer_result.value:.4f}")
print(f"Accuracy: {acc_result.value:.4f}")

# Detector evaluation
import numpy as np

iou = IoUMetric(iou_threshold=0.5)
prf = PrecisionRecallF1Metric(iou_threshold=0.5)

pred_polygons = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])]
gt_polygons = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])]

iou_result = iou.compute(pred_polygons, gt_polygons)
prf_result = prf.compute(pred_polygons, gt_polygons)

print(f"IoU: {iou_result.value:.4f}")
print(f"Precision: {prf_result.details['precision']:.4f}")
print(f"Recall: {prf_result.details['recall']:.4f}")
print(f"F1: {prf_result.details['f1']:.4f}")
```

**Batch evaluation with accumulation:**

```python
from training.evaluate import CERMetric

cer = CERMetric()

# Process multiple batches
for batch_preds, batch_targets in dataloader:
    cer.update(batch_preds, batch_targets)

# Get aggregated result
result = cer.aggregate()
print(f"Overall CER: {result.value:.4f}")

# Reset for next evaluation
cer.reset()
```

### 5. Export Models

**To ONNX:**
```python
from training.export import export_to_onnx

export_to_onnx(
    model,
    output_path=Path("models/crnn.onnx"),
    input_shape=(1, 1, 32, 128),
)
```

**To HuggingFace Hub:**
```python
from training.export import upload_to_hub, upload_lora_adapter

# Upload LoRA adapter
url = upload_lora_adapter(
    adapter_path=Path("training/outputs/trocr_lora/best_adapter"),
    repo_id="your-username/trocr-ancient-greek-lora",
    base_model="microsoft/trocr-base-handwritten",
)
```

## Configuration

### CRNN Config (`training/configs/crnn.yaml`)

```yaml
data_dir: data/exports/recognizer
output_dir: training/outputs/crnn

hidden_size: 256
num_layers: 2

epochs: 100
batch_size: 32
learning_rate: 0.001
weight_decay: 0.0001
img_height: 32
img_width: 128

scheduler: cosine
warmup_epochs: 5
patience: 10
```

### PP-OCR Config (`training/configs/ppocr.yaml`)

```yaml
data_dir: data/exports/recognizer
output_dir: training/outputs/ppocr

backbone_scale: 0.5
hidden_size: 256

epochs: 100
batch_size: 64
learning_rate: 0.001
accumulation_steps: 4  # Gradient accumulation
img_height: 32
img_width: 320

patience: 15
```

### trOCR LoRA Config (`training/configs/trocr_lora.yaml`)

```yaml
data_dir: data/exports/recognizer
output_dir: training/outputs/trocr_lora

model_name: microsoft/trocr-base-handwritten
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1
target_modules: [q_proj, v_proj]

epochs: 20
batch_size: 8
learning_rate: 0.00005
max_length: 64
num_beams: 4

patience: 10
```

### CRAFT Config (`training/configs/craft.yaml`)

```yaml
data_dir: data/exports/detector/craft
output_dir: training/outputs/craft

pretrained_backbone: true

epochs: 100
batch_size: 8
learning_rate: 0.0001
img_size: 768
neg_ratio: 3.0  # OHEM negative ratio

patience: 10
```

### DB Config (`training/configs/db.yaml`)

```yaml
data_dir: data/exports/detector/db
output_dir: training/outputs/db

pretrained_backbone: true

epochs: 100
batch_size: 8
learning_rate: 0.0001
img_size: 640

# Loss weights
bce_weight: 1.0
l1_weight: 10.0
dice_weight: 1.0

patience: 10
```

## Extending the Framework

### Adding a New Recognizer

```python
from training.finetune.recognizers import CTCRecognizerTrainer
import torch.nn as nn

class MyRecognizerTrainer(CTCRecognizerTrainer):
    @property
    def name(self) -> str:
        return "my_recognizer"

    def create_model(self) -> nn.Module:
        # Return your model
        return MyRecognizerModel(
            num_classes=len(self.char_to_idx),
            hidden_size=self.config.get("hidden_size", 256),
        )
```

### Adding a New Detector

```python
from training.finetune.detectors import DetectorTrainer
import torch.nn as nn

class MyDetectorTrainer(DetectorTrainer):
    @property
    def name(self) -> str:
        return "my_detector"

    @property
    def dataset_class(self) -> type:
        return MyDetectorDataset

    def create_model(self) -> nn.Module:
        return MyDetectorModel()

    def create_criterion(self) -> nn.Module:
        return MyDetectorLoss()

    def compute_loss(self, outputs, targets):
        return self.criterion(outputs, targets)
```

### Adding a New Metric

```python
from training.evaluate.recognizers import RecognizerMetric
from training.evaluate.base import MetricResult

class MyMetric(RecognizerMetric):
    @property
    def name(self) -> str:
        return "my_metric"

    def compute(self, predictions, targets) -> MetricResult:
        # Compute your metric
        value = ...
        return MetricResult(
            name=self.name,
            value=value,
            details={"extra_info": ...}
        )
```

## Dependencies

```toml
[project.optional-dependencies]
training = [
    # Core training
    "torch>=2.0.0",
    "torchvision>=0.15.0",

    # LoRA fine-tuning
    "peft>=0.8.0",
    "transformers>=4.37.0",

    # Evaluation
    "shapely>=2.0.0",  # Polygon IoU

    # HuggingFace Hub
    "huggingface-hub>=0.20.0",
    "datasets>=2.16.0",

    # ONNX export
    "onnx>=1.14.0",
    "onnxruntime>=1.16.0",
    "onnxsim>=0.4.0",  # Optional: ONNX simplification

    # Visualization
    "matplotlib>=3.8.0",
]
```

## Model Registry

After training, update `models/registry.json`:

```json
{
  "crnn": {
    "base": {
      "url": "your-username/crnn-base",
      "type": "huggingface"
    },
    "finetuned": {
      "url": "your-username/crnn-ancient-greek",
      "type": "huggingface"
    }
  },
  "trocr": {
    "base": {
      "url": "microsoft/trocr-base-handwritten",
      "type": "huggingface"
    },
    "finetuned": {
      "url": "your-username/trocr-ancient-greek-lora",
      "type": "huggingface_adapter"
    }
  }
}
```

## Training Workflow Summary

1. **Annotate** - Create labeled dataset using annotation tool
2. **Export** - Convert annotations to detector/recognizer format
3. **Train** - Instantiate trainer with config and call `train()`
4. **Evaluate** - Compute metrics on validation set
5. **Export** - Convert to ONNX and/or upload to HuggingFace
6. **Register** - Update `models/registry.json` with new model URL

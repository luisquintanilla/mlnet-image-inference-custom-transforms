# Models Guide

> How to download, convert, and prepare ONNX models for use with this library.

## Overview

This library runs ONNX models locally via ONNX Runtime. Models must be in `.onnx` format. Most HuggingFace models are in PyTorch format and need conversion using the [Optimum](https://huggingface.co/docs/optimum/en/exporters/onnx/usage_guides/export_a_model) library.

---

## Prerequisites

Install the HuggingFace CLI and Optimum with ONNX Runtime support:

```bash
pip install huggingface-hub optimum[onnxruntime]
```

Verify the installation:

```bash
optimum-cli --help
huggingface-cli --help
```

---

## Downloading and Converting Models

### Quick Start: One-Command Export

The easiest approach is `optimum-cli export onnx`, which downloads the PyTorch model and converts it to ONNX in one step:

```bash
# Image Classification (ViT)
optimum-cli export onnx --model google/vit-base-patch16-224 models/vit/

# Image Embeddings (CLIP vision encoder)
optimum-cli export onnx --model openai/clip-vit-base-patch32 models/clip/

# Object Detection (YOLOv8 — via ultralytics export, see below)

# Image Segmentation (SegFormer)
optimum-cli export onnx --model nvidia/segformer-b0-finetuned-ade-512-512 models/segformer/

# Zero-Shot Classification (CLIP — full model with text encoder)
optimum-cli export onnx --model openai/clip-vit-base-patch32 models/clip-full/
```

### Alternative: Download Pre-Exported Models

Some models already have ONNX versions on HuggingFace:

```bash
# Download a specific file
huggingface-cli download optimum/vit-base-patch16-224 --local-dir models/vit/

# Download only ONNX files from a repo
huggingface-cli download openai/clip-vit-base-patch32 \
    --include "*.onnx" "*.json" \
    --local-dir models/clip/
```

### YOLOv8 (Special Case)

YOLOv8 uses the Ultralytics library for export rather than Optimum:

```bash
pip install ultralytics

python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640)
"
# Move the exported model
mkdir -p models/yolov8
mv yolov8n.onnx models/yolov8/model.onnx
```

---

## Expected Directory Structures

### Classification (ViT)

```
models/vit/
├── model.onnx                  ← required: the ONNX model
├── config.json                 ← model config (num_labels, id2label mapping)
├── preprocessor_config.json    ← preprocessing config (mean, std, image_size)
└── tokenizer.json              ← (not used for vision-only models)
```

**Key file: `preprocessor_config.json`**

```json
{
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [0.485, 0.456, 0.406],
  "image_std": [0.229, 0.224, 0.225],
  "rescale_factor": 0.00392156862745098,
  "size": { "height": 224, "width": 224 }
}
```

Map this to `PreprocessorConfig`:

```csharp
var config = new PreprocessorConfig
{
    Mean = [0.485f, 0.456f, 0.406f],
    Std = [0.229f, 0.224f, 0.225f],
    DoRescale = true,
    RescaleFactor = 1f / 255f,
    DoNormalize = true,
    ImageSize = (224, 224)
};
// Or simply use the built-in preset:
var config = PreprocessorConfig.ImageNet;
```

**Key file: `config.json` — extracting labels**

```json
{
  "id2label": {
    "0": "tench",
    "1": "goldfish",
    "2": "great white shark"
  }
}
```

Use these as the `Labels` array in `OnnxImageClassificationOptions`.

### Embeddings (CLIP)

```
models/clip/
├── model.onnx                  ← vision encoder ONNX model
├── config.json                 ← model config (hidden_size = embedding dimension)
└── preprocessor_config.json    ← CLIP-specific preprocessing config
```

For zero-shot classification, you also need the text model:

```
models/clip-full/
├── model.onnx                  ← combined model, or separate files:
├── text_model.onnx             ← text encoder
├── vision_model.onnx           ← vision encoder
├── vocab.json                  ← BPE vocabulary for ClipTokenizer
├── merges.txt                  ← BPE merge rules for ClipTokenizer
├── config.json
└── preprocessor_config.json
```

### Detection (YOLOv8)

```
models/yolov8/
├── model.onnx                  ← YOLOv8 ONNX model
└── metadata.yaml               ← class names (from Ultralytics export)
```

YOLOv8 uses simple rescaling (no normalization), so use:

```csharp
var config = PreprocessorConfig.YOLOv8;
```

### Segmentation (SegFormer)

```
models/segformer/
├── model.onnx                  ← SegFormer ONNX model
├── config.json                 ← model config (id2label for class names)
└── preprocessor_config.json    ← preprocessing config
```

Use the `SegFormer` preset:

```csharp
var config = PreprocessorConfig.SegFormer;
```

---

## Verifying a Model

After downloading, verify the model loads correctly:

```bash
# Check model inputs/outputs with Python
python -c "
import onnxruntime as ort
session = ort.InferenceSession('models/vit/model.onnx')
print('Inputs:', [(i.name, i.shape, i.type) for i in session.get_inputs()])
print('Outputs:', [(o.name, o.shape, o.type) for o in session.get_outputs()])
"
```

Expected output for ViT classification:

```
Inputs: [('pixel_values', [1, 3, 224, 224], 'tensor(float)')]
Outputs: [('logits', [1, 1000], 'tensor(float)')]
```

Expected output for CLIP vision encoder:

```
Inputs: [('pixel_values', [1, 3, 224, 224], 'tensor(float)')]
Outputs: [('last_hidden_state', [1, 50, 768], 'tensor(float)'), ('pooler_output', [1, 768], 'tensor(float)')]
```

The library uses `ModelMetadataDiscovery` to auto-detect these names and shapes at runtime, so you don't need to specify them manually.

---

## Reading preprocessor_config.json

The `PreprocessorConfig` record maps directly to HuggingFace's `preprocessor_config.json`. Here's a reference for how fields map:

| HuggingFace JSON field | `PreprocessorConfig` property | Notes |
|---|---|---|
| `image_mean` | `Mean` | Per-channel `float[]` |
| `image_std` | `Std` | Per-channel `float[]` |
| `do_rescale` | `DoRescale` | Usually `true` |
| `rescale_factor` | `RescaleFactor` | Usually `1/255` |
| `do_normalize` | `DoNormalize` | Usually `true` (except YOLOv8) |
| `size.height` / `size.width` | `ImageSize` | Target resize dimensions |
| `do_center_crop` | `DoCenterCrop` | `true` for ViT/CLIP |
| `crop_size.height` / `crop_size.width` | `CropSize` | Center crop dimensions |

If a model uses standard ImageNet preprocessing, you can skip reading the JSON and use `PreprocessorConfig.ImageNet` directly.

---

## Recommended Models by Task

| Task | Model | HuggingFace ID | Preset | Notes |
|---|---|---|---|---|
| Classification | ViT-Base | `google/vit-base-patch16-224` | `ImageNet` | 1000 ImageNet classes, 86M params |
| Classification | DeiT-Small | `facebook/deit-small-patch16-224` | `ImageNet` | Efficient ViT variant |
| Embeddings | CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | `CLIP` | 512-dim embeddings |
| Embeddings | DINOv2-Small | `facebook/dinov2-small` | `DINOv2` | 384-dim, good for similarity |
| Detection | YOLOv8n | `ultralytics/yolov8n` | `YOLOv8` | 80 COCO classes, fast |
| Segmentation | SegFormer-B0 | `nvidia/segformer-b0-finetuned-ade-512-512` | `SegFormer` | 150 ADE20K classes |
| Zero-Shot | CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | `CLIP` | Needs both vision + text encoders |

---

## Troubleshooting

### "Model not found" error

Ensure the path points to the actual `.onnx` file, not just the directory:

```csharp
// ✅ Correct — points to the .onnx file
ModelPath = "models/vit/model.onnx"

// ❌ Wrong — points to the directory
ModelPath = "models/vit/"
```

### Input shape mismatch

If you see a shape error, check that `PreprocessorConfig.ImageSize` matches the model's expected input. For example, ViT expects 224×224 but SegFormer expects 512×512.

### Model has dynamic axes

Some ONNX exports use dynamic axes (e.g., batch size = `-1`). This is fine — the library always sends batch size 1. If you see warnings about dynamic shapes, they can be safely ignored.

### Large model files

ONNX models can be 100MB–1GB+. Consider:
- Using quantized models (`--fp16` or `--int8` flags in optimum-cli).
- Storing models outside the Git repository and downloading them at build/run time.
- Using Git LFS if models must be in the repo.

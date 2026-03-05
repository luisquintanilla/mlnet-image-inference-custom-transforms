# ML.NET Custom Image Inference Transforms

Custom ML.NET transforms for image inference and generation backed by local ONNX models. Provides `IEstimator<T>`/`ITransformer` implementations with [Microsoft.Extensions.AI](https://learn.microsoft.com/dotnet/ai/microsoft-extensions-ai) (MEAI) integration.

[![Build](https://github.com/luisquintanilla/mlnet-image-inference-custom-transforms/actions/workflows/ci.yml/badge.svg)](https://github.com/luisquintanilla/mlnet-image-inference-custom-transforms/actions/workflows/ci.yml)

## Packages

| Package | Description | Status |
|---|---|---|
| **MLNet.Image.Core** | Image preprocessing, `MLImage↔DataContent` conversion, result types (`BoundingBox`, `SegmentationMask`, `DepthMap`) | ✅ |
| **MLNet.Image.Tokenizers** | CLIP tokenizer extending `Microsoft.ML.Tokenizers` | ✅ |
| **MLNet.ImageInference.Onnx** | ML.NET transforms: classification, detection, segmentation, embeddings, zero-shot, depth estimation | ✅ |
| **MLNet.ImageGeneration.OnnxGenAI** | Text-to-image generation (Stable Diffusion via OnnxRuntime) | ✅ |

## Quick Start

### Image Classification with ViT

```csharp
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Classification;

var options = new OnnxImageClassificationOptions
{
    ModelPath = "models/vit/model.onnx",
    PreprocessorConfig = PreprocessorConfig.ImageNet,
    TopK = 5
};

var estimator = new OnnxImageClassificationEstimator(options);
using var transformer = estimator.Fit(null!);

using var image = MLImage.CreateFromFile("photo.jpg");
var predictions = transformer.Classify(image);

foreach (var (label, probability) in predictions)
    Console.WriteLine($"  {label}: {probability:P2}");
```

### Image Embeddings with MEAI

```csharp
using Microsoft.Extensions.AI;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.MEAI;

IEmbeddingGenerator<MLImage, Embedding<float>> generator =
    new OnnxImageEmbeddingGenerator("models/clip/model.onnx");

using var image = MLImage.CreateFromFile("photo.jpg");
var embeddings = await generator.GenerateAsync([image]);
// embeddings[0].Vector is ReadOnlyMemory<float> of length 512
```

### Zero-Shot Classification with CLIP

```csharp
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Embeddings;

// Encode image and candidate labels through CLIP, then compare via cosine similarity
var embeddingOptions = new OnnxImageEmbeddingOptions
{
    ModelPath = "models/clip/model.onnx",
    PreprocessorConfig = PreprocessorConfig.CLIP,
    Pooling = PoolingStrategy.ClsToken,
    Normalize = true
};

using var transformer = new OnnxImageEmbeddingTransformer(embeddingOptions);
using var image = MLImage.CreateFromFile("photo.jpg");
float[] imageEmbedding = transformer.GenerateEmbedding(image);
// Compare with text embeddings of candidate labels via cosine similarity
```

## Supported Tasks

| Task | Status | Package | MEAI Interface |
|---|---|---|---|
| Image Classification | ✅ | `MLNet.ImageInference.Onnx` | — |
| Object Detection | ✅ | `MLNet.ImageInference.Onnx` | — |
| Image Segmentation | ✅ | `MLNet.ImageInference.Onnx` | — |
| Image Embeddings | ✅ | `MLNet.ImageInference.Onnx` | `IEmbeddingGenerator<MLImage, Embedding<float>>` |
| Zero-Shot Classification | ✅ | `MLNet.ImageInference.Onnx` | — |
| Depth Estimation | ✅ | `MLNet.ImageInference.Onnx` | — |
| Text-to-Image Generation | ✅ | `MLNet.ImageGeneration.OnnxGenAI` | — |

## Samples

| Sample | Description | Directory |
|---|---|---|
| Image Classification | ViT/SqueezeNet classification with direct + ML.NET pipeline APIs | [`samples/ImageClassification/`](samples/ImageClassification/) |
| Object Detection | YOLOv8 detection with bounding boxes + NMS | [`samples/ObjectDetection/`](samples/ObjectDetection/) |
| Image Segmentation | SegFormer semantic segmentation with mask visualization | [`samples/ImageSegmentation/`](samples/ImageSegmentation/) |
| Image Embeddings | CLIP/DINOv2 embeddings with MEAI integration | [`samples/ImageEmbeddings/`](samples/ImageEmbeddings/) |
| Zero-Shot Classification | CLIP vision+text zero-shot classification | [`samples/ZeroShotClassification/`](samples/ZeroShotClassification/) |
| Depth Estimation | MiDaS/DPT monocular depth estimation | [`samples/DepthEstimation/`](samples/DepthEstimation/) |
| Text-to-Image | Stable Diffusion generation with direct + MEAI APIs | [`samples/TextToImage/`](samples/TextToImage/) |

## Architecture Overview

Every inference task follows a **composed facade pattern** — three reusable sub-transforms chained by a task-level orchestrator:

```
MLImage → ① Preprocess → ② ONNX Score → ③ PostProcess → Result
```

| Stage | Component | Description |
|---|---|---|
| ① Preprocess | `ImagePreprocessingTransformer` | Bilinear resize → Rescale → Per-channel Normalize → CHW tensor |
| ② ONNX Score | `OnnxImageScoringTransformer` | `InferenceSession.Run()` with batch support |
| ③ PostProcess | Task-specific | Softmax / Pooling+L2 / NMS / Argmax / Depth normalize |

Each task transformer composes the shared preprocessing and scoring sub-transforms with task-specific post-processing. Shared base classes (`OnnxImageCursorBase`, `OnnxImageDataViewBase`, `OnnxImageEstimatorBase`) eliminate duplication across all tasks.

### Batch Inference

All inference transforms support **lookahead batching** through ML.NET's `IDataView` cursors:

- **Configurable `BatchSize`** (default 32) — cursors pre-fetch upcoming rows and score them in a single ONNX call
- **True tensor batching** for models with dynamic batch dimensions (MobileNet, CLIP, DINOv2, ResNet) — multiple images are stacked into one `N×C×H×W` tensor
- **Per-image fallback** for fixed-batch models (SqueezeNet, YOLO, SegFormer) — images are scored individually within the batch window
- **Batch convenience methods** on every transformer: `ClassifyBatch`, `DetectBatch`, `GenerateEmbeddingBatch`, `SegmentBatch`, `ZeroShotClassifyBatch`

For full details, see [docs/architecture.md](docs/architecture.md).

## Model Requirements

Models are HuggingFace-compatible ONNX exports. Use `optimum-cli` to export:

```bash
pip install optimum[onnxruntime]

# Classification (ViT)
optimum-cli export onnx --model google/vit-base-patch16-224 models/vit/

# Embeddings (CLIP)
optimum-cli export onnx --model openai/clip-vit-base-patch32 models/clip/
```

See [docs/models-guide.md](docs/models-guide.md) for supported models, preprocessing configs, and detailed export instructions.

## MEAI Integration

The library integrates with [Microsoft.Extensions.AI](https://learn.microsoft.com/dotnet/ai/microsoft-extensions-ai) through adapter classes:

- **`OnnxImageEmbeddingGenerator`** — implements `IEmbeddingGenerator<MLImage, Embedding<float>>`
- **`OnnxImageGenerator`** — generates images from text prompts
- **`MLImage ↔ DataContent`** — extension methods bridge ML.NET images with MEAI content types

## Building

```bash
dotnet restore
dotnet build MLNet.Image.slnx
```

### Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- ONNX models (see [Model Requirements](#model-requirements))

## Testing

**144 tests** across three test projects — all passing.

| Test Project | Tests | Description |
|---|---|---|
| Core | 41 | Image preprocessing, conversions, result types |
| Tokenizers | 14 | CLIP tokenizer encoding/decoding |
| Inference | 89 | End-to-end ONNX inference across 8+ models, all tasks (incl. batch) |

### Tested Models

All supported tasks are validated against real ONNX models across all preprocessor presets (ImageNet, CLIP, DINOv2, YOLOv8, SegFormer, MiDaS, DPT):

| Model | Task | Notes |
|---|---|---|
| SqueezeNet 1.0 | Classification | ImageNet-1K labels, top-K predictions |
| MobileNetV2 | Classification | Lightweight alternative model |
| YOLOv8n | Object Detection | Bounding box NMS post-processing |
| SegFormer-b0 | Segmentation | Dynamic shape handling (bug found & fixed in Phase 4) |
| CLIP ViT-B/32 | Embeddings / Zero-Shot | Image+text embedding, cosine similarity |
| DINOv2 ViT-S/14 | Embeddings | 384-dim, DINOv2 preset, self-supervised |
| ResNet-50 v1.7 | Classification | 1000 ImageNet classes, dynamic batch |
| DeepLabV3-ResNet50 | Segmentation | 21 Pascal VOC classes, 520×520 |

## Related Projects

| Repository | Description |
|---|---|
| [mlnet-text-inference-custom-transforms](https://github.com/luisquintanilla/mlnet-text-inference-custom-transforms) | ML.NET custom transforms for text inference (NER, sentiment, embeddings) |
| [mlnet-audio-custom-transforms](https://github.com/luisquintanilla/mlnet-audio-custom-transforms) | ML.NET custom transforms for audio inference (speech-to-text, classification) |
| [model-packages-prototype](https://github.com/luisquintanilla/model-packages-prototype) | ONNX models packaged as NuGet packages with metadata |
| [dotnet-model-garden-prototype](https://github.com/luisquintanilla/dotnet-model-garden-prototype) | Curated model catalog + download CLI |

## License

[MIT](LICENSE)

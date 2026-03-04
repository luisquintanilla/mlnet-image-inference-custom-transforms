# ML.NET Custom Image Inference Transforms

Custom ML.NET transforms for image inference and generation backed by local ONNX models. Provides `IEstimator<T>`/`ITransformer` implementations with [Microsoft.Extensions.AI](https://learn.microsoft.com/dotnet/ai/microsoft-extensions-ai) (MEAI) integration.

[![Build](https://github.com/luisquintanilla/mlnet-image-inference-custom-transforms/actions/workflows/ci.yml/badge.svg)](https://github.com/luisquintanilla/mlnet-image-inference-custom-transforms/actions/workflows/ci.yml)

## Packages

| Package | Description | Status |
|---|---|---|
| **MLNet.Image.Core** | Image preprocessing, `MLImage↔DataContent` conversion, result types (`BoundingBox`, `SegmentationMask`) | ✅ |
| **MLNet.Image.Tokenizers** | CLIP tokenizer extending `Microsoft.ML.Tokenizers` | ✅ |
| **MLNet.ImageInference.Onnx** | ML.NET transforms: classification, embeddings, detection, segmentation | ✅ |
| **MLNet.ImageGeneration.OnnxGenAI** | Text-to-image generation via OnnxRuntime GenAI | 🚧 Planned |

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
| Image Classification | ✅ Implemented | `MLNet.ImageInference.Onnx` | — |
| Image Embeddings | ✅ Implemented | `MLNet.ImageInference.Onnx` | `IEmbeddingGenerator<MLImage, Embedding<float>>` |
| Object Detection | ✅ Implemented | `MLNet.ImageInference.Onnx` | — |
| Image Segmentation | ✅ Implemented | `MLNet.ImageInference.Onnx` | — |
| Zero-Shot Classification | ✅ Implemented | `MLNet.ImageInference.Onnx` | — |
| Text-to-Image Generation | 📋 Planned | `MLNet.ImageGeneration.OnnxGenAI` | `IImageGenerator` (experimental) |

## Samples

| Sample | Description | Directory |
|---|---|---|
| Image Classification | ViT classification with direct + ML.NET pipeline APIs | [`samples/ImageClassification/`](samples/ImageClassification/) |
| Image Embeddings | CLIP/DINOv2 embeddings with MEAI integration | [`samples/ImageEmbeddings/`](samples/ImageEmbeddings/) |
| Text-to-Image | Planned Stable Diffusion generation API preview | [`samples/TextToImage/`](samples/TextToImage/) |

## Architecture Overview

Every inference task follows a consistent **three-stage pipeline**:

```
MLImage → ① Preprocess → ② ONNX Score → ③ PostProcess → Result
```

| Stage | Component | Description |
|---|---|---|
| ① Preprocess | `HuggingFaceImagePreprocessor` | Resize → Rescale → Per-channel Normalize → CHW tensor |
| ② ONNX Score | `OnnxSessionPool` | `InferenceSession.Run()` with thread-safe session pooling |
| ③ PostProcess | Task-specific | Softmax / Pooling+L2 / NMS / Argmax |

Each task follows a **six-component pattern**: Options → PostProcessor → Estimator → Transformer → MLContext extension → MEAI adapter.

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
- **`OnnxImageGenerator`** — planned `IImageGenerator` implementation (experimental)
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

**123 tests** across three test projects — all passing.

| Test Project | Tests | Description |
|---|---|---|
| Core | 37 | Image preprocessing, conversions, result types |
| Tokenizers | 14 | CLIP tokenizer encoding/decoding |
| Inference | 72 | End-to-end ONNX inference across 8 models, all 5 tasks |

### Tested Models

All five supported tasks are validated against 8 real ONNX models:

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
| [mlnet-text-inference-custom-transforms](https://github.com/dotnet/mlnet-text-inference-custom-transforms) | ML.NET custom transforms for text inference (NER, sentiment, embeddings) |
| [mlnet-audio-inference-custom-transforms](https://github.com/dotnet/mlnet-audio-inference-custom-transforms) | ML.NET custom transforms for audio inference (speech-to-text, classification) |
| [model-packages-prototype](https://github.com/dotnet/model-packages-prototype) | ONNX models packaged as NuGet packages with metadata |
| [dotnet-model-garden-prototype](https://github.com/dotnet/dotnet-model-garden-prototype) | Curated model catalog + download CLI |

## License

[MIT](LICENSE)

# MLNet.ImageInference.Onnx

Custom ML.NET transforms for image inference tasks backed by local ONNX models.

## Supported Tasks

| Task | Facade Estimator | Status |
|---|---|---|
| Image Classification | `OnnxImageClassificationEstimator` | ✅ |
| Image Embeddings | `OnnxImageEmbeddingEstimator` | ✅ |
| Object Detection | `OnnxObjectDetectionEstimator` | 📋 Planned |
| Image Segmentation | `OnnxImageSegmentationEstimator` | 📋 Planned |
| Zero-Shot Classification | `OnnxZeroShotImageClassificationEstimator` | 📋 Planned |

## Architecture

Three-stage pipeline (shared with text and audio repos):

```
MLImage → [HuggingFace Preprocess] → float[] tensor → [ONNX Score] → raw output → [Post-Process] → result
```

## MEAI Integration

- `IEmbeddingGenerator<MLImage, Embedding<float>>` via `OnnxImageEmbeddingGenerator`

## Dependencies

- `MLNet.Image.Core` — preprocessing and MLImage↔DataContent conversion
- `Microsoft.ML` — IEstimator/ITransformer framework
- `Microsoft.ML.OnnxRuntime.Managed` — ONNX inference (runtime-neutral)
- `Microsoft.Extensions.AI.Abstractions` — MEAI interfaces

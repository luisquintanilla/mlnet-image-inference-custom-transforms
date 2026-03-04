# MLNet.Image.Core

Image preprocessing primitives and MLImage↔DataContent conversion for ML.NET image inference transforms.

## Purpose

Fills the gap between ML.NET's built-in image transforms and what HuggingFace vision models expect:

- **ML.NET's `ExtractPixels`** does: `(pixel + offset) * scale` — global offset+scale only
- **HuggingFace models need**: `(pixel / 255.0 - mean[c]) / std[c]` — per-channel mean/std normalization

This package provides `HuggingFaceImagePreprocessor` for that per-channel normalization, plus `MLImage↔DataContent` conversion helpers for bridging ML.NET and Microsoft.Extensions.AI.

## Key Design

**No custom `ImageData` type.** Uses `MLImage` from `Microsoft.ML.Data` directly — the existing ML.NET image primitive backed by SkiaSharp.

## Dependencies

- `Microsoft.ML` + `Microsoft.ML.ImageAnalytics` — provides `MLImage`
- `System.Numerics.Tensors` — tensor math
- `Microsoft.Extensions.AI.Abstractions` — provides `DataContent` for MEAI bridging

# MLNet.ImageGeneration.OnnxGenAI

Custom ML.NET transforms for text-to-image generation backed by ONNX models (Stable Diffusion, SDXL).

## Status

🚧 **Under Development** — Stub implementation. Full implementation pending:
- `Microsoft.ML.OnnxRuntimeGenAI` availability for net10.0
- Stabilization of `IImageGenerator` in Microsoft.Extensions.AI

## Planned Pipeline

```
Text prompt → [CLIP Text Encoder] → text embeddings
    → [UNet Denoising Loop] → latent → [VAE Decoder] → pixel image
```

## Reference

- [ElBruno.Text2Image](https://github.com/elbruno/ElBruno.Text2Image) — ONNX Stable Diffusion reference
- [MEAI IImageGenerator](https://learn.microsoft.com/en-us/dotnet/api/microsoft.extensions.ai) — Experimental interface

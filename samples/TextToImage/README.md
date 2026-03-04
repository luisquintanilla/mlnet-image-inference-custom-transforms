# Text-to-Image Generation Sample (Preview)

> ⚠️ **Planned Feature** — This sample demonstrates the planned text-to-image generation API. Full implementation requires `Microsoft.ML.OnnxRuntimeGenAI` for net10.0.

Demonstrates the API shape for text-to-image generation using Stable Diffusion ONNX models, including the MEAI `OnnxImageGenerator` adapter.

## Reference

This implementation is inspired by [ElBruno.Text2Image](https://github.com/elbruno/ElBruno.Text2Image), which demonstrates Stable Diffusion inference in .NET using ONNX Runtime.

## Setup (When Implemented)

1. **Download the model:**
   ```bash
   pip install optimum[onnxruntime]
   optimum-cli export onnx --model stabilityai/stable-diffusion-2-1 models/sd/
   ```

2. **Expected model directory structure:**
   ```
   models/sd/
   ├── text_encoder/
   │   └── model.onnx
   ├── unet/
   │   └── model.onnx
   ├── vae_decoder/
   │   └── model.onnx
   ├── tokenizer/
   │   ├── vocab.json
   │   └── merges.txt
   └── scheduler/
       └── scheduler_config.json
   ```

3. **Run the sample:**
   ```bash
   dotnet run -- models/sd/
   ```

## Pipeline (Planned)

```
Text prompt → [CLIP Text Encoder] → text embeddings
    → [UNet Denoising Loop (N steps)] → latent tensor
    → [VAE Decoder] → pixel tensor → PNG image
```

## API Preview

### Direct API
```csharp
var options = new OnnxImageGenerationOptions
{
    ModelDirectory = "models/sd/",
    NumInferenceSteps = 20,
    GuidanceScale = 7.5f,
    Width = 512,
    Height = 512
};

using var transformer = new OnnxImageGenerationTransformer(options);
byte[] image = transformer.Generate("a cat sitting on a beach");
```

### MEAI API (Planned)
```csharp
using var generator = new OnnxImageGenerator(options);
byte[] image = await generator.GenerateAsync("a cat sitting on a beach");
```

## Supported Models (Planned)

| Model | HuggingFace ID | Notes |
|---|---|---|
| Stable Diffusion 2.1 | `stabilityai/stable-diffusion-2-1` | 512×512 default |
| Stable Diffusion XL | `stabilityai/stable-diffusion-xl-base-1.0` | 1024×1024 default |
| Stable Diffusion 1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` | 512×512, 50 steps |

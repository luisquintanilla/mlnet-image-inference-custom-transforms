# Image Embeddings Sample

Demonstrates ML.NET image embeddings using CLIP and DINOv2 ONNX models, with MEAI `IEmbeddingGenerator` integration.

## Setup

1. **Download the model:**
   ```bash
   pip install optimum[onnxruntime]
   optimum-cli export onnx --model openai/clip-vit-base-patch32 models/clip/
   ```

2. **Place test images** (JPEG or PNG) in the sample directory.

3. **Run the sample:**
   ```bash
   dotnet run -- models/clip/model.onnx
   ```

## Features Demonstrated

1. **Direct embedding API** — `GenerateEmbedding()` / `GenerateEmbeddings()`
2. **MEAI integration** — `IEmbeddingGenerator<MLImage, Embedding<float>>`
3. **MLImage ↔ DataContent** — Conversion helpers for MEAI interop
4. **Cosine similarity** — Compare image embeddings

## Pipeline

```
Image file → [LoadImage] → MLImage → [HuggingFace Preprocess] → float[3,224,224]
    → [ONNX CLIP] → hidden states → [CLS Pooling] → [L2 Normalize] → float[512] embedding
```

## Supported Models

| Model | HuggingFace ID | Config | Dimensions |
|---|---|---|---|
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | `PreprocessorConfig.CLIP` | 512 |
| DINOv2 Small | `facebook/dinov2-small` | `PreprocessorConfig.DINOv2` | 384 |

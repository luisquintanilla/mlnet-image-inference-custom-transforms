# Image Segmentation Sample

Demonstrates ML.NET semantic segmentation using a SegFormer ONNX model.

## Setup

1. **Download the model:**
   ```bash
   pip install optimum[onnxruntime]
   optimum-cli export onnx --model nvidia/segformer-b0-finetuned-ade-512-512 models/segformer/
   ```

2. **Place a test image** (JPEG or PNG) in the sample directory.

3. **Run the sample:**
   ```bash
   dotnet run -- models/segformer/model.onnx test-image.jpg
   ```

## Pipeline

```
Image file → [LoadImage] → MLImage → [HuggingFace Preprocess] → float[3,512,512]
    → [ONNX SegFormer] → float[1,150,H,W] logits → [Argmax] → SegmentationMask
```

## Supported Models

| Model | HuggingFace ID | Config | Classes |
|---|---|---|---|
| SegFormer B0 (ADE20K) | `nvidia/segformer-b0-finetuned-ade-512-512` | `PreprocessorConfig.ImageNet` | 150 |
| SegFormer B1 (ADE20K) | `nvidia/segformer-b1-finetuned-ade-512-512` | `PreprocessorConfig.ImageNet` | 150 |
| SegFormer B2 (ADE20K) | `nvidia/segformer-b2-finetuned-ade-512-512` | `PreprocessorConfig.ImageNet` | 150 |
| SegFormer B0 (Cityscapes) | `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` | `PreprocessorConfig.ImageNet` | 19 |

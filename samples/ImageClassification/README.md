# Image Classification Sample

Demonstrates ML.NET image classification using a Vision Transformer (ViT) ONNX model.

## Setup

1. **Download the model:**
   ```bash
   pip install optimum[onnxruntime]
   optimum-cli export onnx --model google/vit-base-patch16-224 models/vit/
   ```

2. **Place a test image** (JPEG or PNG) in the sample directory.

3. **Run the sample:**
   ```bash
   dotnet run -- models/vit/model.onnx test-image.jpg
   ```

## Pipeline

```
Image file → [LoadImage] → MLImage → [HuggingFace Preprocess] → float[3,224,224]
    → [ONNX ViT] → float[1000] logits → [Softmax] → { Label, Probability }
```

## Supported Models

| Model | HuggingFace ID | Config |
|---|---|---|
| ViT Base | `google/vit-base-patch16-224` | `PreprocessorConfig.ImageNet` |
| MobileNetV2 | `google/mobilenet_v2_1.0_224` | `PreprocessorConfig.ImageNet` |
| ResNet-50 | `microsoft/resnet-50` | `PreprocessorConfig.ImageNet` |

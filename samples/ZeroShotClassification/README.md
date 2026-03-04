# Zero-Shot Image Classification Sample

Demonstrates ML.NET zero-shot image classification using a CLIP ONNX model.

## Setup

1. **Download the CLIP models:**
   ```bash
   pip install optimum[onnxruntime]
   optimum-cli export onnx --model openai/clip-vit-base-patch32 models/clip/
   ```

2. **Download tokenizer files:**
   ```bash
   wget https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json -O models/clip/vocab.json
   wget https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt -O models/clip/merges.txt
   ```

3. **Place a test image** (JPEG or PNG) in the sample directory.

4. **Run the sample:**
   ```bash
   dotnet run -- models/clip/vision_model.onnx models/clip/text_model.onnx models/clip/vocab.json models/clip/merges.txt test-image.jpg
   ```

## Pipeline

```
Image file → [LoadImage] → MLImage → [HuggingFace Preprocess] → float[3,224,224]
    → [ONNX CLIP Vision Encoder] → float[512] image embedding

Candidate labels → [CLIP Tokenizer] → token IDs
    → [ONNX CLIP Text Encoder] → float[N,512] text embeddings

Image embedding × Text embeddings → [Cosine Similarity] → [Softmax] → { Label, Probability }
```

## Supported Models

| Model | HuggingFace ID | Config | Embedding Dim |
|---|---|---|---|
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | `PreprocessorConfig.CLIP` | 512 |
| CLIP ViT-B/16 | `openai/clip-vit-base-patch16` | `PreprocessorConfig.CLIP` | 512 |
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | `PreprocessorConfig.CLIP` | 768 |

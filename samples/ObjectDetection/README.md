# Object Detection Sample

Demonstrates ML.NET object detection using a YOLOv8 ONNX model.

## Setup

1. **Download the model:**
   ```bash
   pip install optimum[onnxruntime]
   optimum-cli export onnx --model Ultralytics/YOLOv8s models/yolov8/
   ```

2. **Place a test image** (JPEG or PNG) in the sample directory.

3. **Run the sample:**
   ```bash
   dotnet run -- models/yolov8/model.onnx test-image.jpg
   ```

## Pipeline

```
Image file → [LoadImage] → MLImage → [HuggingFace Preprocess] → float[3,640,640]
    → [ONNX YOLOv8] → float[1,84,8400] → [NMS] → BoundingBox[]
```

## Supported Models

| Model | HuggingFace ID | Config |
|---|---|---|
| YOLOv8s | `Ultralytics/YOLOv8s` | `PreprocessorConfig.ImageNet` |
| YOLOv8n | `Ultralytics/YOLOv8n` | `PreprocessorConfig.ImageNet` |
| YOLOv8m | `Ultralytics/YOLOv8m` | `PreprocessorConfig.ImageNet` |
| YOLOv8l | `Ultralytics/YOLOv8l` | `PreprocessorConfig.ImageNet` |

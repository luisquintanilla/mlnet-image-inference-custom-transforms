using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.Detection;

// =====================================================
// Object Detection Sample using YOLOv8
// =====================================================
//
// Before running:
//   1. Download a YOLOv8 ONNX model from HuggingFace:
//      pip install optimum[onnxruntime]
//      optimum-cli export onnx --model Ultralytics/YOLOv8s models/yolov8/
//   2. Place the model at: models/yolov8/model.onnx
//   3. Place a test image at: test-image.jpg
//
// The pipeline: Image → HuggingFace Preprocess → ONNX Score → NMS → BoundingBox[]

Console.WriteLine("=== ML.NET Object Detection with ONNX ===");
Console.WriteLine();

var modelPath = args.Length > 0 ? args[0] : "models/yolov8/model.onnx";
var imagePath = args.Length > 1 ? args[1] : "test-image.jpg";

if (!File.Exists(modelPath))
{
    Console.WriteLine($"Model not found at: {modelPath}");
    Console.WriteLine();
    Console.WriteLine("Download a YOLOv8 ONNX model:");
    Console.WriteLine("  pip install optimum[onnxruntime]");
    Console.WriteLine("  optimum-cli export onnx --model Ultralytics/YOLOv8s models/yolov8/");
    return;
}

if (!File.Exists(imagePath))
{
    Console.WriteLine($"Image not found at: {imagePath}");
    Console.WriteLine("Place a test image (JPEG/PNG) at the path above.");
    return;
}

// --- Style 1: Convenience API (single call) ---
Console.WriteLine("--- Style 1: Direct Detection API ---");
Console.WriteLine();

var options = new OnnxObjectDetectionOptions
{
    ModelPath = modelPath,
    PreprocessorConfig = PreprocessorConfig.ImageNet,
    ConfidenceThreshold = 0.5f,
    IouThreshold = 0.45f
};

var estimator = new OnnxObjectDetectionEstimator(options);
using var transformer = estimator.Fit(null!);

// Load and detect objects in an image
using var image = MLImage.CreateFromFile(imagePath);
var detections = transformer.Detect(image);

Console.WriteLine($"Image: {imagePath}");
Console.WriteLine($"Detected {detections.Length} object(s):");
Console.WriteLine();

foreach (var box in detections)
{
    Console.WriteLine($"  [{box.Label}] confidence: {box.Score:P2}");
    Console.WriteLine($"    position: ({box.X:F1}, {box.Y:F1})  size: {box.Width:F1} x {box.Height:F1}");
}

Console.WriteLine();

// --- Style 2: ML.NET Pipeline (composable) ---
Console.WriteLine("--- Style 2: ML.NET Pipeline ---");
Console.WriteLine();

var mlContext = new MLContext();

// Build pipeline using MLContext extension method
var pipeline = mlContext.Transforms.OnnxObjectDetection(new OnnxObjectDetectionOptions
{
    ModelPath = modelPath,
    InputColumnName = "Image",
    OutputColumnName = "Detections",
    PreprocessorConfig = PreprocessorConfig.ImageNet,
    ConfidenceThreshold = 0.5f,
    IouThreshold = 0.45f
});

Console.WriteLine("Pipeline created successfully.");
Console.WriteLine("Full IDataView Transform() support coming soon.");
Console.WriteLine("For now, use the direct Detect() API shown above.");

Console.WriteLine();
Console.WriteLine("Done!");

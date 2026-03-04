using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.Segmentation;

// =====================================================
// Image Segmentation Sample using SegFormer
// =====================================================
//
// Before running:
//   1. Download a SegFormer ONNX model from HuggingFace:
//      pip install optimum[onnxruntime]
//      optimum-cli export onnx --model nvidia/segformer-b0-finetuned-ade-512-512 models/segformer/
//   2. Place the model at: models/segformer/model.onnx
//   3. Place a test image at: test-image.jpg
//
// The pipeline: Image → HuggingFace Preprocess → ONNX Score → Argmax → SegmentationMask

Console.WriteLine("=== ML.NET Image Segmentation with ONNX ===");
Console.WriteLine();

var modelPath = args.Length > 0 ? args[0] : "models/segformer/model.onnx";
var imagePath = args.Length > 1 ? args[1] : "test-image.jpg";

if (!File.Exists(modelPath))
{
    Console.WriteLine($"Model not found at: {modelPath}");
    Console.WriteLine();
    Console.WriteLine("Download a SegFormer ONNX model:");
    Console.WriteLine("  pip install optimum[onnxruntime]");
    Console.WriteLine("  optimum-cli export onnx --model nvidia/segformer-b0-finetuned-ade-512-512 models/segformer/");
    return;
}

if (!File.Exists(imagePath))
{
    Console.WriteLine($"Image not found at: {imagePath}");
    Console.WriteLine("Place a test image (JPEG/PNG) at the path above.");
    return;
}

// --- Style 1: Convenience API (single call) ---
Console.WriteLine("--- Style 1: Direct Segmentation API ---");
Console.WriteLine();

var options = new OnnxImageSegmentationOptions
{
    ModelPath = modelPath,
    PreprocessorConfig = PreprocessorConfig.ImageNet
};

var estimator = new OnnxImageSegmentationEstimator(options);
using var transformer = estimator.Fit(null!);

// Load and segment an image
using var image = MLImage.CreateFromFile(imagePath);
var mask = transformer.Segment(image);

Console.WriteLine($"Image: {imagePath}");
Console.WriteLine($"Mask size: {mask.Width} x {mask.Height}");

// Find unique class IDs in the mask
var uniqueClasses = mask.ClassIds.Distinct().OrderBy(id => id).ToArray();
Console.WriteLine($"Unique classes found: {uniqueClasses.Length}");
Console.WriteLine();

foreach (var classId in uniqueClasses)
{
    var label = mask.Labels is not null && classId < mask.Labels.Length
        ? mask.Labels[classId]
        : $"class_{classId}";
    var pixelCount = mask.ClassIds.Count(id => id == classId);
    var percentage = (float)pixelCount / mask.ClassIds.Length * 100;
    Console.WriteLine($"  [{classId}] {label}: {pixelCount} pixels ({percentage:F1}%)");
}

Console.WriteLine();

// Text-based visualization of a small region (top-left 16x8 block)
Console.WriteLine("--- Segmentation Preview (top-left 16x8) ---");
Console.WriteLine();

var previewW = Math.Min(16, mask.Width);
var previewH = Math.Min(8, mask.Height);

for (int y = 0; y < previewH; y++)
{
    for (int x = 0; x < previewW; x++)
    {
        var classId = mask.GetClassAt(x, y);
        var label = mask.Labels is not null && classId < mask.Labels.Length
            ? mask.Labels[classId]
            : $"c{classId}";
        // Truncate label to 6 chars for grid alignment
        var display = label.Length > 6 ? label[..6] : label;
        Console.Write($"{display,7}");
    }
    Console.WriteLine();
}

Console.WriteLine();

// --- Style 2: ML.NET Pipeline (composable) ---
Console.WriteLine("--- Style 2: ML.NET Pipeline ---");
Console.WriteLine();

var mlContext = new MLContext();

// Build pipeline using MLContext extension method
var pipeline = mlContext.Transforms.OnnxImageSegmentation(new OnnxImageSegmentationOptions
{
    ModelPath = modelPath,
    InputColumnName = "Image",
    OutputColumnName = "Mask",
    PreprocessorConfig = PreprocessorConfig.ImageNet
});

Console.WriteLine("Pipeline created successfully.");
Console.WriteLine("Full IDataView Transform() support coming soon.");
Console.WriteLine("For now, use the direct Segment() API shown above.");

Console.WriteLine();
Console.WriteLine("Done!");

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.DepthEstimation;

// =====================================================
// Depth Estimation Sample using DPT-Hybrid (MiDaS)
// =====================================================
//
// Before running:
//   1. Export a DPT ONNX model from HuggingFace:
//      python -c "
//        from transformers import DPTForDepthEstimation; import torch, os
//        model = DPTForDepthEstimation.from_pretrained('Intel/dpt-hybrid-midas')
//        model.eval()
//        torch.onnx.export(model, (torch.randn(1,3,384,384),), 'models/dpt-hybrid/model.onnx',
//          opset_version=14, input_names=['pixel_values'], output_names=['predicted_depth'],
//          dynamic_axes={'pixel_values':{0:'batch'}, 'predicted_depth':{0:'batch'}})
//      "
//   2. Place a test image at: test-image.jpg
//
// The pipeline: Image → HuggingFace Preprocess → ONNX Score → Normalize → DepthMap

Console.WriteLine("=== ML.NET Depth Estimation with ONNX ===");
Console.WriteLine();

var modelPath = args.Length > 0 ? args[0] : "models/dpt-hybrid/model.onnx";
var imagePath = args.Length > 1 ? args[1] : "test-image.jpg";

if (!File.Exists(modelPath))
{
    Console.WriteLine($"Model not found at: {modelPath}");
    Console.WriteLine();
    Console.WriteLine("Export a DPT ONNX model:");
    Console.WriteLine("  python -c \"from transformers import DPTForDepthEstimation; ...\"");
    Console.WriteLine("  See source code header for full export script.");
    return;
}

if (!File.Exists(imagePath))
{
    Console.WriteLine($"Image not found at: {imagePath}");
    Console.WriteLine("Place a test image (JPEG/PNG) at the path above.");
    return;
}

// --- Style 1: Convenience API (single call) ---
Console.WriteLine("--- Style 1: Direct Depth Estimation API ---");
Console.WriteLine();

var options = new OnnxImageDepthEstimationOptions
{
    ModelPath = modelPath,
    PreprocessorConfig = PreprocessorConfig.DPT,
    ResizeToOriginal = true
};

var estimator = new OnnxImageDepthEstimationEstimator(options);
using var transformer = estimator.Fit(null!);

using var image = MLImage.CreateFromFile(imagePath);
var depthMap = transformer.Estimate(image);

Console.WriteLine($"Image: {imagePath} ({image.Width}x{image.Height})");
Console.WriteLine($"Depth map: {depthMap.Width}x{depthMap.Height}");
Console.WriteLine($"Raw depth range: [{depthMap.MinDepth:F2}, {depthMap.MaxDepth:F2}]");
Console.WriteLine();

// Text-based depth visualization (top-left 16x8 block)
Console.WriteLine("--- Depth Preview (top-left 16x8) ---");
Console.WriteLine();

var previewW = Math.Min(16, depthMap.Width);
var previewH = Math.Min(8, depthMap.Height);
string[] blocks = ["░", "▒", "▓", "█"];

for (int y = 0; y < previewH; y++)
{
    for (int x = 0; x < previewW; x++)
    {
        float depth = depthMap.GetDepthAt(x, y);
        int blockIdx = Math.Clamp((int)(depth * blocks.Length), 0, blocks.Length - 1);
        Console.Write(blocks[blockIdx]);
    }
    Console.WriteLine();
}
Console.WriteLine("  (░=far  █=near)");

Console.WriteLine();

// Depth statistics
var values = depthMap.Values;
float avg = values.Average();
float median = values.OrderBy(v => v).ElementAt(values.Length / 2);
Console.WriteLine($"Normalized depth stats:");
Console.WriteLine($"  Mean:   {avg:F4}");
Console.WriteLine($"  Median: {median:F4}");
Console.WriteLine($"  Min:    {values.Min():F4}");
Console.WriteLine($"  Max:    {values.Max():F4}");

Console.WriteLine();

// --- Style 2: ML.NET Pipeline (composable) ---
Console.WriteLine("--- Style 2: ML.NET Pipeline ---");
Console.WriteLine();

var mlContext = new MLContext();
var pipeline = mlContext.Transforms.OnnxImageDepthEstimation(new OnnxImageDepthEstimationOptions
{
    ModelPath = modelPath,
    InputColumnName = "Image",
    OutputColumnName = "DepthMap",
    PreprocessorConfig = PreprocessorConfig.DPT,
    ResizeToOriginal = false
});

Console.WriteLine("Pipeline created successfully.");
Console.WriteLine("Full IDataView Transform() support available.");

Console.WriteLine();
Console.WriteLine("Done!");

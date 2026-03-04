using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.Classification;

// =====================================================
// Image Classification Sample using ViT (Vision Transformer)
// =====================================================
//
// Before running:
//   1. Download a ViT ONNX model from HuggingFace:
//      pip install optimum[onnxruntime]
//      optimum-cli export onnx --model google/vit-base-patch16-224 models/vit/
//   2. Place the model at: models/vit/model.onnx
//   3. Place a test image at: test-image.jpg
//
// The pipeline: Image → HuggingFace Preprocess → ONNX Score → Softmax → Labels

Console.WriteLine("=== ML.NET Image Classification with ONNX ===");
Console.WriteLine();

var modelPath = args.Length > 0 ? args[0] : "models/vit/model.onnx";
var imagePath = args.Length > 1 ? args[1] : "test-image.jpg";

if (!File.Exists(modelPath))
{
    Console.WriteLine($"Model not found at: {modelPath}");
    Console.WriteLine();
    Console.WriteLine("Download a ViT ONNX model:");
    Console.WriteLine("  pip install optimum[onnxruntime]");
    Console.WriteLine("  optimum-cli export onnx --model google/vit-base-patch16-224 models/vit/");
    return;
}

if (!File.Exists(imagePath))
{
    Console.WriteLine($"Image not found at: {imagePath}");
    Console.WriteLine("Place a test image (JPEG/PNG) at the path above.");
    return;
}

// --- Style 1: Convenience API (single call) ---
Console.WriteLine("--- Style 1: Direct Classification API ---");
Console.WriteLine();

var options = new OnnxImageClassificationOptions
{
    ModelPath = modelPath,
    PreprocessorConfig = PreprocessorConfig.ImageNet,
    TopK = 5
};

var estimator = new OnnxImageClassificationEstimator(options);
using var transformer = estimator.Fit(null!);

// Load and classify an image
using var image = MLImage.CreateFromFile(imagePath);
var predictions = transformer.Classify(image);

Console.WriteLine($"Image: {imagePath}");
Console.WriteLine($"Top {predictions.Length} predictions:");
foreach (var (label, probability) in predictions)
{
    Console.WriteLine($"  {label}: {probability:P2}");
}

Console.WriteLine();

// --- Style 2: ML.NET Pipeline (composable) ---
Console.WriteLine("--- Style 2: ML.NET Pipeline ---");
Console.WriteLine();

var mlContext = new MLContext();

// Build pipeline using MLContext extension method
var pipeline = mlContext.Transforms.OnnxImageClassification(new OnnxImageClassificationOptions
{
    ModelPath = modelPath,
    InputColumnName = "Image",
    PredictedLabelColumnName = "PredictedLabel",
    ProbabilityColumnName = "Score",
    PreprocessorConfig = PreprocessorConfig.ImageNet,
    TopK = 5
});

Console.WriteLine("Pipeline created successfully.");
Console.WriteLine("Full IDataView Transform() support coming soon.");
Console.WriteLine("For now, use the direct Classify() API shown above.");

Console.WriteLine();
Console.WriteLine("Done!");

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.ImageCaptioning;

// =====================================================
// Image Captioning Sample using GIT (microsoft/git-base-coco)
// =====================================================
//
// Before running:
//   1. Export GIT ONNX models from HuggingFace (see export_git_model.py in this folder or repo README)
//   2. Place encoder.onnx and decoder.onnx in models/git-coco/
//   3. Place vocab.txt in models/git-coco/
//   4. Place a test image at: test-image.jpg
//
// The pipeline: Image → HuggingFace Preprocess → Vision Encoder → Autoregressive Decoder → Caption

Console.WriteLine("=== ML.NET Image Captioning with GIT ===");
Console.WriteLine();

var encoderPath = args.Length > 0 ? args[0] : "models/git-coco/encoder.onnx";
var decoderPath = args.Length > 1 ? args[1] : "models/git-coco/decoder.onnx";
var vocabPath = args.Length > 2 ? args[2] : "models/git-coco/vocab.txt";
var imagePath = args.Length > 3 ? args[3] : "test-image.jpg";

if (!File.Exists(encoderPath) || !File.Exists(decoderPath))
{
    Console.WriteLine($"Model not found at: {encoderPath} or {decoderPath}");
    Console.WriteLine();
    Console.WriteLine("Export GIT ONNX models using:");
    Console.WriteLine("  python export_git_model.py");
    Console.WriteLine("  See README for full instructions.");
    return;
}

if (!File.Exists(vocabPath))
{
    Console.WriteLine($"Vocab file not found at: {vocabPath}");
    Console.WriteLine("Download vocab.txt from microsoft/git-base-coco on HuggingFace.");
    return;
}

if (!File.Exists(imagePath))
{
    Console.WriteLine($"Image not found at: {imagePath}");
    Console.WriteLine("Place a test image (JPEG/PNG) at the path above.");
    return;
}

// --- Style 1: Convenience API (single call) ---
Console.WriteLine("--- Style 1: Direct Captioning API ---");
Console.WriteLine();

var options = new OnnxImageCaptioningOptions
{
    EncoderModelPath = encoderPath,
    DecoderModelPath = decoderPath,
    VocabPath = vocabPath,
    PreprocessorConfig = PreprocessorConfig.GIT,
    MaxLength = 50
};

var estimator = new OnnxImageCaptioningEstimator(options);
using var transformer = estimator.Fit(null!);

using var image = MLImage.CreateFromFile(imagePath);
Console.WriteLine($"Image: {imagePath} ({image.Width}x{image.Height})");

var caption = transformer.GenerateCaption(image);
Console.WriteLine($"Caption: {caption}");
Console.WriteLine();

// --- Style 2: ML.NET Pipeline (composable) ---
Console.WriteLine("--- Style 2: ML.NET Pipeline ---");
Console.WriteLine();

var mlContext = new MLContext();
var pipeline = mlContext.Transforms.OnnxImageCaptioning(new OnnxImageCaptioningOptions
{
    EncoderModelPath = encoderPath,
    DecoderModelPath = decoderPath,
    VocabPath = vocabPath,
    InputColumnName = "Image",
    OutputColumnName = "Caption",
    PreprocessorConfig = PreprocessorConfig.GIT,
    MaxLength = 50
});

Console.WriteLine("Pipeline created successfully.");
Console.WriteLine("Full IDataView Transform() support available.");

Console.WriteLine();
Console.WriteLine("Done!");

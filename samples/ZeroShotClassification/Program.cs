using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.ZeroShot;

// =====================================================
// Zero-Shot Image Classification Sample using CLIP
// =====================================================
//
// Before running:
//   1. Download the CLIP ONNX models from HuggingFace:
//      pip install optimum[onnxruntime]
//      optimum-cli export onnx --model openai/clip-vit-base-patch32 models/clip/
//      This produces: vision_model.onnx, text_model.onnx
//   2. Download tokenizer files:
//      wget https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json -O models/clip/vocab.json
//      wget https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt -O models/clip/merges.txt
//   3. Place a test image at: test-image.jpg
//
// The pipeline: Image → CLIP Vision Encoder → image embedding
//               Text  → CLIP Text Encoder  → text embeddings
//               → Cosine Similarity → Softmax → { Label, Probability }

Console.WriteLine("=== ML.NET Zero-Shot Image Classification with ONNX ===");
Console.WriteLine();

var visionModelPath = args.Length > 0 ? args[0] : "models/clip/vision_model.onnx";
var textModelPath = args.Length > 1 ? args[1] : "models/clip/text_model.onnx";
var vocabPath = args.Length > 2 ? args[2] : "models/clip/vocab.json";
var mergesPath = args.Length > 3 ? args[3] : "models/clip/merges.txt";
var imagePath = args.Length > 4 ? args[4] : "test-image.jpg";

// Check all required files
var requiredFiles = new (string Path, string Description)[]
{
    (visionModelPath, "CLIP vision model"),
    (textModelPath, "CLIP text model"),
    (vocabPath, "Tokenizer vocab"),
    (mergesPath, "Tokenizer merges"),
};

var missing = requiredFiles.Where(f => !File.Exists(f.Path)).ToArray();
if (missing.Length > 0)
{
    foreach (var (path, desc) in missing)
        Console.WriteLine($"Missing {desc}: {path}");

    Console.WriteLine();
    Console.WriteLine("Download CLIP ONNX models and tokenizer files:");
    Console.WriteLine("  pip install optimum[onnxruntime]");
    Console.WriteLine("  optimum-cli export onnx --model openai/clip-vit-base-patch32 models/clip/");
    Console.WriteLine();
    Console.WriteLine("Download tokenizer files:");
    Console.WriteLine("  wget https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json -O models/clip/vocab.json");
    Console.WriteLine("  wget https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt -O models/clip/merges.txt");
    return;
}

if (!File.Exists(imagePath))
{
    Console.WriteLine($"Image not found at: {imagePath}");
    Console.WriteLine("Place a test image (JPEG/PNG) at the path above.");
    return;
}

// Candidate labels for zero-shot classification
string[] candidateLabels =
[
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
    "a photo of a car"
];

// --- Style 1: Convenience API (single call) ---
Console.WriteLine("--- Style 1: Direct Classification API ---");
Console.WriteLine();

var options = new OnnxZeroShotImageClassificationOptions
{
    ImageModelPath = visionModelPath,
    TextModelPath = textModelPath,
    VocabPath = vocabPath,
    MergesPath = mergesPath,
    CandidateLabels = candidateLabels,
    PreprocessorConfig = PreprocessorConfig.CLIP
};

var estimator = new OnnxZeroShotImageClassificationEstimator(options);
using var transformer = estimator.Fit(null!);

// Load and classify an image
using var image = MLImage.CreateFromFile(imagePath);
var predictions = transformer.Classify(image);

Console.WriteLine($"Image: {imagePath}");
Console.WriteLine($"Candidate labels: {string.Join(", ", candidateLabels)}");
Console.WriteLine();
Console.WriteLine("Probabilities:");

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
var pipeline = mlContext.Transforms.OnnxZeroShotImageClassification(new OnnxZeroShotImageClassificationOptions
{
    ImageModelPath = visionModelPath,
    TextModelPath = textModelPath,
    VocabPath = vocabPath,
    MergesPath = mergesPath,
    InputColumnName = "Image",
    PredictedLabelColumnName = "PredictedLabel",
    ProbabilityColumnName = "Score",
    CandidateLabels = candidateLabels,
    PreprocessorConfig = PreprocessorConfig.CLIP
});

Console.WriteLine("Pipeline created successfully.");
Console.WriteLine("Full IDataView Transform() support coming soon.");
Console.WriteLine("For now, use the direct Classify() API shown above.");

Console.WriteLine();
Console.WriteLine("Done!");

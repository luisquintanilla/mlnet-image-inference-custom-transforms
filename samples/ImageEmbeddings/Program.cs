using Microsoft.Extensions.AI;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.Embeddings;
using MLNet.ImageInference.Onnx.MEAI;

// =====================================================
// Image Embeddings Sample using CLIP / DINOv2
// =====================================================
//
// Before running:
//   1. Download a CLIP ONNX model from HuggingFace:
//      pip install optimum[onnxruntime]
//      optimum-cli export onnx --model openai/clip-vit-base-patch32 models/clip/
//   2. Place the vision model at: models/clip/model.onnx
//   3. Place test images in the current directory
//
// The pipeline: Image → HuggingFace Preprocess → ONNX Score → Pooling → L2 Normalize → float[] embedding

Console.WriteLine("=== ML.NET Image Embeddings with ONNX ===");
Console.WriteLine();

var modelPath = args.Length > 0 ? args[0] : "models/clip/model.onnx";

if (!File.Exists(modelPath))
{
    Console.WriteLine($"Model not found at: {modelPath}");
    Console.WriteLine();
    Console.WriteLine("Download a CLIP ONNX model:");
    Console.WriteLine("  pip install optimum[onnxruntime]");
    Console.WriteLine("  optimum-cli export onnx --model openai/clip-vit-base-patch32 models/clip/");
    return;
}

// --- Style 1: Direct Embedding API ---
Console.WriteLine("--- Style 1: Direct Embedding API ---");
Console.WriteLine();

var embeddingOptions = new OnnxImageEmbeddingOptions
{
    ModelPath = modelPath,
    PreprocessorConfig = PreprocessorConfig.CLIP,
    Pooling = PoolingStrategy.ClsToken,
    Normalize = true
};

using var transformer = new OnnxImageEmbeddingTransformer(embeddingOptions);
Console.WriteLine($"Embedding dimension: {transformer.EmbeddingDimension}");
Console.WriteLine();

// Find test images
var imageFiles = Directory.GetFiles(".", "*.jpg")
    .Concat(Directory.GetFiles(".", "*.png"))
    .Take(3)
    .ToArray();

if (imageFiles.Length == 0)
{
    Console.WriteLine("No test images found. Place .jpg or .png files in the current directory.");
    Console.WriteLine("Skipping embedding generation demo.");
}
else
{
    var embeddings = new List<(string File, float[] Embedding)>();

    foreach (var file in imageFiles)
    {
        using var image = MLImage.CreateFromFile(file);
        var embedding = transformer.GenerateEmbedding(image);
        embeddings.Add((Path.GetFileName(file), embedding));
        Console.WriteLine($"  {Path.GetFileName(file)}: [{string.Join(", ", embedding.Take(5).Select(v => v.ToString("F4")))}...]");
    }

    // Compute cosine similarity between all pairs
    if (embeddings.Count >= 2)
    {
        Console.WriteLine();
        Console.WriteLine("Cosine similarities:");
        for (int i = 0; i < embeddings.Count; i++)
        {
            for (int j = i + 1; j < embeddings.Count; j++)
            {
                float similarity = CosineSimilarity(embeddings[i].Embedding, embeddings[j].Embedding);
                Console.WriteLine($"  {embeddings[i].File} ↔ {embeddings[j].File}: {similarity:F4}");
            }
        }
    }
}

Console.WriteLine();

// --- Style 2: MEAI IEmbeddingGenerator<MLImage, Embedding<float>> ---
Console.WriteLine("--- Style 2: MEAI IEmbeddingGenerator ---");
Console.WriteLine();

var generator = new OnnxImageEmbeddingGenerator(modelPath);
Console.WriteLine($"Provider: {generator.Metadata.ProviderName}");
Console.WriteLine($"Model: {generator.Metadata.DefaultModelId}");
Console.WriteLine($"Dimensions: {generator.Metadata.DefaultModelDimensions}");

// The generator implements IEmbeddingGenerator<MLImage, Embedding<float>>
IEmbeddingGenerator<MLImage, Embedding<float>> embeddingGenerator = generator;

if (imageFiles.Length > 0)
{
    var images = imageFiles.Select(f => MLImage.CreateFromFile(f)).ToList();
    var meaiEmbeddings = await embeddingGenerator.GenerateAsync(images);

    Console.WriteLine($"Generated {meaiEmbeddings.Count} embeddings via MEAI interface.");

    foreach (var img in images) img.Dispose();
}

Console.WriteLine();

// --- MLImage ↔ DataContent conversion helpers ---
Console.WriteLine("--- MLImage ↔ DataContent Conversion ---");
Console.WriteLine();

if (imageFiles.Length > 0)
{
    using var sampleImage = MLImage.CreateFromFile(imageFiles[0]);
    Console.WriteLine($"Original MLImage: {sampleImage.Width}x{sampleImage.Height}");

    // Convert to MEAI DataContent
    var dataContent = sampleImage.ToDataContent();
    Console.WriteLine($"DataContent: {dataContent.MediaType}, {dataContent.Data.Length} bytes");

    // Convert back to MLImage
    using var roundTripped = dataContent.ToMLImage();
    Console.WriteLine($"Round-tripped MLImage: {roundTripped.Width}x{roundTripped.Height}");
}

Console.WriteLine();
Console.WriteLine("Done!");

static float CosineSimilarity(float[] a, float[] b)
{
    float dot = 0, normA = 0, normB = 0;
    for (int i = 0; i < a.Length; i++)
    {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (MathF.Sqrt(normA) * MathF.Sqrt(normB));
}

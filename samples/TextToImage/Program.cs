using Microsoft.ML.Data;
using MLNet.ImageGeneration.OnnxGenAI;
using MLNet.ImageGeneration.OnnxGenAI.MEAI;

// =====================================================
// Text-to-Image Generation Sample
// =====================================================
//
// Before running:
//   1. Export a Stable Diffusion ONNX model (see instructions at bottom).
//   2. Provide CLIP tokenizer files (vocab.json + merges.txt) for real prompt encoding.
//   3. Expected model directory structure:
//      models/sd/
//        text_encoder/model.onnx
//        unet/model.onnx
//        vae_decoder/model.onnx
//      models/clip/
//        vocab.json
//        merges.txt

Console.WriteLine("=== ML.NET Text-to-Image Generation ===");
Console.WriteLine();

var modelDirectory = args.Length > 0 ? args[0] : "models/sd";
var vocabPath = args.Length > 1 ? args[1] : "models/clip/vocab.json";
var mergesPath = args.Length > 2 ? args[2] : "models/clip/merges.txt";

// --- Style 1: Direct OnnxImageGenerationTransformer API ---
Console.WriteLine("--- Style 1: Direct Generation API ---");
Console.WriteLine();

var options = new OnnxImageGenerationOptions
{
    ModelDirectory = modelDirectory,
    VocabPath = File.Exists(vocabPath) ? vocabPath : null,
    MergesPath = File.Exists(mergesPath) ? mergesPath : null,
    NumInferenceSteps = 20,
    GuidanceScale = 7.5f,
    Width = 512,
    Height = 512,
    Seed = 42,
    NegativePrompt = "blurry, low quality"
};

Console.WriteLine($"Model directory: {options.ModelDirectory}");
Console.WriteLine($"Tokenizer: {(options.VocabPath is not null ? "CLIP BPE" : "Simple (SOT+EOT only)")}");
Console.WriteLine($"Steps: {options.NumInferenceSteps}, Guidance: {options.GuidanceScale}");
Console.WriteLine($"Dimensions: {options.Width}x{options.Height}, Seed: {options.Seed}");
Console.WriteLine();

try
{
    using var transformer = new OnnxImageGenerationTransformer(options);
    using var image = transformer.Generate("a cat sitting on a beach at sunset");
    Console.WriteLine($"Generated image: {image.Width}x{image.Height}");
}
catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException
    or Microsoft.ML.OnnxRuntime.OnnxRuntimeException)
{
    Console.WriteLine($"⚠️  Model not found: {ex.Message}");
    Console.WriteLine("    Download a Stable Diffusion ONNX model first (see instructions below).");
}

Console.WriteLine();

// --- Style 2: MEAI OnnxImageGenerator API ---
Console.WriteLine("--- Style 2: MEAI OnnxImageGenerator ---");
Console.WriteLine();

try
{
    using var generator = new OnnxImageGenerator(options);
    using var image = await generator.GenerateAsync("a serene mountain landscape at dawn");
    Console.WriteLine($"Generated image: {image.Width}x{image.Height}");
}
catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException
    or Microsoft.ML.OnnxRuntime.OnnxRuntimeException)
{
    Console.WriteLine($"⚠️  Model not found: {ex.Message}");
    Console.WriteLine("    Download a Stable Diffusion ONNX model first (see instructions below).");
}

Console.WriteLine();

// --- Model Download Instructions ---
Console.WriteLine("=== Model Download Instructions ===");
Console.WriteLine();
Console.WriteLine("Export with diffusers + torch.onnx.export:");
Console.WriteLine("  pip install torch diffusers transformers onnx onnxruntime");
Console.WriteLine("  python export_sd.py  # See README for export script");
Console.WriteLine();
Console.WriteLine("CLIP tokenizer files (vocab.json + merges.txt) can be downloaded from:");
Console.WriteLine("  huggingface-cli download openai/clip-vit-base-patch32 vocab.json merges.txt --local-dir models/clip/");
Console.WriteLine();
Console.WriteLine("Done!");

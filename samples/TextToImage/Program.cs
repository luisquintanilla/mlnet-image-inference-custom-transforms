using Microsoft.ML.Data;
using MLNet.ImageGeneration.OnnxGenAI;
using MLNet.ImageGeneration.OnnxGenAI.MEAI;

// =====================================================
// Text-to-Image Generation Sample
// =====================================================
//
// Before running:
//   1. Download a Stable Diffusion ONNX model:
//      pip install optimum[onnxruntime]
//      optimum-cli export onnx --model stabilityai/stable-diffusion-2-1 models/sd/
//   2. Expected model directory structure:
//      models/sd/
//        text_encoder/model.onnx
//        unet/model.onnx
//        vae_decoder/model.onnx

Console.WriteLine("=== ML.NET Text-to-Image Generation ===");
Console.WriteLine();

var modelDirectory = args.Length > 0 ? args[0] : "models/sd";

// --- Style 1: Direct OnnxImageGenerationTransformer API ---
Console.WriteLine("--- Style 1: Direct Generation API ---");
Console.WriteLine();

var options = new OnnxImageGenerationOptions
{
    ModelDirectory = modelDirectory,
    NumInferenceSteps = 20,
    GuidanceScale = 7.5f,
    Width = 512,
    Height = 512,
    Seed = 42,
    NegativePrompt = "blurry, low quality"
};

Console.WriteLine($"Model directory: {options.ModelDirectory}");
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
Console.WriteLine("Option 1: Export with optimum-cli");
Console.WriteLine("  pip install optimum[onnxruntime]");
Console.WriteLine("  optimum-cli export onnx --model stabilityai/stable-diffusion-2-1 models/sd/");
Console.WriteLine();
Console.WriteLine("Option 2: Download pre-exported ONNX model");
Console.WriteLine("  pip install huggingface-hub");
Console.WriteLine("  huggingface-cli download stabilityai/stable-diffusion-2-1-onnx --local-dir models/sd/");
Console.WriteLine();
Console.WriteLine("Done!");

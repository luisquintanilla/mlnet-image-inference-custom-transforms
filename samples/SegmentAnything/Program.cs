using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.SegmentAnything;

// =====================================================
// Segment Anything Model (SAM2) Sample
// =====================================================
//
// Before running:
//   1. Download SAM2 ONNX models:
//      pip install huggingface-hub
//      huggingface-cli download shubham0204/sam2-onnx-models \
//          sam2_hiera_tiny_encoder.onnx sam2_hiera_tiny_decoder.onnx \
//          --local-dir models/sam2-tiny/
//
//   2. Expected model directory structure:
//      models/sam2-tiny/
//        sam2_hiera_tiny_encoder.onnx
//        sam2_hiera_tiny_decoder.onnx

Console.WriteLine("=== ML.NET SAM2 Segment Anything ===");
Console.WriteLine();

var encoderPath = args.Length > 0 ? args[0] : "models/sam2-tiny/sam2_hiera_tiny_encoder.onnx";
var decoderPath = args.Length > 1 ? args[1] : "models/sam2-tiny/sam2_hiera_tiny_decoder.onnx";

if (!File.Exists(encoderPath) || !File.Exists(decoderPath))
{
    Console.WriteLine("⚠️  SAM2 models not found.");
    Console.WriteLine("    Download with: huggingface-cli download shubham0204/sam2-onnx-models sam2_hiera_tiny_encoder.onnx sam2_hiera_tiny_decoder.onnx --local-dir models/sam2-tiny/");
    return;
}

var options = new OnnxSegmentAnythingOptions
{
    EncoderModelPath = encoderPath,
    DecoderModelPath = decoderPath,
    PreprocessorConfig = PreprocessorConfig.SAM2
};

// Create a test image (red rectangle on blue background)
using var image = CreateTestImage(512, 512);
Console.WriteLine($"Test image: {image.Width}x{image.Height}");
Console.WriteLine();

// --- Style 1: Direct API with point prompt ---
Console.WriteLine("--- Style 1: Point Prompt ---");
using var transformer = new OnnxSegmentAnythingTransformer(options);

var pointPrompt = SegmentAnythingPrompt.FromPoint(256f, 256f); // center
var pointResult = transformer.Segment(image, pointPrompt);

Console.WriteLine($"  Masks: {pointResult.NumMasks}");
Console.WriteLine($"  Best mask size: {pointResult.Width}x{pointResult.Height}");
Console.WriteLine($"  Best IoU: {pointResult.GetBestIoU():F4}");
Console.WriteLine($"  Mask pixels: {pointResult.GetBestMask().Count(v => v > 0)} foreground");
Console.WriteLine();

// --- Style 2: Bounding box prompt ---
Console.WriteLine("--- Style 2: Bounding Box Prompt ---");

var boxPrompt = SegmentAnythingPrompt.FromBoundingBox(128f, 128f, 384f, 384f);
var boxResult = transformer.Segment(image, boxPrompt);

Console.WriteLine($"  Masks: {boxResult.NumMasks}");
Console.WriteLine($"  Best IoU: {boxResult.GetBestIoU():F4}");
Console.WriteLine($"  Mask pixels: {boxResult.GetBestMask().Count(v => v > 0)} foreground");
Console.WriteLine();

// --- Style 3: Encode once, segment multiple times ---
Console.WriteLine("--- Style 3: Cached Embeddings (Multiple Prompts) ---");

var embedding = transformer.EncodeImage(image);
Console.WriteLine("  Image encoded (cached)");

for (int i = 0; i < 3; i++)
{
    float x = 128f + i * 128f;
    var prompt = SegmentAnythingPrompt.FromPoint(x, 256f);
    var result = transformer.Segment(embedding, prompt);
    Console.WriteLine($"  Point ({x:F0}, 256) → IoU={result.GetBestIoU():F4}, pixels={result.GetBestMask().Count(v => v > 0)}");
}
Console.WriteLine();

// --- Style 4: ML.NET Pipeline ---
Console.WriteLine("--- Style 4: ML.NET Pipeline ---");

var mlContext = new MLContext();
var estimator = mlContext.Transforms.OnnxSegmentAnything(options);
using var pipeTransformer = estimator.Fit(null!);

Console.WriteLine("  Pipeline created and fitted");
Console.WriteLine();

Console.WriteLine("Done!");

// Create a simple test image with a colored rectangle
static MLImage CreateTestImage(int width, int height)
{
    var pixels = new byte[width * height * 4];
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = (y * width + x) * 4;
            // Blue background with red rectangle in the center
            if (x >= width / 4 && x < 3 * width / 4 && y >= height / 4 && y < 3 * height / 4)
            {
                pixels[idx + 0] = 200; // R
                pixels[idx + 1] = 50;  // G
                pixels[idx + 2] = 50;  // B
            }
            else
            {
                pixels[idx + 0] = 50;  // R
                pixels[idx + 1] = 50;  // G
                pixels[idx + 2] = 200; // B
            }
            pixels[idx + 3] = 255; // A
        }
    }
    return MLImage.CreateFromPixels(width, height, MLPixelFormat.Rgba32, pixels);
}

using Microsoft.Extensions.AI;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.ImageCaptioning;
using MLNet.ImageInference.Onnx.MEAI;

// =====================================================
// Visual Question Answering (VQA) Sample
// =====================================================
//
// This sample demonstrates answering questions about images using
// GIT-VQA (Generative Image-to-text Transformer for Visual Question Answering).
//
// Before running:
//   1. Export the GIT-VQA model (see README for instructions):
//      python export_vqa.py
//   2. Expected model directory structure:
//      models/git-base-textvqa/
//        encoder.onnx
//        decoder.onnx
//        vocab.txt

Console.WriteLine("=== ML.NET Visual Question Answering ===");
Console.WriteLine();

var encoderPath = args.Length > 0 ? args[0] : "models/git-base-textvqa/encoder.onnx";
var decoderPath = args.Length > 1 ? args[1] : "models/git-base-textvqa/decoder.onnx";
var vocabPath = args.Length > 2 ? args[2] : "models/git-base-textvqa/vocab.txt";
var imagePath = args.Length > 3 ? args[3] : null;

var options = new OnnxImageCaptioningOptions
{
    EncoderModelPath = encoderPath,
    DecoderModelPath = decoderPath,
    VocabPath = vocabPath,
    PreprocessorConfig = PreprocessorConfig.GITVQA,
    MaxLength = 30
};

// --- Style 1: Direct Transformer API ---
Console.WriteLine("--- Style 1: Direct Transformer API ---");
Console.WriteLine();

try
{
    using var transformer = new OnnxImageCaptioningTransformer(options);

    using var image = imagePath is not null
        ? MLImage.CreateFromFile(imagePath)
        : CreateSampleImage();

    Console.WriteLine($"Image: {image.Width}x{image.Height}");

    // Ask multiple questions about the same image
    string[] questions =
    [
        "what is in this image?",
        "what color is this?",
        "how many objects are there?",
        "what is the text in the image?"
    ];

    foreach (var question in questions)
    {
        var answer = transformer.AnswerQuestion(image, question);
        Console.WriteLine($"  Q: {question}");
        Console.WriteLine($"  A: {(string.IsNullOrEmpty(answer) ? "(no answer)" : answer)}");
        Console.WriteLine();
    }

    // Also demonstrate captioning mode (same model, no question)
    var caption = transformer.GenerateCaption(image);
    Console.WriteLine($"  Caption: {(string.IsNullOrEmpty(caption) ? "(no caption)" : caption)}");
}
catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException
    or Microsoft.ML.OnnxRuntime.OnnxRuntimeException)
{
    Console.WriteLine($"⚠️  Model not found: {ex.Message}");
    Console.WriteLine("    Export the GIT-VQA model first (see README for instructions).");
}

Console.WriteLine();

// --- Style 2: MEAI IChatClient API ---
Console.WriteLine("--- Style 2: MEAI IChatClient API ---");
Console.WriteLine();

try
{
    using var chatClient = new OnnxImageCaptioningChatClient(options, modelId: "git-base-textvqa");

    using var image = imagePath is not null
        ? MLImage.CreateFromFile(imagePath)
        : CreateSampleImage();

    var imageData = image.ToDataContent();

    // VQA mode: send image + question together
    var vqaResponse = await chatClient.GetResponseAsync([
        new ChatMessage(ChatRole.User, [
            imageData,
            new TextContent("what is in this image?")
        ])
    ]);
    Console.WriteLine($"  VQA response: {vqaResponse.Text}");

    // Captioning mode: send image only (no text)
    var captionResponse = await chatClient.GetResponseAsync([
        new ChatMessage(ChatRole.User, [imageData])
    ]);
    Console.WriteLine($"  Caption response: {captionResponse.Text}");
}
catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException
    or Microsoft.ML.OnnxRuntime.OnnxRuntimeException)
{
    Console.WriteLine($"⚠️  Model not found: {ex.Message}");
}

Console.WriteLine();
Console.WriteLine("Done!");

// Create a simple test image when no real image is provided
static MLImage CreateSampleImage()
{
    // Create a gradient image (more interesting than solid color for VQA)
    int w = 480, h = 480;
    var pixels = new byte[w * h * 4];
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            int idx = (y * w + x) * 4;
            pixels[idx + 0] = (byte)(x * 255 / w);      // R: horizontal gradient
            pixels[idx + 1] = (byte)(y * 255 / h);      // G: vertical gradient
            pixels[idx + 2] = (byte)(128);                // B: constant
            pixels[idx + 3] = 255;
        }
    }
    return MLImage.CreateFromPixels(w, h, MLPixelFormat.Rgba32, pixels);
}

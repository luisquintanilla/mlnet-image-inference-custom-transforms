using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.ZeroShot;
using Xunit;

namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Integration tests for CLIP zero-shot image classification using separate vision and text encoders.
/// Requires models to be present:
///   models/clip/vision_model.onnx
///   models/clip/text_model.onnx
///   models/clip/vocab.json
///   models/clip/merges.txt
/// </summary>
public class ZeroShotIntegrationTests : IDisposable
{
    private const string VisionModelPath = "models/clip/vision_model.onnx";
    private const string TextModelPath = "models/clip/text_model.onnx";
    private const string VocabPath = "models/clip/vocab.json";
    private const string MergesPath = "models/clip/merges.txt";

    private static bool ModelsExist =>
        File.Exists(VisionModelPath) &&
        File.Exists(TextModelPath) &&
        File.Exists(VocabPath) &&
        File.Exists(MergesPath);

    private static readonly string[] CandidateLabels =
    [
        "a photo of a red object",
        "a photo of a green object",
        "a photo of a blue object",
        "a photo of nothing"
    ];

    private readonly List<IDisposable> _disposables = [];

    public void Dispose()
    {
        foreach (var d in _disposables)
            d.Dispose();
    }

    private OnnxZeroShotImageClassificationTransformer CreateTransformer(
        string[]? labels = null, int? topK = null)
    {
        var options = new OnnxZeroShotImageClassificationOptions
        {
            ImageModelPath = VisionModelPath,
            TextModelPath = TextModelPath,
            VocabPath = VocabPath,
            MergesPath = MergesPath,
            CandidateLabels = labels ?? CandidateLabels,
            PreprocessorConfig = PreprocessorConfig.CLIP,
            TopK = topK
        };
        var transformer = new OnnxZeroShotImageClassificationTransformer(options);
        _disposables.Add(transformer);
        return transformer;
    }

    [Fact]
    public void BothModels_LoadCorrectly()
    {
        if (!ModelsExist) return;

        var transformer = CreateTransformer();

        // If we get here without exception, both models loaded and text labels were pre-encoded
        Assert.NotNull(transformer);
    }

    [Fact]
    public void Classify_ReturnsProbabilitiesThatSumToOne()
    {
        if (!ModelsExist) return;

        var transformer = CreateTransformer();
        using var image = CreateTestImage(224, 224);

        var results = transformer.Classify(image);

        Assert.NotNull(results);
        Assert.Equal(CandidateLabels.Length, results.Length);

        // Each probability should be in [0, 1]
        foreach (var (label, prob) in results)
        {
            Assert.InRange(prob, 0f, 1f);
            Assert.NotNull(label);
            Assert.NotEmpty(label);
        }

        // Probabilities should sum to ~1.0
        float sum = results.Sum(r => r.Probability);
        Assert.InRange(sum, 0.99f, 1.01f);
    }

    [Fact]
    public void Classify_ResultsAreSortedDescending()
    {
        if (!ModelsExist) return;

        var transformer = CreateTransformer();
        using var image = CreateTestImage(224, 224);

        var results = transformer.Classify(image);

        Assert.True(results.Length > 1);

        for (int i = 1; i < results.Length; i++)
        {
            Assert.True(results[i].Probability <= results[i - 1].Probability,
                $"Results not sorted: index {i - 1} ({results[i - 1].Probability}) < index {i} ({results[i].Probability})");
        }
    }

    [Fact]
    public void Classify_TopK_ReturnsRequestedCount()
    {
        if (!ModelsExist) return;

        var transformer = CreateTransformer(topK: 2);
        using var image = CreateTestImage(224, 224);

        var results = transformer.Classify(image);

        Assert.Equal(2, results.Length);

        // TopK results should still be sorted descending
        Assert.True(results[0].Probability >= results[1].Probability);
    }

    private static MLImage CreateTestImage(int width, int height)
    {
        var pixels = new byte[width * height * 4];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = (y * width + x) * 4;
                pixels[idx + 0] = (byte)(x * 255 / width);  // R gradient
                pixels[idx + 1] = (byte)(y * 255 / height); // G gradient
                pixels[idx + 2] = 128;                       // B constant
                pixels[idx + 3] = 255;                       // A
            }
        }
        return MLImage.CreateFromPixels(width, height, MLPixelFormat.Rgba32, pixels);
    }
}

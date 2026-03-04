using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Classification;
using MLNet.ImageInference.Onnx.Embeddings;
using MLNet.ImageInference.Onnx.Shared;
using Xunit;

namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Integration tests for batched inference across classification and embedding transformers.
/// Verifies batch results match single-item results for both fixed-batch and dynamic-batch models.
/// Requires models to be downloaded first: pwsh scripts/download-test-models.ps1
/// </summary>
public class BatchIntegrationTests : IDisposable
{
    private const string SqueezeNetModelPath = "models/squeezenet/model.onnx";
    private const string SqueezeNetLabelsPath = "models/squeezenet/imagenet_classes.txt";
    private const string MobileNetModelPath = "models/mobilenet/model.onnx";
    private const string ClipVisionModelPath = "models/clip/vision_model.onnx";

    private readonly List<IDisposable> _disposables = [];

    public void Dispose()
    {
        foreach (var d in _disposables)
            d.Dispose();
    }

    // --- Classification: Fixed-batch model (SqueezeNet) ---

    [Fact]
    public void ClassifyBatch_FixedModel_MatchesSingleResults()
    {
        if (!File.Exists(SqueezeNetModelPath)) return;

        var labels = File.ReadAllLines(SqueezeNetLabelsPath);
        var options = new OnnxImageClassificationOptions
        {
            ModelPath = SqueezeNetModelPath,
            Labels = labels,
            PreprocessorConfig = PreprocessorConfig.ImageNet,
            TopK = 5
        };

        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        using var img1 = CreateTestImage(224, 224, 255, 0, 0);
        using var img2 = CreateTestImage(224, 224, 0, 255, 0);
        using var img3 = CreateTestImage(224, 224, 0, 0, 255);

        // Single inference
        var single1 = transformer.Classify(img1);
        var single2 = transformer.Classify(img2);
        var single3 = transformer.Classify(img3);

        // Batch inference
        var batchResults = transformer.ClassifyBatch([img1, img2, img3]);

        Assert.Equal(3, batchResults.Length);

        // Fixed-batch model loops internally — results should be exactly identical
        AssertClassificationResultsExact(single1, batchResults[0]);
        AssertClassificationResultsExact(single2, batchResults[1]);
        AssertClassificationResultsExact(single3, batchResults[2]);
    }

    // --- Classification: Dynamic-batch model (MobileNet) ---

    [Fact]
    public void ClassifyBatch_DynamicModel_MatchesSingleResults()
    {
        if (!File.Exists(MobileNetModelPath)) return;

        var options = new OnnxImageClassificationOptions
        {
            ModelPath = MobileNetModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet,
            TopK = 5
        };

        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        using var img1 = CreateTestImage(224, 224, 200, 50, 50);
        using var img2 = CreateTestImage(224, 224, 50, 200, 50);
        using var img3 = CreateTestImage(224, 224, 50, 50, 200);

        var single1 = transformer.Classify(img1);
        var single2 = transformer.Classify(img2);
        var single3 = transformer.Classify(img3);

        var batchResults = transformer.ClassifyBatch([img1, img2, img3]);

        Assert.Equal(3, batchResults.Length);

        // Dynamic batching may have small floating-point differences
        AssertClassificationResultsApproximate(single1, batchResults[0]);
        AssertClassificationResultsApproximate(single2, batchResults[1]);
        AssertClassificationResultsApproximate(single3, batchResults[2]);
    }

    // --- Classification: Batch of 1 ---

    [Fact]
    public void ClassifyBatch_SingleImage_MatchesSingle()
    {
        if (!File.Exists(MobileNetModelPath)) return;

        var options = new OnnxImageClassificationOptions
        {
            ModelPath = MobileNetModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet,
            TopK = 5
        };

        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        using var image = CreateTestImage(224, 224, 128, 128, 128);

        var singleResult = transformer.Classify(image);
        var batchResults = transformer.ClassifyBatch([image]);

        Assert.Single(batchResults);
        AssertClassificationResultsApproximate(singleResult, batchResults[0]);
    }

    // --- Embedding: Dynamic-batch model (CLIP) ---

    [Fact]
    public void EmbeddingBatch_DynamicModel_MatchesSingleResults()
    {
        if (!File.Exists(ClipVisionModelPath)) return;

        var options = new OnnxImageEmbeddingOptions
        {
            ModelPath = ClipVisionModelPath,
            PreprocessorConfig = PreprocessorConfig.CLIP,
            Pooling = PoolingStrategy.ClsToken,
            Normalize = true
        };

        var transformer = new OnnxImageEmbeddingTransformer(options);
        _disposables.Add(transformer);

        using var img1 = CreateTestImage(224, 224, 255, 0, 0);
        using var img2 = CreateTestImage(224, 224, 0, 255, 0);
        using var img3 = CreateTestImage(224, 224, 0, 0, 255);

        var single1 = transformer.GenerateEmbedding(img1);
        var single2 = transformer.GenerateEmbedding(img2);
        var single3 = transformer.GenerateEmbedding(img3);

        var batchResults = transformer.GenerateEmbeddingBatch([img1, img2, img3]);

        Assert.Equal(3, batchResults.Length);

        AssertEmbeddingsClose(single1, batchResults[0], tolerance: 1e-4f);
        AssertEmbeddingsClose(single2, batchResults[1], tolerance: 1e-4f);
        AssertEmbeddingsClose(single3, batchResults[2], tolerance: 1e-4f);
    }

    // --- Embedding: Batch of 1 ---

    [Fact]
    public void EmbeddingBatch_SingleImage_MatchesSingle()
    {
        if (!File.Exists(ClipVisionModelPath)) return;

        var options = new OnnxImageEmbeddingOptions
        {
            ModelPath = ClipVisionModelPath,
            PreprocessorConfig = PreprocessorConfig.CLIP,
            Pooling = PoolingStrategy.ClsToken,
            Normalize = true
        };

        var transformer = new OnnxImageEmbeddingTransformer(options);
        _disposables.Add(transformer);

        using var image = CreateTestImage(224, 224, 100, 150, 200);

        var singleResult = transformer.GenerateEmbedding(image);
        var batchResults = transformer.GenerateEmbeddingBatch([image]);

        Assert.Single(batchResults);
        AssertEmbeddingsClose(singleResult, batchResults[0], tolerance: 1e-4f);
    }

    // --- Empty input ---

    [Fact]
    public void ClassifyBatch_EmptyInput_ReturnsEmpty()
    {
        if (!File.Exists(SqueezeNetModelPath)) return;

        var options = new OnnxImageClassificationOptions
        {
            ModelPath = SqueezeNetModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet
        };

        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        var batchResults = transformer.ClassifyBatch([]);

        Assert.NotNull(batchResults);
        Assert.Empty(batchResults);
    }

    // --- IsBatchDynamic property ---

    [Fact]
    public void IsBatchDynamic_CorrectForAllModels()
    {
        if (File.Exists(SqueezeNetModelPath))
        {
            var squeezeMeta = ModelMetadataDiscovery.Discover(SqueezeNetModelPath);
            Assert.False(squeezeMeta.IsBatchDynamic, "SqueezeNet should have fixed batch");
        }

        if (File.Exists(MobileNetModelPath))
        {
            var mobileMeta = ModelMetadataDiscovery.Discover(MobileNetModelPath);
            Assert.True(mobileMeta.IsBatchDynamic, "MobileNet should have dynamic batch");
        }

        if (File.Exists(ClipVisionModelPath))
        {
            var clipMeta = ModelMetadataDiscovery.Discover(ClipVisionModelPath);
            Assert.True(clipMeta.IsBatchDynamic, "CLIP vision model should have dynamic batch");
        }
    }

    // --- Helpers ---

    private static void AssertClassificationResultsExact(
        (string Label, float Probability)[] expected,
        (string Label, float Probability)[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Label, actual[i].Label);
            Assert.Equal(expected[i].Probability, actual[i].Probability);
        }
    }

    private static void AssertClassificationResultsApproximate(
        (string Label, float Probability)[] expected,
        (string Label, float Probability)[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);

        // Top labels should match
        Assert.Equal(expected[0].Label, actual[0].Label);

        // Probabilities should be very close
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Probability, actual[i].Probability, precision: 4);
        }
    }

    private static void AssertEmbeddingsClose(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(
                MathF.Abs(expected[i] - actual[i]) <= tolerance,
                $"Embedding mismatch at index {i}: expected {expected[i]}, actual {actual[i]}, diff {MathF.Abs(expected[i] - actual[i])}");
        }
    }

    private static MLImage CreateTestImage(int width, int height, byte r, byte g, byte b)
    {
        var pixels = new byte[width * height * 4];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = (y * width + x) * 4;
                pixels[idx + 0] = (byte)(r * x / width);
                pixels[idx + 1] = (byte)(g * y / height);
                pixels[idx + 2] = b;
                pixels[idx + 3] = 255;
            }
        }
        return MLImage.CreateFromPixels(width, height, MLPixelFormat.Rgba32, pixels);
    }
}

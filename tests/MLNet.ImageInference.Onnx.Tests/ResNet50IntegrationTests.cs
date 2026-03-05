using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Classification;
using MLNet.ImageInference.Onnx.Shared;
using Xunit;

namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Integration tests using real ResNet-50 ONNX model (1000 ImageNet classes).
/// These tests require the model to be downloaded first:
///   pwsh scripts/download-test-models.ps1
/// </summary>
public class ResNet50IntegrationTests : IDisposable
{
    private const string ModelPath = "models/resnet50/resnet50-v1-7.onnx";

    private static bool ModelExists => File.Exists(ModelPath);

    private readonly List<IDisposable> _disposables = [];

    public void Dispose()
    {
        foreach (var d in _disposables)
            d.Dispose();
    }

    [Fact]
    public void ModelMetadata_HasExpectedInputOutput()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

        var meta = ModelMetadataDiscovery.Discover(ModelPath);

        Assert.Single(meta.InputNames);
        Assert.Equal("data", meta.InputNames[0]);

        Assert.Single(meta.OutputNames);
        Assert.Equal("resnetv17_dense0_fwd", meta.OutputNames[0]);

        // Input shape: [N, 3, 224, 224] (dynamic batch dimension)
        Assert.Equal(4, meta.InputShapes[0].Length);
        Assert.Equal(3, meta.InputShapes[0][1]);
        Assert.Equal(224, meta.InputShapes[0][2]);
        Assert.Equal(224, meta.InputShapes[0][3]);

        // Output shape: [N, 1000]
        Assert.Equal(2, meta.OutputShapes[0].Length);
        Assert.Equal(1000, meta.OutputShapes[0][1]);
    }

    [Fact]
    public void Classify_ReturnsValidProbabilities()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

        var options = new OnnxImageClassificationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet,
            TopK = 5
        };

        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        using var image = CreateTestImage(224, 224);

        var results = transformer.Classify(image);

        Assert.NotNull(results);
        Assert.Equal(5, results.Length);

        foreach (var (label, prob) in results)
        {
            Assert.InRange(prob, 0f, 1f);
            Assert.NotNull(label);
            Assert.NotEmpty(label);
        }

        // Top prediction should have nonzero confidence
        Assert.True(results[0].Probability > 0f);

        // Results should be sorted descending
        for (int i = 1; i < results.Length; i++)
            Assert.True(results[i].Probability <= results[i - 1].Probability);
    }

    [Fact]
    public void Classify_AllClasses_Returns1000Entries()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

        var options = new OnnxImageClassificationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet
            // No TopK — return all classes
        };

        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        using var image = CreateTestImage(224, 224);

        var results = transformer.Classify(image);

        // ResNet-50 has 1000 output classes (standard ImageNet, no background class)
        Assert.Equal(1000, results.Length);

        // OnnxImageClassificationTransformer applies softmax internally,
        // so probabilities should sum to ~1.0
        float sum = results.Sum(r => r.Probability);
        Assert.InRange(sum, 0.99f, 1.01f);
    }

    [Fact]
    public void Classify_TopK_ReturnsExactCount()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

        var options = new OnnxImageClassificationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet,
            TopK = 5
        };

        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        using var image = CreateTestImage(224, 224);

        var results = transformer.Classify(image);

        Assert.Equal(5, results.Length);
    }

    [Fact]
    public void Classify_DifferentFromSqueezeNet()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

        const string squeezeNetPath = "models/squeezenet/model.onnx";
        Skip.Unless(File.Exists(squeezeNetPath), "Model file not available - run scripts/download-test-models.ps1");

        // Run ResNet-50
        var resNetOptions = new OnnxImageClassificationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet,
            TopK = 5
        };
        var resNetTransformer = new OnnxImageClassificationTransformer(resNetOptions);
        _disposables.Add(resNetTransformer);

        // Run SqueezeNet
        var squeezeNetOptions = new OnnxImageClassificationOptions
        {
            ModelPath = squeezeNetPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet,
            TopK = 5
        };
        var squeezeNetTransformer = new OnnxImageClassificationTransformer(squeezeNetOptions);
        _disposables.Add(squeezeNetTransformer);

        using var image = CreateTestImage(224, 224);

        var resNetResults = resNetTransformer.Classify(image);
        var squeezeNetResults = squeezeNetTransformer.Classify(image);

        // Both should produce valid results
        Assert.Equal(5, resNetResults.Length);
        Assert.Equal(5, squeezeNetResults.Length);

        // Predictions should differ — different models produce different outputs
        bool anyDifference = false;
        for (int i = 0; i < resNetResults.Length; i++)
        {
            if (Math.Abs(resNetResults[i].Probability - squeezeNetResults[i].Probability) > 1e-6f)
            {
                anyDifference = true;
                break;
            }
        }
        Assert.True(anyDifference, "ResNet-50 and SqueezeNet should produce different predictions");
    }

    private static MLImage CreateTestImage(int width, int height)
    {
        var pixels = new byte[width * height * 4];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = (y * width + x) * 4;
                pixels[idx + 0] = (byte)(x * 255 / width);  // R
                pixels[idx + 1] = (byte)(y * 255 / height); // G
                pixels[idx + 2] = 128;                       // B
                pixels[idx + 3] = 255;                       // A
            }
        }
        return MLImage.CreateFromPixels(width, height, MLPixelFormat.Rgba32, pixels);
    }
}

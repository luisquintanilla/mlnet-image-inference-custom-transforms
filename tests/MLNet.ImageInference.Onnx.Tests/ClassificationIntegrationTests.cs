using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Classification;
using MLNet.ImageInference.Onnx.Shared;
using Xunit;

namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Integration tests using real SqueezeNet ONNX model.
/// These tests require the model to be downloaded first:
///   pwsh scripts/download-test-models.ps1
/// </summary>
public class ClassificationIntegrationTests : IDisposable
{
    private const string ModelPath = "models/squeezenet/model.onnx";
    private const string LabelsPath = "models/squeezenet/imagenet_classes.txt";

    private static bool ModelExists => File.Exists(ModelPath);

    private static string[] LoadLabels() => File.ReadAllLines(LabelsPath);

    private readonly List<IDisposable> _disposables = [];

    public void Dispose()
    {
        foreach (var d in _disposables)
            d.Dispose();
    }

    [Fact]
    public void ModelMetadata_HasExpectedInputOutput()
    {
        if (!ModelExists) return;

        var meta = ModelMetadataDiscovery.Discover(ModelPath);

        Assert.Single(meta.InputNames);
        Assert.Equal("data_0", meta.InputNames[0]);

        Assert.Single(meta.OutputNames);
        Assert.Equal("softmaxout_1", meta.OutputNames[0]);

        // Input shape: [1, 3, 224, 224]
        Assert.Equal(4, meta.InputShapes[0].Length);
        Assert.Equal(3, meta.InputShapes[0][1]);
        Assert.Equal(224, meta.InputShapes[0][2]);
        Assert.Equal(224, meta.InputShapes[0][3]);
    }

    [Fact]
    public void Classify_WithRealModel_ReturnsValidProbabilities()
    {
        if (!ModelExists) return;

        var labels = LoadLabels();
        var options = new OnnxImageClassificationOptions
        {
            ModelPath = ModelPath,
            Labels = labels,
            PreprocessorConfig = PreprocessorConfig.ImageNet,
            TopK = 5
        };

        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        using var image = CreateTestImage(224, 224);

        var results = transformer.Classify(image);

        Assert.NotNull(results);
        Assert.Equal(5, results.Length);

        // Verify probabilities are valid
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
    public void Classify_AllProbabilities_SumToOne()
    {
        if (!ModelExists) return;

        var options = new OnnxImageClassificationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet
        };

        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        using var image = CreateTestImage(224, 224);

        var results = transformer.Classify(image);

        Assert.Equal(1000, results.Length);

        float sum = results.Sum(r => r.Probability);
        Assert.InRange(sum, 0.99f, 1.01f);
    }

    [Fact]
    public void Transform_IDataView_ReturnsResults()
    {
        if (!ModelExists) return;

        var labels = LoadLabels();
        var options = new OnnxImageClassificationOptions
        {
            ModelPath = ModelPath,
            Labels = labels,
            PreprocessorConfig = PreprocessorConfig.ImageNet,
            TopK = 5
        };

        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        var mlContext = new MLContext();
        var image = CreateTestImage(224, 224);
        _disposables.Add(image);

        var data = mlContext.Data.LoadFromEnumerable(new[] { new ImageInput { Image = image } });

        var result = transformer.Transform(data);

        // Verify schema has output columns
        Assert.NotNull(result.Schema.GetColumnOrNull("PredictedLabel"));
        Assert.NotNull(result.Schema.GetColumnOrNull("Probability"));

        // Iterate and verify
        using var cursor = result.GetRowCursor(result.Schema);
        Assert.True(cursor.MoveNext());

        var labelCol = result.Schema["PredictedLabel"];
        var labelGetter = cursor.GetGetter<ReadOnlyMemory<char>>(labelCol);
        ReadOnlyMemory<char> predictedLabel = default;
        labelGetter(ref predictedLabel);
        Assert.False(predictedLabel.IsEmpty);

        var probCol = result.Schema["Probability"];
        var probGetter = cursor.GetGetter<VBuffer<float>>(probCol);
        VBuffer<float> probs = default;
        probGetter(ref probs);
        Assert.True(probs.Length > 0);
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

public class ImageInput
{
    [ImageType(224, 224)]
    public MLImage Image { get; set; } = null!;
}

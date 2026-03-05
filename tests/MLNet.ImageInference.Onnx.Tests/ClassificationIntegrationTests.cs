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

    [SkippableFact]
    public void ModelMetadata_HasExpectedInputOutput()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

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

    [SkippableFact]
    public void Classify_WithRealModel_ReturnsValidProbabilities()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

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

    [SkippableFact]
    public void Classify_AllProbabilities_SumToOne()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

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

    [SkippableFact]
    public void Transform_IDataView_ReturnsResults()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

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

        var image = CreateTestImage(224, 224);
        _disposables.Add(image);

        var sourceDataView = new SingleImageDataView(image);
        var result = transformer.Transform(sourceDataView);

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

    /// <summary>
    /// Minimal IDataView that serves a single MLImage row for testing the Transform pipeline.
    /// </summary>
    private sealed class SingleImageDataView : IDataView
    {
        private readonly MLImage _image;
        private readonly DataViewSchema _schema;

        public SingleImageDataView(MLImage image)
        {
            _image = image;
            var builder = new DataViewSchema.Builder();
            builder.AddColumn("Image", NumberDataViewType.Single); // Type is not checked; cursor serves MLImage directly
            _schema = builder.ToSchema();
        }

        public DataViewSchema Schema => _schema;
        public bool CanShuffle => false;
        public long? GetRowCount() => 1;

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
            => new SingleImageCursor(this);

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
            => [GetRowCursor(columnsNeeded, rand)];

        private sealed class SingleImageCursor : DataViewRowCursor
        {
            private readonly SingleImageDataView _parent;
            private long _position = -1;

            public SingleImageCursor(SingleImageDataView parent) => _parent = parent;
            public override DataViewSchema Schema => _parent._schema;
            public override long Position => _position;
            public override long Batch => 0;
            public override bool IsColumnActive(DataViewSchema.Column column) => true;
            public override bool MoveNext() => ++_position == 0;
            public override ValueGetter<DataViewRowId> GetIdGetter() =>
                (ref DataViewRowId id) => id = new DataViewRowId((ulong)_position, 0);

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                ValueGetter<MLImage> getter = (ref MLImage value) => value = _parent._image;
                return (ValueGetter<TValue>)(Delegate)getter;
            }
        }
    }
}

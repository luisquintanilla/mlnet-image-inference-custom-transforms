using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Detection;
using MLNet.ImageInference.Onnx.Shared;
using Xunit;

namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Integration tests using real YOLOv8n ONNX model.
/// These tests require the model to be downloaded first:
///   pwsh scripts/download-test-models.ps1
/// </summary>
public class DetectionIntegrationTests : IDisposable
{
    private const string ModelPath = "models/yolov8/model.onnx";

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
        Assert.Equal("images", meta.InputNames[0]);

        Assert.Single(meta.OutputNames);
        Assert.Equal("output0", meta.OutputNames[0]);

        // Input shape: [1, 3, 640, 640]
        Assert.Equal(4, meta.InputShapes[0].Length);
        Assert.Equal(1, meta.InputShapes[0][0]);
        Assert.Equal(3, meta.InputShapes[0][1]);
        Assert.Equal(640, meta.InputShapes[0][2]);
        Assert.Equal(640, meta.InputShapes[0][3]);

        // Output shape: [1, 84, 8400]
        Assert.Equal(3, meta.OutputShapes[0].Length);
        Assert.Equal(1, meta.OutputShapes[0][0]);
        Assert.Equal(84, meta.OutputShapes[0][1]);
        Assert.Equal(8400, meta.OutputShapes[0][2]);
    }

    [Fact]
    public void Detect_WithSyntheticImage_ReturnsBoundingBoxArray()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

        var options = new OnnxObjectDetectionOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.YOLOv8,
            ConfidenceThreshold = 0.5f,
            IouThreshold = 0.45f,
            MaxDetections = 100
        };

        var transformer = new OnnxObjectDetectionTransformer(options);
        _disposables.Add(transformer);

        using var image = CreateTestImage(640, 640);
        var results = transformer.Detect(image);

        Assert.NotNull(results);
        Assert.IsType<BoundingBox[]>(results);

        // Validate each box has valid properties
        foreach (var box in results)
        {
            Assert.True(box.X >= 0, $"Box X should be >= 0, was {box.X}");
            Assert.True(box.Y >= 0, $"Box Y should be >= 0, was {box.Y}");
            Assert.True(box.Width > 0, $"Box Width should be > 0, was {box.Width}");
            Assert.True(box.Height > 0, $"Box Height should be > 0, was {box.Height}");
            Assert.InRange(box.Score, 0f, 1f);
            Assert.InRange(box.ClassId, 0, 79);
        }
    }

    [Fact]
    public void Detect_LowConfidenceReturnsMoreBoxes()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

        using var image = CreateTestImage(640, 640);

        var highConfOptions = new OnnxObjectDetectionOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.YOLOv8,
            ConfidenceThreshold = 0.9f,
            IouThreshold = 0.45f
        };

        var highConfTransformer = new OnnxObjectDetectionTransformer(highConfOptions);
        _disposables.Add(highConfTransformer);
        var highConfResults = highConfTransformer.Detect(image);

        var lowConfOptions = new OnnxObjectDetectionOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.YOLOv8,
            ConfidenceThreshold = 0.01f,
            IouThreshold = 0.45f
        };

        var lowConfTransformer = new OnnxObjectDetectionTransformer(lowConfOptions);
        _disposables.Add(lowConfTransformer);
        var lowConfResults = lowConfTransformer.Detect(image);

        Assert.True(lowConfResults.Length >= highConfResults.Length,
            $"Low confidence ({lowConfResults.Length}) should return >= boxes than high confidence ({highConfResults.Length})");
    }

    [Fact]
    public void Transform_IDataView_ProducesOutputColumns()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

        var options = new OnnxObjectDetectionOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.YOLOv8,
            ConfidenceThreshold = 0.5f,
            IouThreshold = 0.45f,
            MaxDetections = 100
        };

        var transformer = new OnnxObjectDetectionTransformer(options);
        _disposables.Add(transformer);

        var image = CreateTestImage(640, 640);
        _disposables.Add(image);

        var sourceDataView = new SingleImageDataView(image);
        var result = transformer.Transform(sourceDataView);

        // Verify schema has detection output columns
        Assert.NotNull(result.Schema.GetColumnOrNull("DetectedObjects_Boxes"));
        Assert.NotNull(result.Schema.GetColumnOrNull("DetectedObjects_Count"));

        // Iterate and verify cursor works
        using var cursor = result.GetRowCursor(result.Schema);
        Assert.True(cursor.MoveNext());

        var countCol = result.Schema["DetectedObjects_Count"];
        var countGetter = cursor.GetGetter<int>(countCol);
        int count = 0;
        countGetter(ref count);
        Assert.True(count >= 0, $"Detection count should be >= 0, was {count}");
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
            builder.AddColumn("Image", NumberDataViewType.Single);
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

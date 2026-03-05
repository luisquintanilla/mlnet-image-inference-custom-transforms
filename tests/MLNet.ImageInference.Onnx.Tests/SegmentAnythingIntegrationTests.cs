using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.SegmentAnything;
using Xunit;

namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Integration tests for SAM2 (Segment Anything Model v2) using real ONNX models.
/// </summary>
public class SegmentAnythingIntegrationTests : IDisposable
{
    private const string EncoderPath = "models/sam2-tiny/sam2_hiera_tiny_encoder.onnx";
    private const string DecoderPath = "models/sam2-tiny/sam2_hiera_tiny_decoder.onnx";

    private static bool ModelsExist =>
        File.Exists(EncoderPath) && File.Exists(DecoderPath);

    private readonly List<IDisposable> _disposables = [];

    public void Dispose()
    {
        foreach (var d in _disposables)
            d.Dispose();
    }

    [SkippableFact]
    public void Segment_WithPointPrompt_ProducesMask()
    {
        Skip.Unless(ModelsExist, "SAM2 models not found");

        using var transformer = new OnnxSegmentAnythingTransformer(CreateOptions());
        using var image = CreateTestImage(256, 256, r: 200, g: 100, b: 50);

        var prompt = SegmentAnythingPrompt.FromPoint(128f, 128f);
        var result = transformer.Segment(image, prompt);

        Assert.NotNull(result);
        Assert.True(result.NumMasks > 0, "Should produce at least one mask");
        Assert.True(result.Width > 0, "Mask width should be positive");
        Assert.True(result.Height > 0, "Mask height should be positive");
        Assert.True(result.IoUPredictions.Length > 0, "Should have IoU predictions");
    }

    [SkippableFact]
    public void Segment_WithBoundingBoxPrompt_ProducesMask()
    {
        Skip.Unless(ModelsExist, "SAM2 models not found");

        using var transformer = new OnnxSegmentAnythingTransformer(CreateOptions());
        using var image = CreateTestImage(256, 256, r: 200, g: 100, b: 50);

        var prompt = SegmentAnythingPrompt.FromBoundingBox(50f, 50f, 200f, 200f);
        var result = transformer.Segment(image, prompt);

        Assert.NotNull(result);
        Assert.True(result.NumMasks > 0, "Should produce at least one mask");
    }

    [SkippableFact]
    public void SegmentCenter_ProducesMask()
    {
        Skip.Unless(ModelsExist, "SAM2 models not found");

        using var transformer = new OnnxSegmentAnythingTransformer(CreateOptions());
        using var image = CreateTestImage(256, 256, r: 135, g: 206, b: 235);

        var result = transformer.SegmentCenter(image);

        Assert.NotNull(result);
        Assert.True(result.NumMasks > 0, "Should produce at least one mask");
        Assert.NotNull(result.GetBestMask());
        Assert.True(result.GetBestMask().Length > 0, "Best mask should have pixels");
    }

    [SkippableFact]
    public void EncodeImage_CacheAndReuse_MultiplePrompts()
    {
        Skip.Unless(ModelsExist, "SAM2 models not found");

        using var transformer = new OnnxSegmentAnythingTransformer(CreateOptions());
        using var image = CreateTestImage(256, 256, r: 200, g: 100, b: 50);

        // Encode once, segment multiple times with different prompts
        var embedding = transformer.EncodeImage(image);

        var prompt1 = SegmentAnythingPrompt.FromPoint(64f, 64f);
        var result1 = transformer.Segment(embedding, prompt1);

        var prompt2 = SegmentAnythingPrompt.FromPoint(192f, 192f);
        var result2 = transformer.Segment(embedding, prompt2);

        Assert.NotNull(result1);
        Assert.NotNull(result2);
        Assert.True(result1.NumMasks > 0);
        Assert.True(result2.NumMasks > 0);
    }

    [SkippableFact]
    public void Segment_MultiplePoints_ForegroundAndBackground()
    {
        Skip.Unless(ModelsExist, "SAM2 models not found");

        using var transformer = new OnnxSegmentAnythingTransformer(CreateOptions());
        using var image = CreateTestImage(256, 256, r: 200, g: 100, b: 50);

        var coords = new float[,] { { 128f, 128f }, { 10f, 10f } };
        var labels = new float[] { 1f, 0f }; // foreground, background
        var prompt = SegmentAnythingPrompt.FromPoints(coords, labels);
        var result = transformer.Segment(image, prompt);

        Assert.NotNull(result);
        Assert.True(result.NumMasks > 0);
    }

    [SkippableFact]
    public void Estimator_FitAndTransform()
    {
        Skip.Unless(ModelsExist, "SAM2 models not found");

        var options = CreateOptions();
        var estimator = new OnnxSegmentAnythingEstimator(options);
        using var transformer = estimator.Fit(null!);
        _disposables.Add(transformer);

        using var image = CreateTestImage(256, 256, r: 135, g: 206, b: 235);
        var result = transformer.SegmentCenter(image);

        Assert.NotNull(result);
        Assert.True(result.NumMasks > 0);
    }

    [SkippableFact]
    public void Transform_IDataView_ProducesMaskColumns()
    {
        Skip.Unless(ModelsExist, "SAM2 models not found");

        var options = CreateOptions();
        var estimator = new OnnxSegmentAnythingEstimator(options);
        using var transformer = estimator.Fit(null!);
        _disposables.Add(transformer);

        using var image = CreateTestImage(256, 256, r: 200, g: 100, b: 50);
        var sourceData = new SingleImageDataView(image);
        var result = transformer.Transform(sourceData);

        // Verify schema has the expected columns
        var maskCol = result.Schema["SegmentAnythingMask"];
        Assert.NotNull(maskCol);
        var widthCol = result.Schema["SegmentAnythingMask_Width"];
        Assert.NotNull(widthCol);
        var heightCol = result.Schema["SegmentAnythingMask_Height"];
        Assert.NotNull(heightCol);
        var iouCol = result.Schema["SegmentAnythingMask_IoU"];
        Assert.NotNull(iouCol);

        using var cursor = result.GetRowCursor(result.Schema);
        Assert.True(cursor.MoveNext());

        var widthGetter = cursor.GetGetter<int>(widthCol);
        int width = 0;
        widthGetter(ref width);
        Assert.True(width > 0, "Width should be positive");

        var heightGetter = cursor.GetGetter<int>(heightCol);
        int height = 0;
        heightGetter(ref height);
        Assert.True(height > 0, "Height should be positive");

        var iouGetter = cursor.GetGetter<float>(iouCol);
        float iou = 0;
        iouGetter(ref iou);
        Assert.True(iou > 0f || iou == 0f, "IoU should be a valid float");
    }

    [SkippableFact]
    public void MLContextExtension_CreatesEstimator()
    {
        Skip.Unless(ModelsExist, "SAM2 models not found");

        var mlContext = new MLContext();
        var estimator = mlContext.Transforms.OnnxSegmentAnything(CreateOptions());
        Assert.NotNull(estimator);

        using var transformer = estimator.Fit(null!);
        Assert.NotNull(transformer);
    }

    private static OnnxSegmentAnythingOptions CreateOptions() => new()
    {
        EncoderModelPath = EncoderPath,
        DecoderModelPath = DecoderPath,
        PreprocessorConfig = PreprocessorConfig.SAM2
    };

    private static MLImage CreateTestImage(int width, int height, byte r, byte g, byte b)
    {
        var pixels = new byte[width * height * 4];
        for (int i = 0; i < width * height; i++)
        {
            int idx = i * 4;
            pixels[idx + 0] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
            pixels[idx + 3] = 255;
        }
        return MLImage.CreateFromPixels(width, height, MLPixelFormat.Rgba32, pixels);
    }

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

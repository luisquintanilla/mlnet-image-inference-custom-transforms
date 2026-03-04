using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Segmentation;
using MLNet.ImageInference.Onnx.Shared;
using Xunit;

namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Integration tests using real DeepLabV3 (ResNet-50) ONNX model.
/// These tests require the model to be downloaded first:
///   pwsh scripts/download-test-models.ps1
///
/// DeepLabV3 has TWO outputs: "logits" [batch,21,520,520] and "627" (auxiliary).
/// The transformer uses results.First() which relies on ONNX output ordering.
/// </summary>
public class DeepLabV3IntegrationTests : IDisposable
{
    private const string ModelPath = "models/deeplabv3/deeplabv3_resnet50.onnx";
    private const int NumPascalVOCClasses = 21;
    private const int ModelInputSize = 520;

    private static readonly PreprocessorConfig DeepLabV3Config = new()
    {
        ImageSize = (ModelInputSize, ModelInputSize),
        CropSize = (ModelInputSize, ModelInputSize),
        DoCenterCrop = false
    };

    private static bool ModelExists => File.Exists(ModelPath);

    private readonly List<IDisposable> _disposables = [];

    public void Dispose()
    {
        foreach (var d in _disposables)
            d.Dispose();
    }

    [Fact]
    public void ModelMetadata_HasExpectedInput()
    {
        if (!ModelExists) return;

        var meta = ModelMetadataDiscovery.Discover(ModelPath);

        Assert.Single(meta.InputNames);
        Assert.Equal("pixel_values", meta.InputNames[0]);

        // Input shape: [batch, 3, 520, 520] — dynamic batch reports -1
        Assert.Equal(4, meta.InputShapes[0].Length);

        // DeepLabV3 has two outputs: "logits" and "627" (auxiliary)
        Assert.Equal(2, meta.OutputNames.Length);
        Assert.Contains("logits", meta.OutputNames);

        // logits shape has 4 dimensions: [batch, 21, 520, 520]
        var logitsIdx = Array.IndexOf(meta.OutputNames, "logits");
        Assert.Equal(4, meta.OutputShapes[logitsIdx].Length);
    }

    [Fact]
    public void Segment_ReturnsValidSegmentationMask()
    {
        if (!ModelExists) return;

        var options = new OnnxImageSegmentationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = DeepLabV3Config,
            ResizeToOriginal = true
        };

        var transformer = new OnnxImageSegmentationTransformer(options);
        _disposables.Add(transformer);

        using var image = CreateTestImage(ModelInputSize, ModelInputSize);
        var mask = transformer.Segment(image);

        Assert.NotNull(mask);
        Assert.NotNull(mask.ClassIds);
        Assert.NotEmpty(mask.ClassIds);

        // ResizeToOriginal=true: mask should match original image dimensions
        Assert.Equal(ModelInputSize, mask.Width);
        Assert.Equal(ModelInputSize, mask.Height);
        Assert.Equal(ModelInputSize * ModelInputSize, mask.ClassIds.Length);

        // All class IDs should be valid Pascal VOC classes [0, 20]
        foreach (var classId in mask.ClassIds)
        {
            Assert.InRange(classId, 0, NumPascalVOCClasses - 1);
        }
    }

    [Fact]
    public void Segment_GetClassAt_ReturnsValidClassIds()
    {
        if (!ModelExists) return;

        var options = new OnnxImageSegmentationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = DeepLabV3Config,
            ResizeToOriginal = true
        };

        var transformer = new OnnxImageSegmentationTransformer(options);
        _disposables.Add(transformer);

        using var image = CreateTestImage(ModelInputSize, ModelInputSize);
        var mask = transformer.Segment(image);

        // Sample various pixel positions
        int[] testPositions = [0, 100, 259, 400, 519];
        foreach (var x in testPositions)
        {
            foreach (var y in testPositions)
            {
                int classId = mask.GetClassAt(x, y);
                Assert.InRange(classId, 0, NumPascalVOCClasses - 1);
            }
        }
    }

    [Fact]
    public void Segment_MaskDimensions_MatchExpected()
    {
        if (!ModelExists) return;

        var options = new OnnxImageSegmentationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = DeepLabV3Config,
            ResizeToOriginal = false
        };

        var transformer = new OnnxImageSegmentationTransformer(options);
        _disposables.Add(transformer);

        using var image = CreateTestImage(ModelInputSize, ModelInputSize);
        var mask = transformer.Segment(image);

        // Without resize, mask dimensions come from the model's native output spatial resolution
        Assert.True(mask.Width > 0, "Mask width should be positive");
        Assert.True(mask.Height > 0, "Mask height should be positive");
        Assert.Equal(mask.Width * mask.Height, mask.ClassIds.Length);

        // DeepLabV3 output is 520x520 (no downsampling stride like SegFormer)
        Assert.Equal(ModelInputSize, mask.Width);
        Assert.Equal(ModelInputSize, mask.Height);
    }

    [Fact]
    public void Transform_IDataView_ProducesOutputColumns()
    {
        if (!ModelExists) return;

        var options = new OnnxImageSegmentationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = DeepLabV3Config,
            ResizeToOriginal = true
        };

        var transformer = new OnnxImageSegmentationTransformer(options);
        _disposables.Add(transformer);

        var image = CreateTestImage(ModelInputSize, ModelInputSize);
        _disposables.Add(image);

        var sourceDataView = new SingleImageDataView(image);
        var result = transformer.Transform(sourceDataView);

        // Verify schema has segmentation output columns
        Assert.NotNull(result.Schema.GetColumnOrNull("SegmentationMask"));
        Assert.NotNull(result.Schema.GetColumnOrNull("SegmentationMask_Width"));
        Assert.NotNull(result.Schema.GetColumnOrNull("SegmentationMask_Height"));

        // Iterate and verify data
        using var cursor = result.GetRowCursor(result.Schema);
        Assert.True(cursor.MoveNext());

        var maskCol = result.Schema["SegmentationMask"];
        var maskGetter = cursor.GetGetter<VBuffer<int>>(maskCol);
        VBuffer<int> maskBuffer = default;
        maskGetter(ref maskBuffer);
        Assert.True(maskBuffer.Length > 0);

        var widthCol = result.Schema["SegmentationMask_Width"];
        var widthGetter = cursor.GetGetter<int>(widthCol);
        int width = 0;
        widthGetter(ref width);
        Assert.Equal(ModelInputSize, width);

        var heightCol = result.Schema["SegmentationMask_Height"];
        var heightGetter = cursor.GetGetter<int>(heightCol);
        int height = 0;
        heightGetter(ref height);
        Assert.Equal(ModelInputSize, height);
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

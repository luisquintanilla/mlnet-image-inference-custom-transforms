using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.DepthEstimation;
using MLNet.ImageInference.Onnx.Shared;
using Xunit;

namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Integration tests using real DPT-Hybrid ONNX model.
/// These tests require the model to be downloaded first:
///   python -c "from transformers import DPTForDepthEstimation; import torch; ..."
/// </summary>
public class DepthEstimationIntegrationTests : IDisposable
{
    private const string ModelPath = "models/dpt-hybrid/model.onnx";

    private static bool ModelExists => File.Exists(ModelPath);

    private readonly List<IDisposable> _disposables = [];

    public void Dispose()
    {
        foreach (var d in _disposables)
            d.Dispose();
    }

    [SkippableFact]
    public void ModelMetadata_HasExpectedInputOutput()
    {
        Skip.Unless(ModelExists, "DPT-Hybrid model not found");

        var options = CreateOptions();
        var estimator = new OnnxImageDepthEstimationEstimator(options);
        using var transformer = estimator.Fit(null!);

        // Verify the transformer works by estimating depth
        using var image = CreateTestImage(384, 384);
        var depthMap = transformer.Estimate(image);

        Assert.NotNull(depthMap);
        Assert.Equal(384, depthMap.Width);
        Assert.Equal(384, depthMap.Height);
    }

    [SkippableFact]
    public void Estimate_ReturnsValidDepthMap()
    {
        Skip.Unless(ModelExists, "DPT-Hybrid model not found");

        var options = CreateOptions();
        using var transformer = new OnnxImageDepthEstimationTransformer(options);

        using var image = CreateTestImage(384, 384);
        var depthMap = transformer.Estimate(image);

        Assert.NotNull(depthMap);
        Assert.Equal(384, depthMap.Width);
        Assert.Equal(384, depthMap.Height);
        Assert.Equal(384 * 384, depthMap.Values.Length);

        // Values should be normalized to [0, 1]
        Assert.True(depthMap.Values.All(v => v >= 0f && v <= 1f),
            "All depth values should be in [0, 1]");
        Assert.True(depthMap.MinDepth < depthMap.MaxDepth,
            "Min depth should be less than max depth");
    }

    [SkippableFact]
    public void Estimate_WithResizeToOriginal()
    {
        Skip.Unless(ModelExists, "DPT-Hybrid model not found");

        var options = new OnnxImageDepthEstimationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.DPT,
            ResizeToOriginal = true
        };
        using var transformer = new OnnxImageDepthEstimationTransformer(options);

        // Create a non-standard size image (the preprocessor will resize to 384x384 for the model,
        // and the post-processor should resize back to 512x512)
        using var image = CreateTestImage(512, 512);
        var depthMap = transformer.Estimate(image);

        // Depth map should match original image dimensions
        Assert.Equal(512, depthMap.Width);
        Assert.Equal(512, depthMap.Height);
        Assert.Equal(512 * 512, depthMap.Values.Length);
    }

    [SkippableFact]
    public void Estimate_WithoutResize_ReturnsModelNativeResolution()
    {
        Skip.Unless(ModelExists, "DPT-Hybrid model not found");

        var options = new OnnxImageDepthEstimationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.DPT,
            ResizeToOriginal = false
        };
        using var transformer = new OnnxImageDepthEstimationTransformer(options);

        using var image = CreateTestImage(512, 512);
        var depthMap = transformer.Estimate(image);

        // Should be model native resolution (384x384)
        Assert.Equal(384, depthMap.Width);
        Assert.Equal(384, depthMap.Height);
    }

    [SkippableFact]
    public void Estimate_GetDepthAt_ReturnsValidValues()
    {
        Skip.Unless(ModelExists, "DPT-Hybrid model not found");

        var options = CreateOptions();
        using var transformer = new OnnxImageDepthEstimationTransformer(options);

        using var image = CreateTestImage(384, 384);
        var depthMap = transformer.Estimate(image);

        // Sample various pixel locations
        int[] coords = [0, 50, 191, 300, 383];
        foreach (int x in coords)
        {
            foreach (int y in coords)
            {
                float depth = depthMap.GetDepthAt(x, y);
                Assert.InRange(depth, 0f, 1f);
            }
        }
    }

    [SkippableFact]
    public void EstimateBatch_ReturnsBatchResults()
    {
        Skip.Unless(ModelExists, "DPT-Hybrid model not found");

        var options = CreateOptions();
        using var transformer = new OnnxImageDepthEstimationTransformer(options);

        using var img1 = CreateTestImage(384, 384);
        using var img2 = CreateTestImage(384, 384);
        var results = transformer.EstimateBatch([img1, img2]);

        Assert.Equal(2, results.Length);
        foreach (var dm in results)
        {
            Assert.Equal(384, dm.Width);
            Assert.Equal(384, dm.Height);
            Assert.True(dm.Values.All(v => v >= 0f && v <= 1f));
        }
    }

    [SkippableFact]
    public void Transform_IDataView_ProducesOutputColumns()
    {
        Skip.Unless(ModelExists, "DPT-Hybrid model not found");

        var options = CreateOptions();
        var estimator = new OnnxImageDepthEstimationEstimator(options);
        using var transformer = estimator.Fit(null!);
        _disposables.Add(transformer);

        using var image = CreateTestImage(384, 384);
        var sourceData = new SingleImageDataView(image);
        var result = transformer.Transform(sourceData);

        Assert.NotNull(result.Schema["DepthMap"]);
        Assert.NotNull(result.Schema["DepthMap_Width"]);
        Assert.NotNull(result.Schema["DepthMap_Height"]);

        using var cursor = result.GetRowCursor(result.Schema);
        Assert.True(cursor.MoveNext());

        var depthCol = result.Schema["DepthMap"];
        var depthGetter = cursor.GetGetter<VBuffer<float>>(depthCol);
        VBuffer<float> depthBuffer = default;
        depthGetter(ref depthBuffer);
        Assert.True(depthBuffer.Length > 0);

        var widthCol = result.Schema["DepthMap_Width"];
        var widthGetter = cursor.GetGetter<int>(widthCol);
        int width = 0;
        widthGetter(ref width);
        Assert.Equal(384, width);

        var heightCol = result.Schema["DepthMap_Height"];
        var heightGetter = cursor.GetGetter<int>(heightCol);
        int height = 0;
        heightGetter(ref height);
        Assert.Equal(384, height);
    }

    [SkippableFact]
    public void MLContextExtension_CreatesEstimator()
    {
        Skip.Unless(ModelExists, "DPT-Hybrid model not found");

        var mlContext = new MLContext();
        var estimator = mlContext.Transforms.OnnxImageDepthEstimation(CreateOptions());
        Assert.NotNull(estimator);

        using var transformer = estimator.Fit(null!);
        Assert.NotNull(transformer);
    }

    private static OnnxImageDepthEstimationOptions CreateOptions() => new()
    {
        ModelPath = ModelPath,
        PreprocessorConfig = PreprocessorConfig.DPT,
        ResizeToOriginal = false
    };

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

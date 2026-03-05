using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Embeddings;
using MLNet.ImageInference.Onnx.MEAI;
using MLNet.ImageInference.Onnx.Shared;
using Xunit;

namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Integration tests for DINOv2 ViT-S/14 image embeddings using the ONNX model.
/// Requires models/dinov2/dinov2_vits14.onnx (exported from facebook/dinov2-small).
/// </summary>
public class DINOv2IntegrationTests : IDisposable
{
    private const string ModelPath = "models/dinov2/dinov2_vits14.onnx";

    private static bool ModelExists => File.Exists(ModelPath);

    private readonly List<IDisposable> _disposables = [];

    public void Dispose()
    {
        foreach (var d in _disposables)
            d.Dispose();
    }

    private OnnxImageEmbeddingOptions CreateOptions() => new()
    {
        ModelPath = ModelPath,
        PreprocessorConfig = PreprocessorConfig.DINOv2,
        Pooling = PoolingStrategy.ClsToken,
        Normalize = true
    };

    [SkippableFact]
    public void ModelMetadata_HasExpectedInputOutput()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

        var meta = ModelMetadataDiscovery.Discover(ModelPath);

        Assert.Single(meta.InputNames);
        Assert.Equal("input", meta.InputNames[0]);

        Assert.Single(meta.OutputNames);

        // Input shape: [batch, 3, 224, 224]
        Assert.Equal(4, meta.InputShapes[0].Length);
        Assert.Equal(3, meta.InputShapes[0][1]);
        Assert.Equal(224, meta.InputShapes[0][2]);
        Assert.Equal(224, meta.InputShapes[0][3]);

        // Output shape: [batch, 384]
        Assert.Equal(2, meta.OutputShapes[0].Length);
        Assert.Equal(384, meta.OutputShapes[0][1]);
    }

    [SkippableFact]
    public void GenerateEmbedding_Returns384DimVector()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

        var transformer = new OnnxImageEmbeddingTransformer(CreateOptions());
        _disposables.Add(transformer);

        using var image = CreateTestImage(224, 224, 100, 150, 200);

        float[] embedding = transformer.GenerateEmbedding(image);

        Assert.Equal(384, embedding.Length);
        Assert.All(embedding, v => Assert.True(float.IsFinite(v), $"Non-finite value: {v}"));
    }

    [SkippableFact]
    public void Embedding_IsNormalized()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

        var transformer = new OnnxImageEmbeddingTransformer(CreateOptions());
        _disposables.Add(transformer);

        using var image = CreateTestImage(224, 224, 100, 150, 200);

        float[] embedding = transformer.GenerateEmbedding(image);

        // L2 norm ≈ 1.0 (tolerance 0.1 — DINOv2 may or may not normalize natively)
        float norm = MathF.Sqrt(embedding.Sum(v => v * v));
        Assert.InRange(norm, 0.9f, 1.1f);
    }

    [SkippableFact]
    public void DifferentImages_ProduceDifferentEmbeddings()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

        var transformer = new OnnxImageEmbeddingTransformer(CreateOptions());
        _disposables.Add(transformer);

        using var redImage = CreateSolidImage(224, 224, 255, 0, 0);
        using var blueImage = CreateSolidImage(224, 224, 0, 0, 255);

        float[] redEmbedding = transformer.GenerateEmbedding(redImage);
        float[] blueEmbedding = transformer.GenerateEmbedding(blueImage);

        // Cosine similarity should be < 1.0 (not identical)
        float dot = 0f;
        for (int i = 0; i < redEmbedding.Length; i++)
            dot += redEmbedding[i] * blueEmbedding[i];

        // Both are L2-normalized, so dot product = cosine similarity
        Assert.True(dot < 0.999f, $"Embeddings are too similar: cosine={dot}");
    }

    [SkippableFact]
    public async Task MEAIGenerator_Works()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

        var transformer = new OnnxImageEmbeddingTransformer(CreateOptions());
        _disposables.Add(transformer);

        var generator = new OnnxImageEmbeddingGenerator(transformer, modelId: "dinov2-vits14");
        _disposables.Add(generator);

        using var image = CreateTestImage(224, 224, 50, 100, 200);

        var results = await generator.GenerateAsync([image]);

        Assert.NotNull(results);
        Assert.Single(results);
        Assert.Equal(384, results[0].Vector.Length);

        // Verify metadata
        Assert.Equal(384, generator.Metadata.DefaultModelDimensions);
        Assert.Equal("dinov2-vits14", generator.Metadata.DefaultModelId);
    }

    [SkippableFact]
    public void Transform_IDataView_ProducesEmbeddingColumn()
    {
        Skip.Unless(ModelExists, "Model file not available - run scripts/download-test-models.ps1");

        var transformer = new OnnxImageEmbeddingTransformer(CreateOptions());
        _disposables.Add(transformer);

        var image = CreateTestImage(224, 224, 128, 128, 128);
        _disposables.Add(image);

        var sourceDataView = new SingleImageDataView(image);
        var result = transformer.Transform(sourceDataView);

        // Verify schema has Embedding column
        var embeddingCol = result.Schema.GetColumnOrNull("Embedding");
        Assert.NotNull(embeddingCol);

        // Verify it's a vector of float with dimension 384
        var colType = embeddingCol.Value.Type as VectorDataViewType;
        Assert.NotNull(colType);
        Assert.Equal(384, colType.Size);

        // Read via cursor
        using var cursor = result.GetRowCursor(result.Schema);
        Assert.True(cursor.MoveNext());

        var getter = cursor.GetGetter<VBuffer<float>>(embeddingCol.Value);
        VBuffer<float> embeddingBuffer = default;
        getter(ref embeddingBuffer);

        Assert.Equal(384, embeddingBuffer.Length);

        // Verify values are finite
        var values = embeddingBuffer.DenseValues().ToArray();
        Assert.All(values, v => Assert.True(float.IsFinite(v)));

        // No more rows
        Assert.False(cursor.MoveNext());
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

    private static MLImage CreateSolidImage(int width, int height, byte r, byte g, byte b)
    {
        var pixels = new byte[width * height * 4];
        for (int i = 0; i < width * height; i++)
        {
            pixels[i * 4 + 0] = r;
            pixels[i * 4 + 1] = g;
            pixels[i * 4 + 2] = b;
            pixels[i * 4 + 3] = 255;
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

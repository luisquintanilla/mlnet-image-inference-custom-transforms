using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Classification;
using Xunit;

namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Tests for IDataView/Cursor implementations used by task transformers.
/// Uses the SqueezeNet ONNX model (classification) which is the only model
/// available in the test environment.
/// </summary>
public class DataViewTests : IDisposable
{
    private const string ModelPath = "models/squeezenet/model.onnx";
    private const string LabelsPath = "models/squeezenet/imagenet_classes.txt";

    private static bool ModelExists => File.Exists(ModelPath);

    private readonly List<IDisposable> _disposables = [];

    public void Dispose()
    {
        foreach (var d in _disposables)
            d.Dispose();
    }

    [Fact]
    public void Classification_Transform_SchemaHasOutputColumns()
    {
        if (!ModelExists) return;

        var options = new OnnxImageClassificationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet
        };
        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        var image = CreateTestImage(224, 224);
        _disposables.Add(image);

        var source = new SingleImageDataView(image);
        var result = transformer.Transform(source);

        // Output columns exist with correct types
        var labelCol = result.Schema.GetColumnOrNull("PredictedLabel");
        Assert.NotNull(labelCol);
        Assert.Equal(TextDataViewType.Instance, labelCol.Value.Type);

        var probCol = result.Schema.GetColumnOrNull("Probability");
        Assert.NotNull(probCol);
        Assert.IsType<VectorDataViewType>(probCol.Value.Type);
        Assert.Equal(NumberDataViewType.Single, ((VectorDataViewType)probCol.Value.Type).ItemType);

        // Source column is preserved
        Assert.NotNull(result.Schema.GetColumnOrNull("Image"));
    }

    [Fact]
    public void Classification_Transform_CanShuffleIsFalse()
    {
        if (!ModelExists) return;

        var options = new OnnxImageClassificationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet
        };
        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        var image = CreateTestImage(224, 224);
        _disposables.Add(image);

        var result = transformer.Transform(new SingleImageDataView(image));

        Assert.False(result.CanShuffle);
    }

    [Fact]
    public void Classification_Transform_GetRowCountReturnsNull()
    {
        if (!ModelExists) return;

        var options = new OnnxImageClassificationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet
        };
        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        var image = CreateTestImage(224, 224);
        _disposables.Add(image);

        var result = transformer.Transform(new SingleImageDataView(image));

        Assert.Null(result.GetRowCount());
    }

    [Fact]
    public void Classification_Transform_GetRowCursorSetReturnsSingleCursor()
    {
        if (!ModelExists) return;

        var options = new OnnxImageClassificationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet
        };
        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        var image = CreateTestImage(224, 224);
        _disposables.Add(image);

        var result = transformer.Transform(new SingleImageDataView(image));
        var cursors = result.GetRowCursorSet(result.Schema, 4);

        Assert.Single(cursors);
        cursors[0].Dispose();
    }

    [Fact]
    public void Classification_Cursor_IteratesRows()
    {
        if (!ModelExists) return;

        var labels = File.ReadAllLines(LabelsPath);
        var options = new OnnxImageClassificationOptions
        {
            ModelPath = ModelPath,
            Labels = labels,
            PreprocessorConfig = PreprocessorConfig.ImageNet,
            TopK = 5
        };
        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        var img1 = CreateTestImage(224, 224);
        _disposables.Add(img1);
        var img2 = CreateTestImage(224, 224);
        _disposables.Add(img2);

        var source = new TwoImageDataView(img1, img2);
        var result = transformer.Transform(source);

        int rowCount = 0;
        using var cursor = result.GetRowCursor(result.Schema);
        var labelCol = result.Schema["PredictedLabel"];
        var probCol = result.Schema["Probability"];

        var labelGetter = cursor.GetGetter<ReadOnlyMemory<char>>(labelCol);
        var probGetter = cursor.GetGetter<VBuffer<float>>(probCol);

        while (cursor.MoveNext())
        {
            ReadOnlyMemory<char> label = default;
            VBuffer<float> probs = default;
            labelGetter(ref label);
            probGetter(ref probs);

            Assert.False(label.IsEmpty);
            Assert.True(probs.Length > 0);
            rowCount++;
        }

        Assert.Equal(2, rowCount);
    }

    [Fact]
    public void Classification_Cursor_SourceColumnsPassthrough()
    {
        if (!ModelExists) return;

        var options = new OnnxImageClassificationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet,
            TopK = 3
        };
        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        var image = CreateTestImage(224, 224);
        _disposables.Add(image);

        var source = new SingleImageDataView(image);
        var result = transformer.Transform(source);

        using var cursor = result.GetRowCursor(result.Schema);
        Assert.True(cursor.MoveNext());

        // Source Image column should be accessible via passthrough
        var imageCol = result.Schema["Image"];
        var imageGetter = cursor.GetGetter<MLImage>(imageCol);
        MLImage retrieved = null!;
        imageGetter(ref retrieved);
        Assert.NotNull(retrieved);
    }

    [Fact]
    public void Classification_GetOutputSchema_MatchesTransformSchema()
    {
        if (!ModelExists) return;

        var options = new OnnxImageClassificationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet
        };
        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        var image = CreateTestImage(224, 224);
        _disposables.Add(image);

        var source = new SingleImageDataView(image);
        var outputSchema = transformer.GetOutputSchema(source.Schema);
        var transformedSchema = transformer.Transform(source).Schema;

        Assert.Equal(outputSchema.Count, transformedSchema.Count);
        for (int i = 0; i < outputSchema.Count; i++)
        {
            Assert.Equal(outputSchema[i].Name, transformedSchema[i].Name);
            Assert.Equal(outputSchema[i].Type, transformedSchema[i].Type);
        }
    }

    [Fact]
    public void Classification_Cursor_PositionAdvances()
    {
        if (!ModelExists) return;

        var options = new OnnxImageClassificationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet,
            TopK = 1
        };
        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        var img1 = CreateTestImage(224, 224);
        _disposables.Add(img1);
        var img2 = CreateTestImage(224, 224);
        _disposables.Add(img2);

        var result = transformer.Transform(new TwoImageDataView(img1, img2));

        using var cursor = result.GetRowCursor(result.Schema);
        Assert.Equal(-1, cursor.Position);

        Assert.True(cursor.MoveNext());
        Assert.Equal(0, cursor.Position);

        Assert.True(cursor.MoveNext());
        Assert.Equal(1, cursor.Position);

        Assert.False(cursor.MoveNext());
    }

    [Fact]
    public void Classification_CustomColumnNames_ReflectedInSchema()
    {
        if (!ModelExists) return;

        var options = new OnnxImageClassificationOptions
        {
            ModelPath = ModelPath,
            PreprocessorConfig = PreprocessorConfig.ImageNet,
            PredictedLabelColumnName = "MyLabel",
            ProbabilityColumnName = "MyProbs"
        };
        var transformer = new OnnxImageClassificationTransformer(options);
        _disposables.Add(transformer);

        var image = CreateTestImage(224, 224);
        _disposables.Add(image);

        var result = transformer.Transform(new SingleImageDataView(image));

        Assert.NotNull(result.Schema.GetColumnOrNull("MyLabel"));
        Assert.NotNull(result.Schema.GetColumnOrNull("MyProbs"));
        Assert.Null(result.Schema.GetColumnOrNull("PredictedLabel"));
        Assert.Null(result.Schema.GetColumnOrNull("Probability"));
    }

    // --- Helpers ---

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
    /// Minimal IDataView that serves a single MLImage row.
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
            => new ImageCursor(this, [_image]);

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
            => [GetRowCursor(columnsNeeded, rand)];

        private sealed class ImageCursor : DataViewRowCursor
        {
            private readonly SingleImageDataView _parent;
            private readonly MLImage[] _images;
            private long _position = -1;

            public ImageCursor(SingleImageDataView parent, MLImage[] images)
            {
                _parent = parent;
                _images = images;
            }

            public override DataViewSchema Schema => _parent._schema;
            public override long Position => _position;
            public override long Batch => 0;
            public override bool IsColumnActive(DataViewSchema.Column column) => true;
            public override bool MoveNext() => ++_position < _images.Length;
            public override ValueGetter<DataViewRowId> GetIdGetter() =>
                (ref DataViewRowId id) => id = new DataViewRowId((ulong)_position, 0);

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                ValueGetter<MLImage> getter = (ref MLImage value) => value = _images[_position];
                return (ValueGetter<TValue>)(Delegate)getter;
            }
        }
    }

    /// <summary>
    /// Minimal IDataView that serves two MLImage rows.
    /// </summary>
    private sealed class TwoImageDataView : IDataView
    {
        private readonly MLImage[] _images;
        private readonly DataViewSchema _schema;

        public TwoImageDataView(MLImage img1, MLImage img2)
        {
            _images = [img1, img2];
            var builder = new DataViewSchema.Builder();
            builder.AddColumn("Image", NumberDataViewType.Single);
            _schema = builder.ToSchema();
        }

        public DataViewSchema Schema => _schema;
        public bool CanShuffle => false;
        public long? GetRowCount() => 2;

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
            => new ImageCursor(this);

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
            => [GetRowCursor(columnsNeeded, rand)];

        private sealed class ImageCursor : DataViewRowCursor
        {
            private readonly TwoImageDataView _parent;
            private long _position = -1;

            public ImageCursor(TwoImageDataView parent) => _parent = parent;
            public override DataViewSchema Schema => _parent._schema;
            public override long Position => _position;
            public override long Batch => 0;
            public override bool IsColumnActive(DataViewSchema.Column column) => true;
            public override bool MoveNext() => ++_position < _parent._images.Length;
            public override ValueGetter<DataViewRowId> GetIdGetter() =>
                (ref DataViewRowId id) => id = new DataViewRowId((ulong)_position, 0);

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                ValueGetter<MLImage> getter = (ref MLImage value) => value = _parent._images[_position];
                return (ValueGetter<TValue>)(Delegate)getter;
            }
        }
    }
}

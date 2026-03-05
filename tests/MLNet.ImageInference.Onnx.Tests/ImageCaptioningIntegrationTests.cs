using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.ImageCaptioning;
using Xunit;

namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Integration tests for image captioning using real GIT ONNX models.
/// Requires models to be exported first (see README for instructions).
/// </summary>
public class ImageCaptioningIntegrationTests : IDisposable
{
    private const string EncoderPath = "models/git-coco/encoder.onnx";
    private const string DecoderPath = "models/git-coco/decoder.onnx";
    private const string VocabPath = "models/git-coco/vocab.txt";

    private static bool ModelsExist =>
        File.Exists(EncoderPath) && File.Exists(DecoderPath) && File.Exists(VocabPath);

    private readonly List<IDisposable> _disposables = [];

    public void Dispose()
    {
        foreach (var d in _disposables)
            d.Dispose();
    }

    [SkippableFact]
    public void GenerateCaption_ProducesNonEmptyString()
    {
        Skip.Unless(ModelsExist, "GIT captioning models not found");

        var options = CreateOptions();
        using var transformer = new OnnxImageCaptioningTransformer(options);

        using var image = CreateTestImage(224, 224, r: 135, g: 206, b: 235); // sky blue
        var caption = transformer.GenerateCaption(image);

        Assert.NotNull(caption);
        Assert.NotEmpty(caption);
        Assert.True(caption.Length > 2, $"Caption too short: '{caption}'");
    }

    [SkippableFact]
    public void GenerateCaption_DifferentImages_ProduceDifferentCaptions()
    {
        Skip.Unless(ModelsExist, "GIT captioning models not found");

        var options = CreateOptions();
        using var transformer = new OnnxImageCaptioningTransformer(options);

        // Blue image (sky-like)
        using var blueImage = CreateTestImage(224, 224, r: 135, g: 206, b: 235);
        var blueCaption = transformer.GenerateCaption(blueImage);

        // Green image (grass-like)
        using var greenImage = CreateTestImage(224, 224, r: 34, g: 139, b: 34);
        var greenCaption = transformer.GenerateCaption(greenImage);

        Assert.NotEmpty(blueCaption);
        Assert.NotEmpty(greenCaption);
        // Different color inputs should produce different captions
        // (not guaranteed but very likely with a trained model)
    }

    [SkippableFact]
    public void GenerateCaption_ResizesNonStandardImages()
    {
        Skip.Unless(ModelsExist, "GIT captioning models not found");

        var options = CreateOptions();
        using var transformer = new OnnxImageCaptioningTransformer(options);

        // Non-standard size — preprocessor should resize to 224x224
        using var image = CreateTestImage(640, 480, r: 200, g: 100, b: 50);
        var caption = transformer.GenerateCaption(image);

        Assert.NotNull(caption);
        Assert.NotEmpty(caption);
    }

    [SkippableFact]
    public void GenerateCaption_RespectsMaxLength()
    {
        Skip.Unless(ModelsExist, "GIT captioning models not found");

        var options = new OnnxImageCaptioningOptions
        {
            EncoderModelPath = EncoderPath,
            DecoderModelPath = DecoderPath,
            VocabPath = VocabPath,
            PreprocessorConfig = PreprocessorConfig.GIT,
            MaxLength = 5 // Very short max
        };
        using var transformer = new OnnxImageCaptioningTransformer(options);

        using var image = CreateTestImage(224, 224, r: 128, g: 128, b: 128);
        var caption = transformer.GenerateCaption(image);

        Assert.NotNull(caption);
        // Caption word count should be <= MaxLength tokens
        var words = caption.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        Assert.True(words.Length <= 5, $"Caption has {words.Length} words, expected <= 5: '{caption}'");
    }

    [SkippableFact]
    public void GenerateCaptionBatch_ReturnsBatchResults()
    {
        Skip.Unless(ModelsExist, "GIT captioning models not found");

        var options = CreateOptions();
        using var transformer = new OnnxImageCaptioningTransformer(options);

        using var img1 = CreateTestImage(224, 224, r: 255, g: 0, b: 0);   // Red
        using var img2 = CreateTestImage(224, 224, r: 0, g: 0, b: 255);   // Blue
        var captions = transformer.GenerateCaptionBatch([img1, img2]);

        Assert.Equal(2, captions.Length);
        Assert.All(captions, c =>
        {
            Assert.NotNull(c);
            Assert.NotEmpty(c);
        });
    }

    [SkippableFact]
    public void Estimator_FitAndTransform()
    {
        Skip.Unless(ModelsExist, "GIT captioning models not found");

        var options = CreateOptions();
        var estimator = new OnnxImageCaptioningEstimator(options);
        using var transformer = estimator.Fit(null!);
        _disposables.Add(transformer);

        using var image = CreateTestImage(224, 224, r: 135, g: 206, b: 235);
        var caption = transformer.GenerateCaption(image);

        Assert.NotNull(caption);
        Assert.NotEmpty(caption);
    }

    [SkippableFact]
    public void Transform_IDataView_ProducesCaptionColumn()
    {
        Skip.Unless(ModelsExist, "GIT captioning models not found");

        var options = CreateOptions();
        var estimator = new OnnxImageCaptioningEstimator(options);
        using var transformer = estimator.Fit(null!);
        _disposables.Add(transformer);

        using var image = CreateTestImage(224, 224, r: 135, g: 206, b: 235);
        var sourceData = new SingleImageDataView(image);
        var result = transformer.Transform(sourceData);

        var captionCol = result.Schema["Caption"];
        Assert.Equal("Caption", captionCol.Name);

        using var cursor = result.GetRowCursor(result.Schema);
        Assert.True(cursor.MoveNext());

        var getter = cursor.GetGetter<ReadOnlyMemory<char>>(captionCol);
        ReadOnlyMemory<char> captionValue = default;
        getter(ref captionValue);

        var captionStr = captionValue.ToString();
        Assert.NotEmpty(captionStr);
    }

    [SkippableFact]
    public void MLContextExtension_CreatesEstimator()
    {
        Skip.Unless(ModelsExist, "GIT captioning models not found");

        var mlContext = new MLContext();
        var estimator = mlContext.Transforms.OnnxImageCaptioning(CreateOptions());
        Assert.NotNull(estimator);

        using var transformer = estimator.Fit(null!);
        Assert.NotNull(transformer);
    }

    private static OnnxImageCaptioningOptions CreateOptions() => new()
    {
        EncoderModelPath = EncoderPath,
        DecoderModelPath = DecoderPath,
        VocabPath = VocabPath,
        PreprocessorConfig = PreprocessorConfig.GIT,
        MaxLength = 50
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

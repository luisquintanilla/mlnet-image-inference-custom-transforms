using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.SegmentAnything;

/// <summary>
/// Transformer for SAM2 (Segment Anything Model v2) prompt-based image segmentation.
/// Runs a two-stage pipeline: image encoder → prompt decoder → segmentation masks.
/// The encoder is run once per image; the decoder can be called multiple times with different prompts.
/// </summary>
public sealed class OnnxSegmentAnythingTransformer : ITransformer, IDisposable
{
    private readonly OnnxSegmentAnythingOptions _options;
    private readonly ImagePreprocessingTransformer _preprocessor;
    private readonly OnnxSessionPool _encoderPool;
    private readonly OnnxSessionPool _decoderPool;
    private bool _disposed;

    public OnnxSegmentAnythingTransformer(OnnxSegmentAnythingOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _preprocessor = new ImagePreprocessingTransformer(new ImagePreprocessingOptions
        {
            InputColumnName = options.InputColumnName,
            PreprocessorConfig = options.PreprocessorConfig
        });
        _encoderPool = new OnnxSessionPool(options.EncoderModelPath);
        _decoderPool = new OnnxSessionPool(options.DecoderModelPath);
    }

    /// <summary>
    /// Encode an image to produce image embeddings that can be reused with multiple prompts.
    /// </summary>
    public SegmentAnythingImageEmbedding EncodeImage(MLImage image, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ObjectDisposedException.ThrowIf(_disposed, this);

        int originalWidth = image.Width;
        int originalHeight = image.Height;

        var preprocessed = _preprocessor.Preprocess(image);
        int h = _options.PreprocessorConfig.ImageSize.Height;
        int w = _options.PreprocessorConfig.ImageSize.Width;
        var inputTensor = new DenseTensor<float>(preprocessed, [1, 3, h, w]);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image", inputTensor)
        };

        using var results = _encoderPool.Session.Run(inputs);
        var resultList = results.ToList();

        // Copy outputs to owned arrays (results are disposed after this block)
        var highRes0 = CopyTensor(resultList[0].AsTensor<float>(), [1, 32, 256, 256]);
        var highRes1 = CopyTensor(resultList[1].AsTensor<float>(), [1, 64, 128, 128]);
        var imageEmbed = CopyTensor(resultList[2].AsTensor<float>(), [1, 256, 64, 64]);

        return new SegmentAnythingImageEmbedding(highRes0, highRes1, imageEmbed, originalWidth, originalHeight);
    }

    /// <summary>
    /// Segment an image with a prompt using cached image embeddings.
    /// </summary>
    public SegmentAnythingResult Segment(SegmentAnythingImageEmbedding embedding, SegmentAnythingPrompt prompt, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ObjectDisposedException.ThrowIf(_disposed, this);

        int numPoints = prompt.NumPoints;

        // Build point_coords [1, numPoints, 2]
        var pointCoords = new DenseTensor<float>([1, numPoints, 2]);
        for (int i = 0; i < numPoints; i++)
        {
            pointCoords[0, i, 0] = prompt.PointCoords[i, 0];
            pointCoords[0, i, 1] = prompt.PointCoords[i, 1];
        }

        // Build point_labels [1, numPoints]
        var pointLabels = new DenseTensor<float>([1, numPoints]);
        for (int i = 0; i < numPoints; i++)
            pointLabels[0, i] = prompt.PointLabels[i];

        // Build mask_input [1, 1, 256, 256] and has_mask_input [1]
        var maskInput = new DenseTensor<float>([1, 1, 256, 256]);
        var hasMaskInput = new DenseTensor<float>([1]);
        if (prompt.PreviousMask != null)
        {
            hasMaskInput[0] = 1f;
            for (int y = 0; y < 256; y++)
                for (int x = 0; x < 256; x++)
                    maskInput[0, 0, y, x] = prompt.PreviousMask[y, x];
        }

        var origImSize = new DenseTensor<int>(new[] { embedding.OriginalHeight, embedding.OriginalWidth }, [2]);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image_embed", embedding.ImageEmbed),
            NamedOnnxValue.CreateFromTensor("high_res_feats_0", embedding.HighResFeats0),
            NamedOnnxValue.CreateFromTensor("high_res_feats_1", embedding.HighResFeats1),
            NamedOnnxValue.CreateFromTensor("point_coords", pointCoords),
            NamedOnnxValue.CreateFromTensor("point_labels", pointLabels),
            NamedOnnxValue.CreateFromTensor("mask_input", maskInput),
            NamedOnnxValue.CreateFromTensor("has_mask_input", hasMaskInput),
            NamedOnnxValue.CreateFromTensor("orig_im_size", origImSize)
        };

        using var results = _decoderPool.Session.Run(inputs);
        var resultList = results.ToList();

        var masksTensor = resultList[0].AsTensor<float>();
        var iouTensor = resultList[1].AsTensor<float>();

        int numMasks = (int)masksTensor.Dimensions[1];
        int maskH = (int)masksTensor.Dimensions[2];
        int maskW = (int)masksTensor.Dimensions[3];

        var masks = new float[numMasks][];
        for (int m = 0; m < numMasks; m++)
        {
            masks[m] = new float[maskH * maskW];
            for (int y = 0; y < maskH; y++)
                for (int x = 0; x < maskW; x++)
                    masks[m][y * maskW + x] = masksTensor[0, m, y, x] > _options.MaskThreshold ? 1f : 0f;
        }

        var iouPredictions = new float[numMasks];
        for (int m = 0; m < numMasks; m++)
            iouPredictions[m] = iouTensor[0, m];

        return new SegmentAnythingResult(masks, iouPredictions, maskW, maskH);
    }

    /// <summary>
    /// Convenience method: encode image and segment with a single prompt in one call.
    /// </summary>
    public SegmentAnythingResult Segment(MLImage image, SegmentAnythingPrompt prompt, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var embedding = EncodeImage(image, cancellationToken);
        return Segment(embedding, prompt, cancellationToken);
    }

    /// <summary>
    /// Segment using the center point of the image as a foreground prompt.
    /// Useful for automatic single-object segmentation.
    /// </summary>
    public SegmentAnythingResult SegmentCenter(MLImage image, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var prompt = SegmentAnythingPrompt.FromPoint(image.Width / 2f, image.Height / 2f);
        return Segment(image, prompt, cancellationToken);
    }

    private static DenseTensor<float> CopyTensor(Tensor<float> source, int[] dims)
    {
        var copy = new DenseTensor<float>(dims);
        source.ToArray().CopyTo(copy.Buffer.Span);
        return copy;
    }

    internal OnnxSegmentAnythingOptions Options => _options;

    public bool IsRowToRowMapper => false;

    public IDataView Transform(IDataView input)
        => new SegmentAnythingDataView(input, this);

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        for (int i = 0; i < inputSchema.Count; i++)
            builder.AddColumn(inputSchema[i].Name, inputSchema[i].Type, inputSchema[i].Annotations);
        builder.AddColumn(_options.OutputColumnName, new VectorDataViewType(NumberDataViewType.Single));
        builder.AddColumn(_options.OutputColumnName + "_Width", NumberDataViewType.Int32);
        builder.AddColumn(_options.OutputColumnName + "_Height", NumberDataViewType.Int32);
        builder.AddColumn(_options.OutputColumnName + "_IoU", NumberDataViewType.Single);
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new InvalidOperationException("This transformer does not support row-to-row mapping.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException("Use transformer-specific save/load instead of mlContext.Model.Save().");

    public void Dispose()
    {
        if (!_disposed)
        {
            _encoderPool?.Dispose();
            _decoderPool?.Dispose();
            _disposed = true;
        }
    }
}

/// <summary>
/// Cached image embeddings from the SAM2 encoder.
/// Allows running the decoder multiple times with different prompts without re-encoding.
/// </summary>
public sealed class SegmentAnythingImageEmbedding
{
    public DenseTensor<float> HighResFeats0 { get; }
    public DenseTensor<float> HighResFeats1 { get; }
    public DenseTensor<float> ImageEmbed { get; }
    public int OriginalWidth { get; }
    public int OriginalHeight { get; }

    public SegmentAnythingImageEmbedding(
        DenseTensor<float> highResFeats0,
        DenseTensor<float> highResFeats1,
        DenseTensor<float> imageEmbed,
        int originalWidth,
        int originalHeight)
    {
        HighResFeats0 = highResFeats0;
        HighResFeats1 = highResFeats1;
        ImageEmbed = imageEmbed;
        OriginalWidth = originalWidth;
        OriginalHeight = originalHeight;
    }
}

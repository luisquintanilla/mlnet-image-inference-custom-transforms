using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Segmentation;

/// <summary>
/// Transformer that performs image segmentation: MLImage → preprocessed tensor → ONNX → argmax → SegmentationMask.
/// Composes ImagePreprocessingTransformer + OnnxImageScoringTransformer + argmax post-processing.
/// </summary>
public sealed class OnnxImageSegmentationTransformer : ITransformer, IDisposable
{
    private readonly OnnxImageSegmentationOptions _options;
    private readonly ImagePreprocessingTransformer _preprocessor;
    private readonly OnnxImageScoringTransformer _scorer;

    public bool IsRowToRowMapper => true;

    public OnnxImageSegmentationTransformer(OnnxImageSegmentationOptions options)
        : this(options,
              new ImagePreprocessingTransformer(new ImagePreprocessingOptions
              {
                  InputColumnName = options.InputColumnName,
                  PreprocessorConfig = options.PreprocessorConfig
              }),
              new OnnxImageScoringTransformer(new OnnxImageScoringOptions
              {
                  ModelPath = options.ModelPath,
                  ImageHeight = options.PreprocessorConfig.ImageSize.Height,
                  ImageWidth = options.PreprocessorConfig.ImageSize.Width,
                  BatchSize = options.BatchSize
              }))
    {
    }

    internal OnnxImageSegmentationTransformer(
        OnnxImageSegmentationOptions options,
        ImagePreprocessingTransformer preprocessor,
        OnnxImageScoringTransformer scorer)
    {
        _options = options;
        _preprocessor = preprocessor;
        _scorer = scorer;
    }

    /// <summary>
    /// Segment a single image and return the segmentation mask.
    /// </summary>
    public SegmentationMask Segment(MLImage image)
    {
        // Stage 1: Preprocess
        var tensor = _preprocessor.Preprocess(image);

        // Stage 2: Score (with dimensions for accurate output shape)
        var (output, dims) = _scorer.ScoreWithDimensions(tensor);

        // Stage 3: Post-process (argmax + optional resize)
        // Actual output shape: [1, numClasses, outH, outW]
        int numClasses = dims[1];
        int outH = dims[2];
        int outW = dims[3];

        int? originalWidth = _options.ResizeToOriginal ? image.Width : null;
        int? originalHeight = _options.ResizeToOriginal ? image.Height : null;

        return MaskPostProcessor.Apply(output, numClasses, outH, outW, originalWidth, originalHeight, _options.Labels);
    }

    /// <summary>
    /// Segments a batch of images. Uses true tensor batching if the model supports dynamic batch,
    /// otherwise loops individual inference calls.
    /// </summary>
    public SegmentationMask[] SegmentBatch(IReadOnlyList<MLImage> images)
    {
        if (images == null || images.Count == 0)
            return Array.Empty<SegmentationMask>();

        if (_scorer.IsBatchDynamic)
        {
            return SegmentBatchDynamic(images);
        }
        else
        {
            var results = new SegmentationMask[images.Count];
            for (int i = 0; i < images.Count; i++)
                results[i] = Segment(images[i]);
            return results;
        }
    }

    private SegmentationMask[] SegmentBatchDynamic(IReadOnlyList<MLImage> images)
    {
        int n = images.Count;

        // Stage 1: Batch preprocess
        var batchTensor = _preprocessor.PreprocessBatch(images);

        // Stage 2: Batch score (with dimensions for accurate output shape)
        var (output, _, dims) = _scorer.ScoreBatchWithDimensions(batchTensor, n);

        // Actual output shape: [N, numClasses, outH, outW]
        int numClasses = dims[1];
        int outH = dims[2];
        int outW = dims[3];

        int outputPerImage = output.Length / n;
        var batchResults = new SegmentationMask[n];

        for (int i = 0; i < n; i++)
        {
            // Stage 3: Post-process each image's output
            var imageOutput = output.AsSpan(i * outputPerImage, outputPerImage).ToArray();
            int? origW = _options.ResizeToOriginal ? images[i].Width : null;
            int? origH = _options.ResizeToOriginal ? images[i].Height : null;
            batchResults[i] = MaskPostProcessor.Apply(imageOutput, numClasses, outH, outW, origW, origH, _options.Labels);
        }

        return batchResults;
    }

    internal OnnxImageSegmentationOptions Options => _options;
    internal ImagePreprocessingTransformer Preprocessor => _preprocessor;
    internal OnnxImageScoringTransformer Scorer => _scorer;

    public IDataView Transform(IDataView input)
    {
        return new SegmentationDataView(input, this);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        for (int i = 0; i < inputSchema.Count; i++)
            builder.AddColumn(inputSchema[i].Name, inputSchema[i].Type, inputSchema[i].Annotations);
        builder.AddColumn(_options.OutputColumnName, new VectorDataViewType(NumberDataViewType.Int32));
        builder.AddColumn(_options.OutputColumnName + "_Width", NumberDataViewType.Int32);
        builder.AddColumn(_options.OutputColumnName + "_Height", NumberDataViewType.Int32);
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new InvalidOperationException("This transformer does not support row-to-row mapping.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException("Use transformer-specific save/load instead of mlContext.Model.Save().");

    public void Dispose()
    {
        _scorer?.Dispose();
    }
}

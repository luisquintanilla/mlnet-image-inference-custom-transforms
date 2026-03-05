using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.DepthEstimation;

/// <summary>
/// Transformer that performs monocular depth estimation: MLImage → preprocess → ONNX → normalize → DepthMap.
/// Composes ImagePreprocessingTransformer + OnnxImageScoringTransformer + depth normalization.
/// </summary>
public sealed class OnnxImageDepthEstimationTransformer : ITransformer, IDisposable
{
    private readonly OnnxImageDepthEstimationOptions _options;
    private readonly ImagePreprocessingTransformer _preprocessor;
    private readonly OnnxImageScoringTransformer _scorer;

    public bool IsRowToRowMapper => true;

    public OnnxImageDepthEstimationTransformer(OnnxImageDepthEstimationOptions options)
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

    internal OnnxImageDepthEstimationTransformer(
        OnnxImageDepthEstimationOptions options,
        ImagePreprocessingTransformer preprocessor,
        OnnxImageScoringTransformer scorer)
    {
        _options = options;
        _preprocessor = preprocessor;
        _scorer = scorer;
    }

    /// <summary>
    /// Estimate depth for a single image.
    /// </summary>
    public DepthMap Estimate(MLImage image)
    {
        var tensor = _preprocessor.Preprocess(image);
        var (output, dims) = _scorer.ScoreWithDimensions(tensor);

        // DPT output: [1, H, W] or [1, 1, H, W]
        int outH, outW;
        if (dims.Length == 3)
        {
            outH = dims[1];
            outW = dims[2];
        }
        else
        {
            outH = dims[2];
            outW = dims[3];
        }

        int? origW = _options.ResizeToOriginal ? image.Width : null;
        int? origH = _options.ResizeToOriginal ? image.Height : null;

        return DepthMapPostProcessor.Apply(output, outH, outW, origW, origH);
    }

    /// <summary>
    /// Estimate depth for a batch of images.
    /// </summary>
    public DepthMap[] EstimateBatch(IReadOnlyList<MLImage> images)
    {
        if (images == null || images.Count == 0)
            return Array.Empty<DepthMap>();

        if (_scorer.IsBatchDynamic)
        {
            return EstimateBatchDynamic(images);
        }
        else
        {
            var results = new DepthMap[images.Count];
            for (int i = 0; i < images.Count; i++)
                results[i] = Estimate(images[i]);
            return results;
        }
    }

    private DepthMap[] EstimateBatchDynamic(IReadOnlyList<MLImage> images)
    {
        int n = images.Count;
        var batchTensor = _preprocessor.PreprocessBatch(images);
        var (output, _, dims) = _scorer.ScoreBatchWithDimensions(batchTensor, n);

        int outH, outW;
        if (dims.Length == 3)
        {
            outH = dims[1];
            outW = dims[2];
        }
        else
        {
            outH = dims[2];
            outW = dims[3];
        }

        int outputPerImage = output.Length / n;
        var batchResults = new DepthMap[n];

        for (int i = 0; i < n; i++)
        {
            var imageOutput = output.AsSpan(i * outputPerImage, outputPerImage).ToArray();
            int? origW = _options.ResizeToOriginal ? images[i].Width : null;
            int? origH = _options.ResizeToOriginal ? images[i].Height : null;
            batchResults[i] = DepthMapPostProcessor.Apply(imageOutput, outH, outW, origW, origH);
        }

        return batchResults;
    }

    internal OnnxImageDepthEstimationOptions Options => _options;
    internal ImagePreprocessingTransformer Preprocessor => _preprocessor;
    internal OnnxImageScoringTransformer Scorer => _scorer;

    public IDataView Transform(IDataView input)
        => new DepthEstimationDataView(input, this);

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        for (int i = 0; i < inputSchema.Count; i++)
            builder.AddColumn(inputSchema[i].Name, inputSchema[i].Type, inputSchema[i].Annotations);
        builder.AddColumn(_options.OutputColumnName, new VectorDataViewType(NumberDataViewType.Single));
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

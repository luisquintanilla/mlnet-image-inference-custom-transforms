using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Segmentation;

/// <summary>
/// Transformer that performs image segmentation: MLImage → preprocessed tensor → ONNX → argmax → SegmentationMask.
/// </summary>
public sealed class OnnxImageSegmentationTransformer : ITransformer, IDisposable
{
    private readonly OnnxImageSegmentationOptions _options;
    private readonly OnnxSessionPool _sessionPool;
    private readonly ModelMetadataDiscovery.ModelMetadata _metadata;

    public bool IsRowToRowMapper => false;

    public OnnxImageSegmentationTransformer(OnnxImageSegmentationOptions options)
    {
        _options = options;
        _sessionPool = new OnnxSessionPool(options.ModelPath);
        _metadata = ModelMetadataDiscovery.Discover(_sessionPool.Session);
    }

    /// <summary>
    /// Segment a single image and return the segmentation mask.
    /// </summary>
    public SegmentationMask Segment(MLImage image)
    {
        var tensor = HuggingFaceImagePreprocessor.Preprocess(image, _options.PreprocessorConfig);
        int height = _options.PreprocessorConfig.ImageSize.Height;
        int width = _options.PreprocessorConfig.ImageSize.Width;

        // Create ONNX input tensor [1, 3, H, W]
        var inputTensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(tensor, [1, 3, height, width]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_metadata.InputNames[0], inputTensor)
        };

        // Run inference
        using var results = _sessionPool.Session.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();

        // Determine output shape: [1, numClasses, outH, outW]
        var outputShape = _metadata.OutputShapes[0];
        int numClasses = (int)outputShape[1];
        int outH = outputShape.Length > 2 ? (int)outputShape[2] : height;
        int outW = outputShape.Length > 3 ? (int)outputShape[3] : width;

        // If output shape has dynamic dimensions, infer from output length
        if (outH <= 0 || outW <= 0)
        {
            outH = height;
            outW = width;
        }
        if (numClasses <= 0)
        {
            numClasses = output.Length / (outH * outW);
        }

        // Apply argmax post-processing with optional resize to original dimensions
        int? originalWidth = _options.ResizeToOriginal ? image.Width : null;
        int? originalHeight = _options.ResizeToOriginal ? image.Height : null;

        return MaskPostProcessor.Apply(output, numClasses, outH, outW, originalWidth, originalHeight, _options.Labels);
    }

    public IDataView Transform(IDataView input)
    {
        throw new NotImplementedException(
            "Full IDataView Transform is under development. " +
            "Use the Segment() method for single-image segmentation.");
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumn(_options.OutputColumnName, new VectorDataViewType(NumberDataViewType.Int32));
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new InvalidOperationException("This transformer does not support row-to-row mapping.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException("Use transformer-specific save/load instead of mlContext.Model.Save().");

    public void Dispose()
    {
        _sessionPool?.Dispose();
    }
}

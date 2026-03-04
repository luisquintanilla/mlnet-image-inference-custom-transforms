using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Detection;

/// <summary>
/// Transformer that performs object detection: MLImage → preprocessed tensor → ONNX → NMS → BoundingBox[].
/// </summary>
public sealed class OnnxObjectDetectionTransformer : ITransformer, IDisposable
{
    private readonly OnnxObjectDetectionOptions _options;
    private readonly OnnxSessionPool _sessionPool;
    private readonly ModelMetadataDiscovery.ModelMetadata _metadata;

    public bool IsRowToRowMapper => false;

    public OnnxObjectDetectionTransformer(OnnxObjectDetectionOptions options)
    {
        _options = options;
        _sessionPool = new OnnxSessionPool(options.ModelPath);
        _metadata = ModelMetadataDiscovery.Discover(_sessionPool.Session);
    }

    /// <summary>
    /// Detect objects in a single image and return bounding boxes.
    /// </summary>
    public BoundingBox[] Detect(MLImage image)
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

        // Determine dimensions from model output shape: [1, numClasses+4, numBoxes]
        var outputShape = _metadata.OutputShapes[0];
        int numClasses = (int)outputShape[1] - 4;
        int numBoxes = (int)outputShape[2];

        // Apply NMS post-processing
        var detections = NmsPostProcessor.Apply(
            output,
            numClasses,
            numBoxes,
            _options.ConfidenceThreshold,
            _options.IouThreshold,
            _options.Labels);

        // Apply MaxDetections if specified
        if (_options.MaxDetections.HasValue && _options.MaxDetections.Value < detections.Length)
        {
            detections = detections[.._options.MaxDetections.Value];
        }

        return detections;
    }

    public IDataView Transform(IDataView input)
    {
        // TODO: Implement full IDataView transform with DataView/Cursor pattern
        throw new NotImplementedException(
            "Full IDataView Transform is under development. " +
            "Use the Detect() method for single-image detection.");
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumn(_options.OutputColumnName, new VectorDataViewType(NumberDataViewType.Single));
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

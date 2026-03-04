using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Detection;

/// <summary>
/// Facade estimator that chains: image preprocessing → ONNX scoring → NMS object detection.
/// </summary>
public sealed class OnnxObjectDetectionEstimator : IEstimator<OnnxObjectDetectionTransformer>
{
    private readonly OnnxObjectDetectionOptions _options;

    public OnnxObjectDetectionEstimator(OnnxObjectDetectionOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        _options = options;
    }

    public OnnxObjectDetectionTransformer Fit(IDataView input)
    {
        return new OnnxObjectDetectionTransformer(_options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name);

        columns[_options.OutputColumnName] = SchemaShapeHelper.CreateColumn(
            _options.OutputColumnName,
            SchemaShape.Column.VectorKind.Vector,
            NumberDataViewType.Single,
            isKey: false);

        return new SchemaShape(columns.Values);
    }
}

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Segmentation;

/// <summary>
/// Facade estimator that chains: image preprocessing → ONNX scoring → argmax segmentation.
/// </summary>
public sealed class OnnxImageSegmentationEstimator : IEstimator<OnnxImageSegmentationTransformer>
{
    private readonly OnnxImageSegmentationOptions _options;

    public OnnxImageSegmentationEstimator(OnnxImageSegmentationOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        _options = options;
    }

    public OnnxImageSegmentationTransformer Fit(IDataView input)
    {
        return new OnnxImageSegmentationTransformer(_options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name);

        columns[_options.OutputColumnName] = SchemaShapeHelper.CreateColumn(
            _options.OutputColumnName,
            SchemaShape.Column.VectorKind.Vector,
            NumberDataViewType.Int32,
            isKey: false);

        return new SchemaShape(columns.Values);
    }
}

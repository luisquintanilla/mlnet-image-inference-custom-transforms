using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Classification;

/// <summary>
/// Facade estimator that chains: image preprocessing → ONNX scoring → softmax classification.
/// </summary>
public sealed class OnnxImageClassificationEstimator : IEstimator<OnnxImageClassificationTransformer>
{
    private readonly OnnxImageClassificationOptions _options;

    public OnnxImageClassificationEstimator(OnnxImageClassificationOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        _options = options;
    }

    public OnnxImageClassificationTransformer Fit(IDataView input)
    {
        return new OnnxImageClassificationTransformer(_options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name);

        columns[_options.PredictedLabelColumnName] = SchemaShapeHelper.CreateColumn(
            _options.PredictedLabelColumnName,
            SchemaShape.Column.VectorKind.Scalar,
            TextDataViewType.Instance,
            isKey: false);

        columns[_options.ProbabilityColumnName] = SchemaShapeHelper.CreateColumn(
            _options.ProbabilityColumnName,
            SchemaShape.Column.VectorKind.Vector,
            NumberDataViewType.Single,
            isKey: false);

        return new SchemaShape(columns.Values);
    }
}

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.ZeroShot;

/// <summary>
/// Facade estimator for zero-shot image classification using CLIP.
/// </summary>
public sealed class OnnxZeroShotImageClassificationEstimator : IEstimator<OnnxZeroShotImageClassificationTransformer>
{
    private readonly OnnxZeroShotImageClassificationOptions _options;

    public OnnxZeroShotImageClassificationEstimator(OnnxZeroShotImageClassificationOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        _options = options;
    }

    public OnnxZeroShotImageClassificationTransformer Fit(IDataView input)
    {
        return new OnnxZeroShotImageClassificationTransformer(_options);
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

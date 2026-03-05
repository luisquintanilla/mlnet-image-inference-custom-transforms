using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Classification;

/// <summary>
/// Facade estimator that chains: image preprocessing → ONNX scoring → softmax classification.
/// </summary>
public sealed class OnnxImageClassificationEstimator
    : OnnxImageEstimatorBase<OnnxImageClassificationTransformer, OnnxImageClassificationOptions>
{
    public OnnxImageClassificationEstimator(OnnxImageClassificationOptions options)
        : base(options) { }

    protected override OnnxImageClassificationTransformer CreateTransformer()
        => new(Options);

    protected override void ConfigureOutputSchema(IDictionary<string, SchemaShape.Column> columns)
    {
        columns[Options.PredictedLabelColumnName] = SchemaShapeHelper.CreateColumn(
            Options.PredictedLabelColumnName,
            SchemaShape.Column.VectorKind.Scalar,
            TextDataViewType.Instance,
            isKey: false);

        columns[Options.ProbabilityColumnName] = SchemaShapeHelper.CreateColumn(
            Options.ProbabilityColumnName,
            SchemaShape.Column.VectorKind.Vector,
            NumberDataViewType.Single,
            isKey: false);
    }
}

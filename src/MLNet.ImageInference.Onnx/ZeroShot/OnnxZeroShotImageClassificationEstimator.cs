using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.ZeroShot;

/// <summary>
/// Facade estimator for zero-shot image classification using CLIP.
/// </summary>
public sealed class OnnxZeroShotImageClassificationEstimator
    : OnnxImageEstimatorBase<OnnxZeroShotImageClassificationTransformer, OnnxZeroShotImageClassificationOptions>
{
    public OnnxZeroShotImageClassificationEstimator(OnnxZeroShotImageClassificationOptions options)
        : base(options) { }

    protected override OnnxZeroShotImageClassificationTransformer CreateTransformer()
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

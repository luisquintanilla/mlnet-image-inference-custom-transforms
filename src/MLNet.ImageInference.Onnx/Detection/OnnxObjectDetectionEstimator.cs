using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Detection;

/// <summary>
/// Facade estimator that chains: image preprocessing → ONNX scoring → NMS object detection.
/// </summary>
public sealed class OnnxObjectDetectionEstimator
    : OnnxImageEstimatorBase<OnnxObjectDetectionTransformer, OnnxObjectDetectionOptions>
{
    public OnnxObjectDetectionEstimator(OnnxObjectDetectionOptions options)
        : base(options) { }

    protected override OnnxObjectDetectionTransformer CreateTransformer()
        => new(Options);

    protected override void ConfigureOutputSchema(IDictionary<string, SchemaShape.Column> columns)
    {
        columns[Options.OutputColumnName] = SchemaShapeHelper.CreateColumn(
            Options.OutputColumnName,
            SchemaShape.Column.VectorKind.Vector,
            NumberDataViewType.Single,
            isKey: false);
    }
}

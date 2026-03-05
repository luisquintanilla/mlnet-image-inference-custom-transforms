using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Segmentation;

/// <summary>
/// Facade estimator that chains: image preprocessing → ONNX scoring → argmax segmentation.
/// </summary>
public sealed class OnnxImageSegmentationEstimator
    : OnnxImageEstimatorBase<OnnxImageSegmentationTransformer, OnnxImageSegmentationOptions>
{
    public OnnxImageSegmentationEstimator(OnnxImageSegmentationOptions options)
        : base(options) { }

    protected override OnnxImageSegmentationTransformer CreateTransformer()
        => new(Options);

    protected override void ConfigureOutputSchema(IDictionary<string, SchemaShape.Column> columns)
    {
        columns[Options.OutputColumnName] = SchemaShapeHelper.CreateColumn(
            Options.OutputColumnName,
            SchemaShape.Column.VectorKind.Vector,
            NumberDataViewType.Int32,
            isKey: false);
    }
}

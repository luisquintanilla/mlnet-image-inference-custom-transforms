using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.SegmentAnything;

/// <summary>
/// Facade estimator for SAM2: image preprocessing → encoder → decoder → segmentation mask.
/// In the IDataView pipeline, uses center-point prompting for automatic single-object segmentation.
/// </summary>
public sealed class OnnxSegmentAnythingEstimator
    : OnnxImageEstimatorBase<OnnxSegmentAnythingTransformer, OnnxSegmentAnythingOptions>
{
    public OnnxSegmentAnythingEstimator(OnnxSegmentAnythingOptions options)
        : base(options) { }

    protected override OnnxSegmentAnythingTransformer CreateTransformer()
    {
        return new OnnxSegmentAnythingTransformer(Options);
    }

    protected override void ConfigureOutputSchema(IDictionary<string, SchemaShape.Column> columns)
    {
        columns[Options.OutputColumnName] = SchemaShapeHelper.CreateColumn(
            Options.OutputColumnName,
            SchemaShape.Column.VectorKind.Vector,
            NumberDataViewType.Single,
            isKey: false);

        columns[Options.OutputColumnName + "_Width"] = SchemaShapeHelper.CreateColumn(
            Options.OutputColumnName + "_Width",
            SchemaShape.Column.VectorKind.Scalar,
            NumberDataViewType.Int32,
            isKey: false);

        columns[Options.OutputColumnName + "_Height"] = SchemaShapeHelper.CreateColumn(
            Options.OutputColumnName + "_Height",
            SchemaShape.Column.VectorKind.Scalar,
            NumberDataViewType.Int32,
            isKey: false);

        columns[Options.OutputColumnName + "_IoU"] = SchemaShapeHelper.CreateColumn(
            Options.OutputColumnName + "_IoU",
            SchemaShape.Column.VectorKind.Scalar,
            NumberDataViewType.Single,
            isKey: false);
    }
}

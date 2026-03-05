using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Detection;

/// <summary>
/// Custom IDataView that wraps a source IDataView and appends detection output columns
/// (DetectedObjects_Boxes, DetectedObjects_Count) by running ONNX inference lazily per row.
/// </summary>
internal sealed class DetectionDataView : OnnxImageDataViewBase
{
    private readonly OnnxObjectDetectionTransformer _transformer;

    public DetectionDataView(IDataView source, OnnxObjectDetectionTransformer transformer)
        : base(source, transformer.Options.InputColumnName, (builder, nextIndex) =>
        {
            builder.AddColumn(transformer.Options.OutputColumnName + "_Boxes", new VectorDataViewType(NumberDataViewType.Single));
            builder.AddColumn(transformer.Options.OutputColumnName + "_Count", NumberDataViewType.Int32);
        })
    {
        _transformer = transformer;
        BoxesIndex = source.Schema.Count;
        CountIndex = source.Schema.Count + 1;
    }

    protected override DataViewRowCursor CreateCursor(
        DataViewRowCursor sourceCursor,
        DataViewSchema.Column? inputCol)
    {
        return new DetectionCursor(this, sourceCursor, _transformer, inputCol);
    }

    internal int BoxesIndex { get; private set; }
    internal int CountIndex { get; private set; }
}

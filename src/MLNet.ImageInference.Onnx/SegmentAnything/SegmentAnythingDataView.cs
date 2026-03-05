using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.SegmentAnything;

/// <summary>
/// Custom IDataView that wraps a source and appends SAM2 mask output columns
/// by running segmentation inference lazily per row using center-point prompting.
/// </summary>
internal sealed class SegmentAnythingDataView : OnnxImageDataViewBase
{
    private readonly OnnxSegmentAnythingTransformer _transformer;

    public SegmentAnythingDataView(IDataView source, OnnxSegmentAnythingTransformer transformer)
        : base(source, transformer.Options.InputColumnName, (builder, nextIndex) =>
        {
            builder.AddColumn(transformer.Options.OutputColumnName,
                new VectorDataViewType(NumberDataViewType.Single));
            builder.AddColumn(transformer.Options.OutputColumnName + "_Width",
                NumberDataViewType.Int32);
            builder.AddColumn(transformer.Options.OutputColumnName + "_Height",
                NumberDataViewType.Int32);
            builder.AddColumn(transformer.Options.OutputColumnName + "_IoU",
                NumberDataViewType.Single);
        })
    {
        _transformer = transformer;
        int baseIndex = source.Schema.Count;
        MaskIndex = baseIndex;
        WidthIndex = baseIndex + 1;
        HeightIndex = baseIndex + 2;
        IoUIndex = baseIndex + 3;
    }

    protected override DataViewRowCursor CreateCursor(
        DataViewRowCursor sourceCursor,
        DataViewSchema.Column? inputCol)
    {
        return new SegmentAnythingCursor(this, sourceCursor, _transformer, inputCol);
    }

    internal int MaskIndex { get; }
    internal int WidthIndex { get; }
    internal int HeightIndex { get; }
    internal int IoUIndex { get; }
}

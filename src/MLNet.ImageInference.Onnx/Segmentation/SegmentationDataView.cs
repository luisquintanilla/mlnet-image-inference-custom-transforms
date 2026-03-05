using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Segmentation;

/// <summary>
/// Custom IDataView that wraps a source IDataView and appends segmentation output columns
/// (SegmentationMask, SegmentationMask_Width, SegmentationMask_Height) by running ONNX inference lazily per row.
/// </summary>
internal sealed class SegmentationDataView : OnnxImageDataViewBase
{
    private readonly OnnxImageSegmentationTransformer _transformer;

    public SegmentationDataView(IDataView source, OnnxImageSegmentationTransformer transformer)
        : base(source, transformer.Options.InputColumnName, (builder, nextIndex) =>
        {
            builder.AddColumn(transformer.Options.OutputColumnName, new VectorDataViewType(NumberDataViewType.Int32));
            builder.AddColumn(transformer.Options.OutputColumnName + "_Width", NumberDataViewType.Int32);
            builder.AddColumn(transformer.Options.OutputColumnName + "_Height", NumberDataViewType.Int32);
        })
    {
        _transformer = transformer;
        MaskIndex = source.Schema.Count;
        WidthIndex = source.Schema.Count + 1;
        HeightIndex = source.Schema.Count + 2;
    }

    protected override DataViewRowCursor CreateCursor(
        DataViewRowCursor sourceCursor,
        DataViewSchema.Column? inputCol)
    {
        return new SegmentationCursor(this, sourceCursor, _transformer, inputCol);
    }

    internal int MaskIndex { get; private set; }
    internal int WidthIndex { get; private set; }
    internal int HeightIndex { get; private set; }
}

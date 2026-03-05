using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Segmentation;

/// <summary>
/// Cursor that lazily performs image segmentation inference per row,
/// reading MLImage from the source and producing SegmentationMask, Width, and Height columns.
/// </summary>
internal sealed class SegmentationCursor : OnnxImageCursorBase<SegmentationMask>
{
    private readonly SegmentationDataView _parent;
    private readonly OnnxImageSegmentationTransformer _transformer;

    private int[] _classIds = [];
    private int _width;
    private int _height;

    public SegmentationCursor(
        SegmentationDataView parent,
        DataViewRowCursor sourceCursor,
        OnnxImageSegmentationTransformer transformer,
        DataViewSchema.Column? inputCol)
        : base(parent, sourceCursor, inputCol, transformer.Options.BatchSize)
    {
        _parent = parent;
        _transformer = transformer;
    }

    protected override bool IsOutputColumn(int columnIndex)
        => columnIndex == _parent.MaskIndex || columnIndex == _parent.WidthIndex || columnIndex == _parent.HeightIndex;

    protected override SegmentationMask CreateEmptyResult()
        => new();

    protected override void ExtractCurrentResult(SegmentationMask result)
    {
        _classIds = result.ClassIds;
        _width = result.Width;
        _height = result.Height;
    }

    protected override void RunBatchInference(List<MLImage> images)
    {
        var batchResults = _transformer.SegmentBatch(images);
        for (int i = 0; i < batchResults.Length; i++)
            BatchResults.Add(batchResults[i]);
    }

    protected override ValueGetter<TValue> CreateOutputGetter<TValue>(DataViewSchema.Column column)
    {
        if (column.Index == _parent.MaskIndex)
        {
            ValueGetter<VBuffer<int>> getter = (ref VBuffer<int> value) =>
                value = new VBuffer<int>(_classIds.Length, _classIds);
            return (ValueGetter<TValue>)(Delegate)getter;
        }

        if (column.Index == _parent.WidthIndex)
        {
            ValueGetter<int> getter = (ref int value) => value = _width;
            return (ValueGetter<TValue>)(Delegate)getter;
        }

        ValueGetter<int> heightGetter = (ref int value) => value = _height;
        return (ValueGetter<TValue>)(Delegate)heightGetter;
    }

    public override ValueGetter<DataViewRowId> GetIdGetter()
        => SourceCursor.GetIdGetter();
}

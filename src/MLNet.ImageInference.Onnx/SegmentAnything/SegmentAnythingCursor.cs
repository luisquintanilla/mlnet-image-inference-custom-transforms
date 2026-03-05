using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.SegmentAnything;

/// <summary>
/// Cursor that lazily performs SAM2 segmentation inference per row,
/// reading MLImage from the source and producing mask, width, height, and IoU columns.
/// Uses center-point prompting for automatic segmentation in the IDataView pipeline.
/// </summary>
internal sealed class SegmentAnythingCursor : OnnxImageCursorBase<SegmentAnythingResult>
{
    private readonly SegmentAnythingDataView _parent;
    private readonly OnnxSegmentAnythingTransformer _transformer;

    private VBuffer<float> _mask;
    private int _width;
    private int _height;
    private float _iou;

    public SegmentAnythingCursor(
        SegmentAnythingDataView parent,
        DataViewRowCursor sourceCursor,
        OnnxSegmentAnythingTransformer transformer,
        DataViewSchema.Column? inputCol)
        : base(parent, sourceCursor, inputCol, transformer.Options.BatchSize)
    {
        _parent = parent;
        _transformer = transformer;
    }

    protected override bool IsOutputColumn(int columnIndex)
        => columnIndex == _parent.MaskIndex
        || columnIndex == _parent.WidthIndex
        || columnIndex == _parent.HeightIndex
        || columnIndex == _parent.IoUIndex;

    protected override SegmentAnythingResult CreateEmptyResult()
        => new([], [], 0, 0);

    protected override void ExtractCurrentResult(SegmentAnythingResult result)
    {
        if (result.NumMasks > 0)
        {
            _mask = new VBuffer<float>(result.GetBestMask().Length, result.GetBestMask());
            _width = result.Width;
            _height = result.Height;
            _iou = result.GetBestIoU();
        }
        else
        {
            _mask = new VBuffer<float>(0, Array.Empty<float>());
            _width = 0;
            _height = 0;
            _iou = 0f;
        }
    }

    protected override void RunBatchInference(List<MLImage> images)
    {
        // SAM2 encoder is fixed batch=1, so process sequentially
        foreach (var image in images)
        {
            var result = _transformer.SegmentCenter(image);
            BatchResults.Add(result);
        }
    }

    protected override ValueGetter<TValue> CreateOutputGetter<TValue>(DataViewSchema.Column column)
    {
        if (column.Index == _parent.MaskIndex)
        {
            ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) => value = _mask;
            return (ValueGetter<TValue>)(Delegate)getter;
        }
        if (column.Index == _parent.WidthIndex)
        {
            ValueGetter<int> getter = (ref int value) => value = _width;
            return (ValueGetter<TValue>)(Delegate)getter;
        }
        if (column.Index == _parent.HeightIndex)
        {
            ValueGetter<int> getter = (ref int value) => value = _height;
            return (ValueGetter<TValue>)(Delegate)getter;
        }
        if (column.Index == _parent.IoUIndex)
        {
            ValueGetter<float> getter = (ref float value) => value = _iou;
            return (ValueGetter<TValue>)(Delegate)getter;
        }
        throw new InvalidOperationException($"Unexpected column index: {column.Index}");
    }

    public override ValueGetter<DataViewRowId> GetIdGetter()
        => SourceCursor.GetIdGetter();
}

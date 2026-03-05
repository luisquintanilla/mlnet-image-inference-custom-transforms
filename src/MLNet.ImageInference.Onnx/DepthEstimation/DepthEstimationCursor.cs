using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.DepthEstimation;

/// <summary>
/// Cursor that lazily performs depth estimation inference per row,
/// reading MLImage from the source and producing DepthMap, Width, and Height columns.
/// </summary>
internal sealed class DepthEstimationCursor : OnnxImageCursorBase<DepthMap>
{
    private readonly DepthEstimationDataView _parent;
    private readonly OnnxImageDepthEstimationTransformer _transformer;

    private float[] _values = [];
    private int _width;
    private int _height;

    public DepthEstimationCursor(
        DepthEstimationDataView parent,
        DataViewRowCursor sourceCursor,
        OnnxImageDepthEstimationTransformer transformer,
        DataViewSchema.Column? inputCol)
        : base(parent, sourceCursor, inputCol, transformer.Options.BatchSize)
    {
        _parent = parent;
        _transformer = transformer;
    }

    protected override bool IsOutputColumn(int columnIndex)
        => columnIndex == _parent.DepthIndex || columnIndex == _parent.WidthIndex || columnIndex == _parent.HeightIndex;

    protected override DepthMap CreateEmptyResult()
        => new();

    protected override void ExtractCurrentResult(DepthMap result)
    {
        _values = result.Values;
        _width = result.Width;
        _height = result.Height;
    }

    protected override void RunBatchInference(List<MLImage> images)
    {
        var batchResults = _transformer.EstimateBatch(images);
        for (int i = 0; i < batchResults.Length; i++)
            BatchResults.Add(batchResults[i]);
    }

    protected override ValueGetter<TValue> CreateOutputGetter<TValue>(DataViewSchema.Column column)
    {
        if (column.Index == _parent.DepthIndex)
        {
            ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
                value = new VBuffer<float>(_values.Length, _values);
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

using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.ImageInference.Onnx.Segmentation;

/// <summary>
/// Cursor that lazily performs image segmentation inference per row,
/// reading MLImage from the source and producing SegmentationMask, Width, and Height columns.
/// </summary>
internal sealed class SegmentationCursor : DataViewRowCursor
{
    private readonly SegmentationDataView _parent;
    private readonly DataViewRowCursor _sourceCursor;
    private readonly OnnxImageSegmentationTransformer _transformer;
    private readonly DataViewSchema.Column? _inputCol;

    private int[] _classIds = [];
    private int _width;
    private int _height;
    private bool _disposed;

    public SegmentationCursor(
        SegmentationDataView parent,
        DataViewRowCursor sourceCursor,
        OnnxImageSegmentationTransformer transformer,
        DataViewSchema.Column? inputCol)
    {
        _parent = parent;
        _sourceCursor = sourceCursor;
        _transformer = transformer;
        _inputCol = inputCol;
    }

    public override DataViewSchema Schema => _parent.Schema;

    public override long Position => _sourceCursor.Position;

    public override long Batch => _sourceCursor.Batch;

    public override bool IsColumnActive(DataViewSchema.Column column)
    {
        if (column.Index == _parent.MaskIndex || column.Index == _parent.WidthIndex || column.Index == _parent.HeightIndex)
            return true;
        if (column.Index < _sourceCursor.Schema.Count)
            return _sourceCursor.IsColumnActive(_sourceCursor.Schema[column.Index]);
        return false;
    }

    public override bool MoveNext()
    {
        if (!_sourceCursor.MoveNext())
            return false;

        if (_inputCol.HasValue)
        {
            MLImage image = null!;
            var getter = _sourceCursor.GetGetter<MLImage>(_inputCol.Value);
            getter(ref image);

            var mask = _transformer.Segment(image);
            _classIds = mask.ClassIds;
            _width = mask.Width;
            _height = mask.Height;
        }
        else
        {
            _classIds = [];
            _width = 0;
            _height = 0;
        }

        return true;
    }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
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

        if (column.Index == _parent.HeightIndex)
        {
            ValueGetter<int> getter = (ref int value) => value = _height;
            return (ValueGetter<TValue>)(Delegate)getter;
        }

        // Passthrough to source cursor
        if (column.Index < _sourceCursor.Schema.Count)
            return _sourceCursor.GetGetter<TValue>(_sourceCursor.Schema[column.Index]);

        throw new ArgumentOutOfRangeException(nameof(column), $"Column index {column.Index} is out of range.");
    }

    public override ValueGetter<DataViewRowId> GetIdGetter()
        => _sourceCursor.GetIdGetter();

    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
                _sourceCursor.Dispose();
            _disposed = true;
        }
        base.Dispose(disposing);
    }
}

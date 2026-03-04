using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.ImageInference.Onnx.Detection;

/// <summary>
/// Cursor that lazily performs object detection inference per row,
/// reading MLImage from the source and producing DetectedObjects_Boxes + DetectedObjects_Count columns.
/// Boxes are flattened as [x, y, w, h, classId, score, ...] with 6 values per detection.
/// </summary>
internal sealed class DetectionCursor : DataViewRowCursor
{
    private readonly DetectionDataView _parent;
    private readonly DataViewRowCursor _sourceCursor;
    private readonly OnnxObjectDetectionTransformer _transformer;
    private readonly DataViewSchema.Column? _inputCol;

    private float[] _boxes = [];
    private int _count;
    private bool _disposed;

    public DetectionCursor(
        DetectionDataView parent,
        DataViewRowCursor sourceCursor,
        OnnxObjectDetectionTransformer transformer,
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
        if (column.Index == _parent.BoxesIndex || column.Index == _parent.CountIndex)
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

            var detections = _transformer.Detect(image);
            _count = detections.Length;

            // Flatten: 6 values per detection [x, y, w, h, classId, score]
            _boxes = new float[_count * 6];
            for (int i = 0; i < _count; i++)
            {
                int offset = i * 6;
                _boxes[offset] = detections[i].X;
                _boxes[offset + 1] = detections[i].Y;
                _boxes[offset + 2] = detections[i].Width;
                _boxes[offset + 3] = detections[i].Height;
                _boxes[offset + 4] = detections[i].ClassId;
                _boxes[offset + 5] = detections[i].Score;
            }
        }
        else
        {
            _boxes = [];
            _count = 0;
        }

        return true;
    }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        if (column.Index == _parent.BoxesIndex)
        {
            ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
                value = new VBuffer<float>(_boxes.Length, _boxes);
            return (ValueGetter<TValue>)(Delegate)getter;
        }

        if (column.Index == _parent.CountIndex)
        {
            ValueGetter<int> getter = (ref int value) => value = _count;
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

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
    private readonly int _batchSize;

    private float[] _boxes = [];
    private int _count;
    private bool _disposed;

    // Lookahead batching state
    private List<(float[] Boxes, int Count)> _batchResults = new();
    private int _batchIndex = -1;
    private bool _inputExhausted;
    private long _position = -1;

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
        _batchSize = transformer.Options.BatchSize;
    }

    public override DataViewSchema Schema => _parent.Schema;

    public override long Position => _position;

    public override long Batch => 0;

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
        _batchIndex++;

        if (_batchResults.Count == 0 || _batchIndex >= _batchResults.Count)
        {
            if (_inputExhausted)
                return false;
            if (!FillNextBatch())
                return false;
        }

        _boxes = _batchResults[_batchIndex].Boxes;
        _count = _batchResults[_batchIndex].Count;
        _position++;

        return true;
    }

    private bool FillNextBatch()
    {
        _batchResults.Clear();
        _batchIndex = 0;

        var images = new List<MLImage>();

        for (int i = 0; i < _batchSize; i++)
        {
            if (!_sourceCursor.MoveNext())
            {
                _inputExhausted = true;
                break;
            }

            if (_inputCol.HasValue)
            {
                MLImage image = null!;
                var getter = _sourceCursor.GetGetter<MLImage>(_inputCol.Value);
                getter(ref image);
                images.Add(image);
            }
            else
            {
                _batchResults.Add((Array.Empty<float>(), 0));
            }
        }

        if (images.Count > 0)
        {
            var batchResults = _transformer.DetectBatch(images);

            for (int i = 0; i < batchResults.Length; i++)
            {
                var detections = batchResults[i];
                var count = detections.Length;

                // Flatten: 6 values per detection [x, y, w, h, classId, score]
                var boxes = new float[count * 6];
                for (int j = 0; j < count; j++)
                {
                    int offset = j * 6;
                    boxes[offset] = detections[j].X;
                    boxes[offset + 1] = detections[j].Y;
                    boxes[offset + 2] = detections[j].Width;
                    boxes[offset + 3] = detections[j].Height;
                    boxes[offset + 4] = detections[j].ClassId;
                    boxes[offset + 5] = detections[j].Score;
                }

                _batchResults.Add((boxes, count));
            }
        }

        return _batchResults.Count > 0;
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

using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.ImageInference.Onnx.ZeroShot;

/// <summary>
/// Cursor that lazily performs zero-shot image classification inference per row,
/// reading MLImage from the source and producing PredictedLabel + Probability columns.
/// </summary>
internal sealed class ZeroShotCursor : DataViewRowCursor
{
    private readonly ZeroShotDataView _parent;
    private readonly DataViewRowCursor _sourceCursor;
    private readonly OnnxZeroShotImageClassificationTransformer _transformer;
    private readonly DataViewSchema.Column? _inputCol;
    private readonly int _batchSize;

    private string _predictedLabel = string.Empty;
    private float[] _probabilities = [];
    private bool _disposed;

    // Lookahead batching state
    private List<(string Label, float[] Probabilities)> _batchResults = new();
    private int _batchIndex = -1;
    private bool _inputExhausted;
    private long _position = -1;

    public ZeroShotCursor(
        ZeroShotDataView parent,
        DataViewRowCursor sourceCursor,
        OnnxZeroShotImageClassificationTransformer transformer,
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
        if (column.Index == _parent.PredictedLabelIndex || column.Index == _parent.ProbabilityIndex)
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

        _predictedLabel = _batchResults[_batchIndex].Label;
        _probabilities = _batchResults[_batchIndex].Probabilities;
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
                _batchResults.Add((string.Empty, Array.Empty<float>()));
            }
        }

        if (images.Count > 0)
        {
            var batchResults = _transformer.ClassifyBatch(images);

            for (int i = 0; i < batchResults.Length; i++)
            {
                var results = batchResults[i];
                var label = results.Length > 0 ? results[0].Label : string.Empty;
                var probs = new float[results.Length];
                for (int j = 0; j < results.Length; j++)
                    probs[j] = results[j].Probability;
                _batchResults.Add((label, probs));
            }
        }

        return _batchResults.Count > 0;
    }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        if (column.Index == _parent.PredictedLabelIndex)
        {
            ValueGetter<ReadOnlyMemory<char>> getter = (ref ReadOnlyMemory<char> value) =>
                value = _predictedLabel.AsMemory();
            return (ValueGetter<TValue>)(Delegate)getter;
        }

        if (column.Index == _parent.ProbabilityIndex)
        {
            ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
                value = new VBuffer<float>(_probabilities.Length, _probabilities);
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

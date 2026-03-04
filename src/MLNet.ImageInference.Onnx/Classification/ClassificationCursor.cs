using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.ImageInference.Onnx.Classification;

/// <summary>
/// Cursor that lazily performs image classification inference per row,
/// reading MLImage from the source and producing PredictedLabel + Probability columns.
/// </summary>
internal sealed class ClassificationCursor : DataViewRowCursor
{
    private readonly ClassificationDataView _parent;
    private readonly DataViewRowCursor _sourceCursor;
    private readonly OnnxImageClassificationTransformer _transformer;
    private readonly DataViewSchema.Column? _inputCol;

    private string _predictedLabel = string.Empty;
    private float[] _probabilities = [];
    private bool _disposed;

    public ClassificationCursor(
        ClassificationDataView parent,
        DataViewRowCursor sourceCursor,
        OnnxImageClassificationTransformer transformer,
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
        if (column.Index == _parent.PredictedLabelIndex || column.Index == _parent.ProbabilityIndex)
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

            var results = _transformer.Classify(image);

            _predictedLabel = results.Length > 0 ? results[0].Label : string.Empty;
            _probabilities = new float[results.Length];
            for (int i = 0; i < results.Length; i++)
                _probabilities[i] = results[i].Probability;
        }
        else
        {
            _predictedLabel = string.Empty;
            _probabilities = [];
        }

        return true;
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

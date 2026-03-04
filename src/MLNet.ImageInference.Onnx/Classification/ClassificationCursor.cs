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
    private readonly int _batchSize;

    private string _predictedLabel = string.Empty;
    private float[] _probabilities = [];
    private bool _disposed;

    // Lookahead batching state
    private List<(string Label, float[] Probabilities)> _batchResults = new();
    private List<MLImage> _batchImages = new();
    private readonly Dictionary<int, IColumnCache> _passthroughCaches = new();
    private int _batchIndex = -1;
    private bool _inputExhausted;
    private long _position = -1;

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
        _batchImages.Clear();
        foreach (var cache in _passthroughCaches.Values)
            cache.Clear();
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
                _batchImages.Add(image);
            }
            else
            {
                // No input column — produce empty results for this row
                _batchResults.Add((string.Empty, Array.Empty<float>()));
            }

            // Cache any registered passthrough columns
            foreach (var cache in _passthroughCaches.Values)
                cache.ReadCurrentRow();
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

        // Passthrough: serve cached image for the input column
        if (column.Index < _sourceCursor.Schema.Count)
        {
            if (_inputCol.HasValue && column.Index == _inputCol.Value.Index)
            {
                ValueGetter<MLImage> imgGetter = (ref MLImage value) => value = _batchImages[_batchIndex];
                return (ValueGetter<TValue>)(Delegate)imgGetter;
            }

            // Lazy-register passthrough cache for other source columns
            if (!_passthroughCaches.TryGetValue(column.Index, out _))
            {
                var sourceGetter = _sourceCursor.GetGetter<TValue>(_sourceCursor.Schema[column.Index]);
                _passthroughCaches[column.Index] = new PassthroughCache<TValue>(sourceGetter);
            }

            var cache = (PassthroughCache<TValue>)_passthroughCaches[column.Index];
            return (ref TValue value) =>
            {
                if (_batchIndex >= 0 && _batchIndex < cache.Values.Count)
                    value = cache.Values[_batchIndex];
            };
        }

        throw new ArgumentOutOfRangeException(nameof(column), $"Column index {column.Index} is out of range.");
    }

    public override ValueGetter<DataViewRowId> GetIdGetter()
        => (ref DataViewRowId id) => id = new DataViewRowId((ulong)_position, 0);

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

    /// <summary>Type-erased interface for passthrough column value caching.</summary>
    private interface IColumnCache
    {
        void ReadCurrentRow();
        void Clear();
    }

    /// <summary>Typed cache that reads and stores values from a source cursor getter.</summary>
    private sealed class PassthroughCache<T> : IColumnCache
    {
        private readonly ValueGetter<T> _sourceGetter;
        public readonly List<T> Values = new();

        public PassthroughCache(ValueGetter<T> sourceGetter)
        {
            _sourceGetter = sourceGetter;
        }

        public void ReadCurrentRow()
        {
            T val = default!;
            _sourceGetter(ref val);
            Values.Add(val);
        }

        public void Clear() => Values.Clear();
    }
}

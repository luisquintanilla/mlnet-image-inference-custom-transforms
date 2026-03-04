using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.ImageInference.Onnx.Embeddings;

/// <summary>
/// Cursor that lazily performs image embedding inference per row,
/// reading MLImage from the source and producing an Embedding vector column.
/// Uses lookahead batching to amortize ONNX inference overhead.
/// </summary>
internal sealed class EmbeddingCursor : DataViewRowCursor
{
    private readonly EmbeddingDataView _parent;
    private readonly DataViewRowCursor _sourceCursor;
    private readonly OnnxImageEmbeddingTransformer _transformer;
    private readonly DataViewSchema.Column? _inputCol;
    private readonly int _batchSize;

    private float[] _embedding = [];
    private bool _disposed;

    // Lookahead batching state
    private List<float[]> _batchResults = new();
    private int _batchIndex = -1;
    private bool _inputExhausted;
    private long _position = -1;

    public EmbeddingCursor(
        EmbeddingDataView parent,
        DataViewRowCursor sourceCursor,
        OnnxImageEmbeddingTransformer transformer,
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
        if (column.Index == _parent.EmbeddingIndex)
            return true;
        if (column.Index < _sourceCursor.Schema.Count)
            return _sourceCursor.IsColumnActive(_sourceCursor.Schema[column.Index]);
        return false;
    }

    public override bool MoveNext()
    {
        _batchIndex++;

        if (_batchIndex < _batchResults.Count)
        {
            _embedding = _batchResults[_batchIndex];
            _position++;
            return true;
        }

        if (_inputExhausted)
            return false;

        FillNextBatch();

        if (_batchResults.Count == 0)
            return false;

        _embedding = _batchResults[_batchIndex];
        _position++;
        return true;
    }

    private void FillNextBatch()
    {
        _batchResults.Clear();
        _batchIndex = 0;

        if (!_inputCol.HasValue)
        {
            // No input column — produce zero embeddings one at a time
            if (_sourceCursor.MoveNext())
            {
                _batchResults.Add(new float[_transformer.EmbeddingDimension]);
            }
            else
            {
                _inputExhausted = true;
            }
            return;
        }

        var images = new List<MLImage>();
        var getter = _sourceCursor.GetGetter<MLImage>(_inputCol.Value);

        for (int i = 0; i < _batchSize; i++)
        {
            if (!_sourceCursor.MoveNext())
            {
                _inputExhausted = true;
                break;
            }

            MLImage image = null!;
            getter(ref image);
            images.Add(image);
        }

        if (images.Count > 0)
        {
            var embeddings = _transformer.GenerateEmbeddingBatch(images);
            _batchResults.AddRange(embeddings);
        }
    }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        if (column.Index == _parent.EmbeddingIndex)
        {
            ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
                value = new VBuffer<float>(_embedding.Length, _embedding);
            return (ValueGetter<TValue>)(Delegate)getter;
        }

        // Passthrough to source cursor
        if (column.Index < _sourceCursor.Schema.Count)
            return _sourceCursor.GetGetter<TValue>(_sourceCursor.Schema[column.Index]);

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
}

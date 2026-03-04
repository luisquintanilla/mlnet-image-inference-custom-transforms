using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.ImageInference.Onnx.Embeddings;

/// <summary>
/// Cursor that lazily performs image embedding inference per row,
/// reading MLImage from the source and producing an Embedding vector column.
/// </summary>
internal sealed class EmbeddingCursor : DataViewRowCursor
{
    private readonly EmbeddingDataView _parent;
    private readonly DataViewRowCursor _sourceCursor;
    private readonly OnnxImageEmbeddingTransformer _transformer;
    private readonly DataViewSchema.Column? _inputCol;

    private float[] _embedding = [];
    private bool _disposed;

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
    }

    public override DataViewSchema Schema => _parent.Schema;

    public override long Position => _sourceCursor.Position;

    public override long Batch => _sourceCursor.Batch;

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
        if (!_sourceCursor.MoveNext())
            return false;

        if (_inputCol.HasValue)
        {
            MLImage image = null!;
            var getter = _sourceCursor.GetGetter<MLImage>(_inputCol.Value);
            getter(ref image);

            _embedding = _transformer.GenerateEmbedding(image);
        }
        else
        {
            _embedding = new float[_transformer.EmbeddingDimension];
        }

        return true;
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

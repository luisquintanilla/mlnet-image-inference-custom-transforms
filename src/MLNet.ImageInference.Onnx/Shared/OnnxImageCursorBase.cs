using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.ImageInference.Onnx.Shared;

/// <summary>
/// Base class for all ONNX image inference cursors with lookahead batching.
/// Derived types only need to implement batch inference and result extraction.
/// </summary>
/// <typeparam name="TResult">The per-row result type produced by inference (e.g., tuple, array, mask).</typeparam>
internal abstract class OnnxImageCursorBase<TResult> : DataViewRowCursor
{
    private readonly OnnxImageDataViewBase _parent;
    protected readonly DataViewRowCursor SourceCursor;
    private readonly DataViewSchema.Column? _inputCol;
    private readonly int _batchSize;

    private bool _disposed;
    private bool _inputExhausted;

    protected List<TResult> BatchResults = new();
    protected int BatchIndex = -1;
    protected long CurrentPosition = -1;

    protected OnnxImageCursorBase(
        OnnxImageDataViewBase parent,
        DataViewRowCursor sourceCursor,
        DataViewSchema.Column? inputCol,
        int batchSize)
    {
        _parent = parent;
        SourceCursor = sourceCursor;
        _inputCol = inputCol;
        _batchSize = batchSize;
    }

    public override DataViewSchema Schema => _parent.Schema;

    public override long Position => CurrentPosition;

    public override long Batch => 0;

    /// <summary>Return true if the column at the given index is an output column produced by this cursor.</summary>
    protected abstract bool IsOutputColumn(int columnIndex);

    /// <summary>Run batch inference on a list of images and add results to <see cref="BatchResults"/>.</summary>
    protected abstract void RunBatchInference(List<MLImage> images);

    /// <summary>Add a default/empty result when no input column is found.</summary>
    protected abstract TResult CreateEmptyResult();

    /// <summary>Extract the current result into task-specific fields for GetGetter to return.</summary>
    protected abstract void ExtractCurrentResult(TResult result);

    /// <summary>Create a ValueGetter for a task-specific output column.</summary>
    protected abstract ValueGetter<TValue> CreateOutputGetter<TValue>(DataViewSchema.Column column);

    public override bool IsColumnActive(DataViewSchema.Column column)
    {
        if (IsOutputColumn(column.Index))
            return true;
        if (column.Index < SourceCursor.Schema.Count)
            return SourceCursor.IsColumnActive(SourceCursor.Schema[column.Index]);
        return false;
    }

    public override bool MoveNext()
    {
        BatchIndex++;

        if (BatchResults.Count == 0 || BatchIndex >= BatchResults.Count)
        {
            if (_inputExhausted)
                return false;
            if (!FillNextBatch())
                return false;
        }

        ExtractCurrentResult(BatchResults[BatchIndex]);
        CurrentPosition++;
        return true;
    }

    private bool FillNextBatch()
    {
        BatchResults.Clear();
        OnBatchClear();
        BatchIndex = 0;

        var images = new List<MLImage>();

        for (int i = 0; i < _batchSize; i++)
        {
            if (!SourceCursor.MoveNext())
            {
                _inputExhausted = true;
                break;
            }

            if (_inputCol.HasValue)
            {
                MLImage image = null!;
                var getter = SourceCursor.GetGetter<MLImage>(_inputCol.Value);
                getter(ref image);
                images.Add(image);
                OnImageRead(image);
            }
            else
            {
                BatchResults.Add(CreateEmptyResult());
            }

            OnSourceRowRead();
        }

        if (images.Count > 0)
            RunBatchInference(images);

        return BatchResults.Count > 0;
    }

    /// <summary>Called when a new batch starts and caches should be cleared. Override for passthrough caching.</summary>
    protected virtual void OnBatchClear() { }

    /// <summary>Called after each image is read from source. Override to cache the image.</summary>
    protected virtual void OnImageRead(MLImage image) { }

    /// <summary>Called after each source row is read. Override for passthrough column caching.</summary>
    protected virtual void OnSourceRowRead() { }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        if (IsOutputColumn(column.Index))
            return CreateOutputGetter<TValue>(column);

        // Passthrough to source cursor
        if (column.Index < SourceCursor.Schema.Count)
            return SourceCursor.GetGetter<TValue>(SourceCursor.Schema[column.Index]);

        throw new ArgumentOutOfRangeException(nameof(column), $"Column index {column.Index} is out of range.");
    }

    public override ValueGetter<DataViewRowId> GetIdGetter()
        => (ref DataViewRowId id) => id = new DataViewRowId((ulong)CurrentPosition, 0);

    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
                SourceCursor.Dispose();
            _disposed = true;
        }
        base.Dispose(disposing);
    }
}

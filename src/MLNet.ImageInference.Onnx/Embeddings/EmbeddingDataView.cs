using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.ImageInference.Onnx.Embeddings;

/// <summary>
/// Custom IDataView that wraps a source IDataView and appends an Embedding vector column
/// by running ONNX inference lazily per row.
/// </summary>
internal sealed class EmbeddingDataView : IDataView
{
    private readonly IDataView _source;
    private readonly OnnxImageEmbeddingTransformer _transformer;
    private readonly DataViewSchema _schema;
    private readonly int _embeddingIndex;

    public EmbeddingDataView(IDataView source, OnnxImageEmbeddingTransformer transformer)
    {
        _source = source;
        _transformer = transformer;

        var builder = new DataViewSchema.Builder();
        for (int i = 0; i < source.Schema.Count; i++)
            builder.AddColumn(source.Schema[i].Name, source.Schema[i].Type, source.Schema[i].Annotations);

        _embeddingIndex = source.Schema.Count;
        builder.AddColumn(transformer.Options.EmbeddingColumnName,
            new VectorDataViewType(NumberDataViewType.Single, transformer.EmbeddingDimension));

        _schema = builder.ToSchema();
    }

    public DataViewSchema Schema => _schema;

    public bool CanShuffle => false;

    public long? GetRowCount() => null;

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        // Determine which source columns are needed (passthrough)
        var sourceColumnsNeeded = columnsNeeded
            .Where(c => c.Index < _source.Schema.Count)
            .Select(c => _source.Schema[c.Index]);

        // Always request the image input column so we can run inference
        var inputCol = _source.Schema.GetColumnOrNull(_transformer.Options.InputColumnName);
        if (inputCol.HasValue)
            sourceColumnsNeeded = sourceColumnsNeeded.Append(inputCol.Value);

        var sourceCursor = _source.GetRowCursor(sourceColumnsNeeded.Distinct(), rand);
        return new EmbeddingCursor(this, sourceCursor, _transformer, inputCol);
    }

    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        // Sequential inference — return a single cursor
        return [GetRowCursor(columnsNeeded, rand)];
    }

    internal int EmbeddingIndex => _embeddingIndex;
}

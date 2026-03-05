using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Embeddings;

/// <summary>
/// Custom IDataView that wraps a source IDataView and appends an Embedding vector column
/// by running ONNX inference lazily per row.
/// </summary>
internal sealed class EmbeddingDataView : OnnxImageDataViewBase
{
    private readonly OnnxImageEmbeddingTransformer _transformer;

    public EmbeddingDataView(IDataView source, OnnxImageEmbeddingTransformer transformer)
        : base(source, transformer.Options.InputColumnName, (builder, nextIndex) =>
        {
            builder.AddColumn(transformer.Options.EmbeddingColumnName,
                new VectorDataViewType(NumberDataViewType.Single, transformer.EmbeddingDimension));
        })
    {
        _transformer = transformer;
        EmbeddingIndex = source.Schema.Count;
    }

    protected override DataViewRowCursor CreateCursor(
        DataViewRowCursor sourceCursor,
        DataViewSchema.Column? inputCol)
    {
        return new EmbeddingCursor(this, sourceCursor, _transformer, inputCol);
    }

    internal int EmbeddingIndex { get; private set; }
}
